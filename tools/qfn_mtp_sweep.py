#!/usr/bin/env python3
"""Explore MTP windows/confidence on one QFN engine; confirm winners with HTTP A/B.

The running expert cache adapts between requests. These observations select
candidates; they are not independent paired proof of a speedup.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--server', required=True, help='server.py of the measured control version')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases', required=True, help='JSON [name, prompt] pairs')
    ap.add_argument('--windows', nargs='+', type=int, default=[4, 6, 8])
    ap.add_argument('--thresholds', nargs='+', type=float, default=[0.5, 0.7])
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--max-new', type=int, default=512)
    args = ap.parse_args()
    if os.environ.get('STRATA_TURING_ISOLATED') != '1':
        ap.error('run through tools/turing-run.sh')
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT) or min(args.rounds, args.max_new) <= 0:
        ap.error('out must be inside the checkout; rounds and max-new must be positive')
    if any(w < 2 or w > 8 for w in args.windows) or any(not 0 <= p <= 1 for p in args.thresholds):
        ap.error('windows must be 2..8 and thresholds 0..1')
    base = json.loads(Path(args.config).read_text())
    if base.get('api_key') or not Path(base['exe']).is_absolute():
        ap.error('use a local benchmark config without an API key and with an absolute executable')
    out.mkdir(parents=True, exist_ok=False)
    server = Path(args.server).resolve()
    sys.path.insert(0, str(server.parent))
    srv = importlib.import_module(server.stem)
    if Path(srv.__file__).resolve() != server:
        raise RuntimeError('wrong server module imported')
    from calibrate import chat_ids, with_arg
    import strata_tokenizer as st
    tpath = Path(base['tokenizer'])
    vocab = json.loads((tpath / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    tok = st.Tokenizer(tokens, (tpath / 'merges.txt').read_text().split('\n'),
                       json.loads((tpath / 'token_type.json').read_text()))
    cases = json.loads(Path(args.cases).read_text())
    if not cases or any(not isinstance(x, list) or len(x) != 2 or
                        not all(isinstance(v, str) and v for v in x) for x in cases):
        ap.error('cases must be nonempty [name, prompt] pairs')
    encoded = [(name, chat_ids(tok, prompt)) for name, prompt in cases]
    metadata = {'config': base, 'arguments': vars(args), 'server_sha256': hashlib.sha256(server.read_bytes()).hexdigest(),
                'engine_sha256': hashlib.sha256(Path(base['exe']).read_bytes()).hexdigest(),
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'prompts': cases, 'note': 'Exploratory, adaptive cache state; confirm selected settings independently.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    with (out / 'results.jsonl').open('w') as result:
        for window in args.windows:
            cfg = copy.deepcopy(base)
            cfg['args'] = with_arg(cfg['args'], '--spec', str(window))
            cfg['log'] = str(out / f'window-{window}-engine.log')
            engine = srv.StrataEngine(cfg['exe'], srv.engine_args(cfg), cwd=str(ROOT),
                                      log=cfg['log'], env=srv.child_env(cfg))
            try:
                list(engine.generate(chat_ids(tok, 'Say hello.'), 16, {'temperature': 0}, threading.Event()))
                for rnd in range(args.rounds):
                    thresholds = args.thresholds if rnd % 2 == 0 else args.thresholds[::-1]
                    for threshold in thresholds:
                        for name, ids in encoded:
                            sampling = {'temperature': 0, 'strata_checkpoint': False,
                                        'strata_tune': {'spec_min_p': threshold}}
                            cancel = threading.Event()
                            timer = threading.Timer(300, cancel.set)
                            timer.start()
                            started = time.perf_counter()
                            try:
                                answer = [t for t in engine.generate(ids, args.max_new, sampling, cancel) if t is not None]
                            finally:
                                timer.cancel()
                            last = dict(engine.last or {})
                            if cancel.is_set() or engine.last_err or not answer or not last.get('decode_ms'):
                                raise RuntimeError(f'{name}: cancelled, empty or failed: {engine.last_err}')
                            if last.get('reused', 0):
                                raise RuntimeError('prompt cache was reused; disable it in the benchmark config')
                            row = {'window': window, 'threshold': threshold, 'round': rnd, 'case': name,
                                   'input_tokens': len(ids), 'output_tokens': len(answer), 'engine': last,
                                   'decode_tps': len(answer) * 1000 / last['decode_ms'],
                                   'total_s': time.perf_counter() - started, 'token_ids': answer, 'text': tok.decode(answer)}
                            result.write(json.dumps(row, ensure_ascii=False) + '\n')
                            result.flush()
                            print(json.dumps({k: row[k] for k in ('window', 'threshold', 'round', 'case', 'output_tokens', 'decode_tps')}), flush=True)
            finally:
                engine.close()


if __name__ == '__main__':
    main()
