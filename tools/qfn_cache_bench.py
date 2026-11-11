#!/usr/bin/env python3
"""Compare repeated QFN turns, prefix edits and queued clients on local configs.

This measures client latency and real prefix reuse separately from cold prefill.
Synthetic retrieval checks are not a broad long-context quality evaluation.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from qfn_turing_bench import ROOT, local_path, retrieval_prompt, stop, wait_ready
from strata_bench import chat, digest, http_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate-config')
    parser.add_argument('--server', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--words', type=int, default=30000)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--continuity-only', action='store_true',
                        help='long-context check: omit unrelated conversation and prefix replacement')
    parser.add_argument('--prime-parked', action='store_true',
                        help='first seed an independent 30K-word chat to test retained-cache memory pressure')
    args = parser.parse_args()
    if os.environ.get('STRATA_TURING_ISOLATED') != '1':
        parser.error('run through tools/turing-run.sh')
    if min(args.words, args.rounds) <= 0:
        parser.error('words and rounds must be positive')
    configs = {'baseline': json.loads(local_path(args.config).read_text())}
    if args.candidate_config:
        configs['candidate'] = json.loads(local_path(args.candidate_config).read_text())
    for cfg in configs.values():
        if cfg.get('api_key') or not Path(cfg['exe']).is_file():
            parser.error('use a local benchmark config with an existing executable and no API key')
        for flag in ('STRATA_ROUTE_PRIOR', 'STRATA_ROUTE_TAIL_SKIP', 'STRATA_KVA',
                     'STRATA_PREFILL_CPU_SHARE', 'STRATA_HC_Q8'):
            if cfg.get('env', {}).get(flag) != '0':
                parser.error(f'set {flag}=0 explicitly in benchmark configs')
    if len({cfg['exe'] for cfg in configs.values()}) != 1:
        parser.error('both arms must use the same executable')
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', args.port))
    out = local_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    server = Path(args.server).resolve()
    document = retrieval_prompt(args.words)
    messages = [{'role': 'user', 'content': document}]
    changed = document.replace('ALPHA has access code 729183', 'ALPHA has access code 654321', 1)
    assert changed != document
    metadata = {'arguments': vars(args), 'configs': configs,
        'engine_sha256': hashlib.sha256(Path(configs['baseline']['exe']).read_bytes()).hexdigest(),
        'server_sha256': hashlib.sha256(server.read_bytes()).hexdigest(),
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'document_sha256': hashlib.sha256(document.encode()).hexdigest(),
        'note': 'Synthetic multi-turn retrieval; one serving slot; two clients are queued, not two resident contexts.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    url = f'http://127.0.0.1:{args.port}'
    with (out / 'results.jsonl').open('w') as results:
        for rnd in range(args.rounds):
            order = list(configs)
            if rnd % 2:
                order.reverse()
            for arm in order:
                label = f'r{rnd}-{arm}'
                cfg = copy.deepcopy(configs[arm])
                cfg.update(cwd=str(ROOT), host='127.0.0.1', port=args.port, open_browser=False,
                           log=str(out / f'{label}-engine.log'))
                path = out / f'{label}.json'
                path.write_text(json.dumps(cfg, indent=2) + '\n')
                with (out / f'{label}-server.log').open('w') as log:
                    proc = subprocess.Popen([sys.executable, str(server), '--engine', 'strata',
                        '--config', str(path), '--port', str(args.port)], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        wait_ready(url, proc, 600)
                        status = http_json(url, '/v1/status')
                        assert status['context']['max_positions'] == 262144
                        assert status['concurrency']['serving'] == 1
                        warm = chat(url, '', [{'role': 'user', 'content': 'Return only the number 7.'}], 16)
                        assert warm['err'] is None

                        def ask(name, msgs, expected, concurrent=False):
                            response = chat(url, '', msgs, 64, timeout=600)
                            metrics = digest(response)
                            correct = metrics['ok'] and re.findall(r'\d+', response['text']) == expected
                            return {'round': rnd, 'arm': arm, 'case': name, 'concurrent': concurrent,
                                'messages_sha256': hashlib.sha256(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest(),
                                'expected_values': expected, 'correct_values': correct,
                                'metrics': metrics, 'response': response}

                        def record(row):
                            results.write(json.dumps(row, ensure_ascii=False) + '\n')
                            results.flush()
                            print(json.dumps({k: row[k] for k in ('round', 'arm', 'case', 'correct_values', 'metrics')}), flush=True)
                            assert row['correct_values'], f"{label}/{row['case']}: content check failed"

                        if args.prime_parked:
                            seed = [{'role': 'user', 'content': 'Independent cache seed.\n' + retrieval_prompt(30000)}]
                            record(ask('parked-seed', seed, ['729183', '416927', '853204']))
                        first = ask('cold', messages, ['729183', '416927', '853204'])
                        record(first)
                        record(ask('repeat', messages, ['729183', '416927', '853204']))
                        history = messages + [{'role': 'assistant', 'content': first['response']['text']}]
                        beta = history + [{'role': 'user', 'content': 'Return only the access code for BETA.'}]
                        gamma = history + [{'role': 'user', 'content': 'Return only the access code for GAMMA.'}]
                        record(ask('followup', beta, ['416927']))
                        record(ask('edit-last-turn', gamma, ['853204']))
                        if not args.continuity_only:
                            record(ask('unrelated', [{'role': 'user', 'content': 'Return only the number: 19 + 23 = ?'}], ['42']))
                            record(ask('resume', beta, ['416927']))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                            futures = [pool.submit(ask, 'queued-beta', beta, ['416927'], True),
                                       pool.submit(ask, 'queued-gamma', gamma, ['853204'], True)]
                            for future in futures:
                                record(future.result())
                        if not args.continuity_only:
                            record(ask('edit-prefix', [{'role': 'user', 'content': changed}], ['654321', '416927', '853204']))
                    finally:
                        stop(proc)


if __name__ == '__main__':
    main()
