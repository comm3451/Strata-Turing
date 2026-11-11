#!/usr/bin/env python3
"""Copy this node's IQ3_S/BF16/DFlash2 config with its serving speed settings.

Keeps credentials, model files, network settings and the engine executable unchanged.
Run through turing-run.sh; the output must be a new file in this checkout.
See docs/TURING.md for the measurements and capacity limits.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from qfn_turing_profile import ROOT, set_arg


def make_config(source):
    cfg = copy.deepcopy(source)
    # Owner requirements apply even when the source config enables a shortcut.
    cfg.setdefault('env', {}).update(dict.fromkeys((
        'STRATA_ROUTE_PRIOR', 'STRATA_ROUTE_TAIL_SKIP', 'STRATA_KVA',
        'STRATA_PREFILL_CPU_SHARE', 'STRATA_HC_Q8'), '0'))
    cfg['env'].update(STRATA_ADAPT_FETCH='2', STRATA_ADAPT_OVERLAP='0')
    effective = cfg['args'] + cfg.get('user_args', [])
    if '--resident-experts' in effective:
        raise ValueError('the measured fetch-admission profile requires the full expert arena')

    def value(flag):
        positions = [i for i, item in enumerate(effective) if item == flag]
        return effective[positions[-1] + 1] if positions else None

    native = (value('--native') or '').lower()
    ngram = (value('--ple-gguf') or '').lower()
    if not ('flash-next' in native and 'iq3_s' in native and 'bf16' in ngram):
        raise ValueError('requires the existing QFN IQ3_S model and BF16 N-gram table')
    if value('--max-context') != '262144' or cfg.get('parallel', 1) != 1:
        raise ValueError('requires 262144 context and one serving slot')
    if (cfg.get('drafting') != 'dflash2' or value('--spec') != '6'
            or value('--draft-conf') != '0.5'):
        raise ValueError('requires the measured DFlash2 window 6 / confidence 0.5 configuration')
    settings = {
        '--prompt-cache': '2',
        '--conversation-cache-mib': '1024',
        '--conversation-cache-min-free-mib': '3072',
        '--conversation-cache-min-tokens': '4096',
        '--conversation-cache-park-gain': '4096',
        '--vram-reserve-mib': '512',
    }
    if any(flag in cfg.get('user_args', []) for flag in settings):
        raise ValueError('remove conflicting cache/VRAM overrides from user_args first')
    for flag, setting in settings.items():
        cfg['args'] = set_arg(cfg['args'], flag, setting)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if os.environ.get('STRATA_TURING_ISOLATED') != '1':
        parser.error('run through tools/turing-run.sh')
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT):
        parser.error('out must be inside this checkout')
    try:
        cfg = make_config(json.loads(Path(args.source).read_text()))
    except (ValueError, KeyError, IndexError) as error:
        parser.error(str(error))
    cfg.update(cwd=str(ROOT), log=str(out.with_suffix('.engine.log')))
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(cfg, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(out)


if __name__ == '__main__':
    main()
