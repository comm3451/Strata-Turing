#!/usr/bin/env python3
"""Give this node's QFN IQ3_XXS/BF16 service a larger host conversation cache.

The 8 GiB budget is an upper bound, not a reservation or an engine maximum.
Parking retains the existing 3 GiB RAM floor. Use tools/turing-run.sh.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from qfn_turing_profile import ROOT, set_arg


def make_config(source):
    cfg = copy.deepcopy(source)
    effective = cfg['args'] + cfg.get('user_args', [])

    def value(flag):
        positions = [i for i, item in enumerate(effective) if item == flag]
        return effective[positions[-1] + 1] if positions else None

    if not ('flash-next' in (value('--native') or '').lower()
            and 'iq3_xxs' in (value('--native') or '').lower()
            and 'bf16' in (value('--ple-gguf') or '').lower()):
        raise ValueError('requires QFN IQ3_XXS with the BF16 N-gram table')
    if value('--max-context') != '262144' or cfg.get('parallel', 1) != 1:
        raise ValueError('requires 262144 context and one serving slot')
    for flag in ('STRATA_ROUTE_PRIOR', 'STRATA_ROUTE_TAIL_SKIP', 'STRATA_KVA',
                 'STRATA_PREFILL_CPU_SHARE', 'STRATA_HC_Q8'):
        if cfg.get('env', {}).get(flag) != '0':
            raise ValueError(f'set {flag}=0 explicitly before changing the cache')
    settings = {'--prompt-cache': '8', '--conversation-cache-mib': '8192',
                '--conversation-cache-slots': '8', '--conversation-cache-min-free-mib': '3072'}
    if any(flag in cfg.get('user_args', []) for flag in settings):
        raise ValueError('remove conflicting cache overrides from user_args first')
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
    out.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(cfg, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(out)


if __name__ == '__main__':
    main()
