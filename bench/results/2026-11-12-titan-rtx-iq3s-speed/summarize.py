"""Recompute the final-profile summaries from recorded responses and counters."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
FLAGS = ('STRATA_ROUTE_PRIOR', 'STRATA_ROUTE_TAIL_SKIP', 'STRATA_KVA',
         'STRATA_PREFILL_CPU_SHARE', 'STRATA_HC_Q8')


def read(name):
    return json.loads((ROOT / name).read_text())


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        assert row['metrics']['ok']
        groups[row['case'], row['arm']].append(row['metrics'])
    result = {}
    for (case, arm), values in groups.items():
        assert len(values) == 3, (case, arm, len(values))
        result.setdefault(case, {})[arm] = {
            field: {'median': median(v[field] for v in values),
                    'min': min(v[field] for v in values),
                    'max': max(v[field] for v in values)} for field in fields}
    return result


def main():
    perf = read('confirm-results.json')
    cache = read('cache-confirm-results.json')
    assert len(perf) == 30 and len(cache) == 54
    assert all(row['metrics']['reused_tokens'] == 0 for row in perf)
    assert all(row['correct_values'] for row in cache)
    meta = read('confirm-metadata.json')
    for cfg in (meta['config'], meta['candidate_config']):
        assert all(cfg['env'][flag] == '0' for flag in FLAGS)
    fields = ('decode_tps', 'prompt_tps', 'ttft_s', 'total_s', 'generated_tokens')
    performance = grouped(perf, fields)
    for values in performance.values():
        values['candidate_vs_baseline_percent'] = {
            field: 100 * (values['candidate'][field]['median'] /
                          values['baseline'][field]['median'] - 1) for field in fields}
    near = read('near-continuity-results.json')
    assert len(near) == 7 and all(row['correct_values'] for row in near)
    queued = {}
    for arm in ('baseline', 'candidate'):
        orders = []
        for rnd in range(3):
            pair = [row['metrics'] for row in cache if row['arm'] == arm
                    and row['round'] == rnd and row['concurrent']]
            assert len(pair) == 2
            orders.append(sorted(pair, key=lambda value: value['ttft_s']))
        queued[arm] = {
            label: {field: median(pair[index][field] for pair in orders)
                    for field in ('ttft_s', 'total_s')}
            for index, label in enumerate(('first', 'second'))}
    result = {
        'required_flags': dict.fromkeys(FLAGS, '0'),
        'engine_sha256': meta['engine_sha256'],
        'performance': performance,
        'cache': grouped(cache, ('ttft_s', 'total_s', 'reused_tokens', 'read_tokens')),
        'queued_by_response_order': queued,
        'near_continuity': {row['case']: row['metrics'] for row in near},
        'near_code': read('near-code-results.json')[0]['metrics'],
        'answer_checks': read('answer-checks.json'),
        'memory': read('memory-summary.json'),
    }
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'performance_requests': len(perf), 'cache_requests': len(cache),
                      'near_continuity_requests': len(near), 'summary': str(ROOT / 'summary.json')}))


if __name__ == '__main__':
    main()
