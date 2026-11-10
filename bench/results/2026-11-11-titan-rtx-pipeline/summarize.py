"""Recompute measurements from this report's recorded responses and samples."""
import json
from pathlib import Path
import statistics

p = Path(__file__).resolve().parent
out = {}
for name in ('exact-pilot', 'chunk32-pilot', 'overlap-pilot', 'scheduling-pilot',
             'cost-pilot', 'final-ab', 'near-256k', 'near-code'):
    rows = json.loads((p/(name+'-results.json')).read_text())
    result = {}
    for case in sorted({r['case'] for r in rows}):
        entry = {}
        for arm in sorted({r['arm'] for r in rows}):
            sel = [r for r in rows if r['case'] == case and r['arm'] == arm]
            assert sel and all(r['metrics']['ok'] and r['metrics']['reused_tokens'] == 0 for r in sel)
            entry[arm] = {'requests': len(sel),
                          'prompt_tokens': sorted({r['metrics']['prompt_tokens'] for r in sel}),
                          'generated_tokens': [r['metrics']['generated_tokens'] for r in sel]}
            for key in ('prompt_tps', 'decode_tps', 'ttft_s', 'total_s'):
                entry[arm][key] = statistics.median(r['metrics'][key] for r in sel)
                entry[arm][key+'_range'] = [min(r['metrics'][key] for r in sel), max(r['metrics'][key] for r in sel)]
        if 'baseline' in entry and 'candidate' in entry:
            entry['change_percent'] = {k: 100*(entry['candidate'][k]/entry['baseline'][k]-1)
                                       for k in ('prompt_tps', 'decode_tps', 'ttft_s', 'total_s')}
            entry['identical_text_pairs'] = sum(
                next(r['response']['text'] for r in rows if r['case'] == case and r['round'] == rnd and r['arm'] == 'baseline') ==
                next(r['response']['text'] for r in rows if r['case'] == case and r['round'] == rnd and r['arm'] == 'candidate')
                for rnd in sorted({r['round'] for r in rows}))
        result[case] = entry
    out[name] = result
for f in sorted(p.glob('*-memory.jsonl')):
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    gpu = [list(map(float, r['gpu'].split(','))) for r in rows]
    out[f.stem] = {'samples': len(rows), 'max_gpu_used_mib': max(r[0] for r in gpu),
                   'min_gpu_free_mib': min(r[1] for r in gpu), 'max_gpu_temperature_c': max(r[2] for r in gpu),
                   'min_host_available_gib': min(r['MemAvailable'] for r in rows)/1048576,
                   'initial_swap_used_mib': (rows[0]['SwapTotal']-rows[0]['SwapFree'])/1024,
                   'max_swap_used_mib': max(r['SwapTotal']-r['SwapFree'] for r in rows)/1024}
mtp = [json.loads(x) for x in (p/'mtp-sweep-results.jsonl').read_text().splitlines()]
out['mtp_second_round'] = []
for row in mtp:
    if row['round'] == 1:
        out['mtp_second_round'].append({k: row[k] for k in ('window', 'threshold', 'case', 'output_tokens', 'decode_tps')})
print(json.dumps(out, indent=2))
