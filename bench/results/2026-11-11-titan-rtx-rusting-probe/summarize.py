"""Summarize recorded probes; never equate their different engine algorithms."""
import json,re,statistics
from pathlib import Path

root=Path(__file__).resolve().parent
out={}
for label in ('latest','latest-warm','strata'):
    path=root/f'{label}-results.json'
    if not path.exists():continue
    rows=json.loads(path.read_text())
    assert all(r['metrics']['ok'] for r in rows)
    summary={}
    for case in ('code','korean','4k'):
        selected=[r for r in rows if r['case']==case]
        assert selected and all(r['metrics']['reused_tokens']==0 for r in selected)
        stats={}
        for key in ('ttft_s','client_decode_tps'):
            values=[r['response']['first'] if key=='ttft_s' else
                    (r['metrics']['generated_tokens']-1)/(r['response']['total']-r['response']['first'])
                    for r in selected]
            stats[key]={'median':statistics.median(values),'min':min(values),'max':max(values),'n':len(values)}
        summary[case]=stats
    if label.startswith('latest'):
        # The Rust server emits prefill timing in its log, not the streamed API.
        pp=[float(x) for x in re.findall(r'prompt=4349 tok \(0 cached\).*?prefill ([0-9.]+) tok/s',
                                          (root/f'{label}-server.txt').read_text())]
        assert len(pp)==summary['4k']['ttft_s']['n']
        summary['4k']['engine_prefill_tps']={'median':statistics.median(pp),'min':min(pp),'max':max(pp),'n':len(pp)}
    out[label]=summary
print(json.dumps(out,indent=2))
