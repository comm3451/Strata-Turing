"""Recompute the median throughput/latency and sampled memory statistics."""
import json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
out={}
for f in sorted(p.glob("*-results.json")):
    if f.name=="quality-results.json": continue
    rows=json.loads(f.read_text())
    if not isinstance(rows,list) or not rows or "metrics" not in rows[0]: continue
    result={}
    for case in sorted({r["case"] for r in rows}):
        entry={}
        for arm in ["baseline","candidate"]:
            sel=[r for r in rows if r["case"]==case and r["arm"]==arm]
            if not sel: continue
            assert all(r["metrics"]["ok"] and r["metrics"]["reused_tokens"]==0 for r in sel)
            entry[arm]={"requests":len(sel),"prompt_tokens":sorted({r["metrics"]["prompt_tokens"] for r in sel})}
            for k in ["prompt_tps","decode_tps","ttft_s","total_s"]:
                entry[arm][k]=statistics.median(r["metrics"][k] for r in sel)
        if "baseline" in entry and "candidate" in entry:
            entry["change_percent"]={k:100*(entry["candidate"][k]/entry["baseline"][k]-1) for k in ["prompt_tps","decode_tps","ttft_s","total_s"]}
        result[case]=entry
    out[f.stem]=result
for f in sorted(p.glob("*-memory.jsonl")):
    rows=[json.loads(l) for l in f.read_text().splitlines()]
    gpu=[list(map(float,r["gpu"].split(","))) for r in rows]
    out[f.stem]={"samples":len(rows),"max_gpu_used_mib":max(r[0] for r in gpu),"min_gpu_free_mib":min(r[1] for r in gpu),"max_gpu_temperature_c":max(r[2] for r in gpu),"min_host_available_gib":min(r["MemAvailable"] for r in rows)/1048576,"max_swap_used_mib":max(r["SwapTotal"]-r["SwapFree"] for r in rows)/1024}
print(json.dumps(out,indent=2))
