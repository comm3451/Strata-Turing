"""Summarize the recorded runs without writing files or rerunning inference."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
rows = json.loads((root / "steady-ab-results.json").read_text())
assert len(rows) == 30 and all(r["metrics"]["ok"] and r["metrics"]["reused_tokens"] == 0 for r in rows)
summary = {}
for case in dict.fromkeys(r["case"] for r in rows):
    subset = [r for r in rows if r["case"] == case]
    values = {}
    for metric in ("prompt_tps", "decode_tps", "ttft_s", "total_s", "generated_tokens"):
        a = [r["metrics"][metric] for r in subset if r["arm"] == "baseline"]
        b = [r["metrics"][metric] for r in subset if r["arm"] == "candidate"]
        values[metric] = {"baseline_median": median(a), "candidate_median": median(b),
                          "percent_change": 100 * (median(b) / median(a) - 1)}
    summary[case] = values
memory = {}
for name in ("steady-ab", "near-256k", "concurrency"):
    samples = [json.loads(line) for line in (root / f"{name}-memory.jsonl").read_text().splitlines()]
    used = [r["SwapTotal"] - r["SwapFree"] for r in samples]
    memory[name] = {
        "minimum_available_ram_gib": min(r["MemAvailable"] for r in samples) / 2**20,
        "peak_vram_mib": max(float(r["gpu"].split(",")[0]) for r in samples),
        "swap_used_kib_range": [min(used), max(used)],
    }
print(json.dumps({"steady": summary, "memory": memory}, indent=2))
