"""Print recorded IQ3_S tuning medians and observed memory; no inference."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
rows = json.loads((root / "iq3s-pin-need40-confirm-results.json").read_text())
assert len(rows) == 18 and all(r["metrics"]["ok"] and r["metrics"]["reused_tokens"] == 0 for r in rows)
summary = {}
for case in dict.fromkeys(r["case"] for r in rows):
    result = {}
    for metric in ("decode_tps", "prompt_tps", "ttft_s", "total_s", "generated_tokens"):
        a = median(r["metrics"][metric] for r in rows if r["case"] == case and r["arm"] == "baseline")
        b = median(r["metrics"][metric] for r in rows if r["case"] == case and r["arm"] == "candidate")
        result[metric] = {"prefix": a, "by_need": b, "percent_change": 100 * (b / a - 1)}
    summary[case] = result
memory = {}
for name in ("iq3s-pin-need40-confirm", "iq3s-near-256k"):
    samples = [json.loads(line) for line in (root / f"{name}-memory.jsonl").read_text().splitlines()]
    memory[name] = {
        "min_available_ram_gib": min(r["MemAvailable"] for r in samples) / 2**20,
        "peak_vram_mib": max(float(r["gpu"].split(",")[0]) for r in samples),
        "swap_used_kib_range": [min(r["SwapTotal"] - r["SwapFree"] for r in samples),
                                max(r["SwapTotal"] - r["SwapFree"] for r in samples)],
    }
print(json.dumps({"pinning_comparison": summary, "memory": memory}, indent=2))
