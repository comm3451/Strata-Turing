"""Check the one inspected, complete merge program in the published responses.

    python3 -B bench/results/2026-11-11-titan-rtx-pipeline/check-code.py

The hash guard must match before executing any generated code. This checks one
coding task, not overall model quality. It writes nothing.
"""
import hashlib
import json
from pathlib import Path
import random
import re

root = Path(__file__).resolve().parent
answers = [row["response"]["text"]
           for name in ("final-ab",)
           for row in json.loads((root / f"{name}-results.json").read_text())
           if row["case"] == "code"]
assert len(answers) == 6 and len(set(answers)) == 1
code = re.fullmatch(r"```python\n(.*)\n```\s*", answers[0], re.S).group(1)
if hashlib.sha256(code.encode()).hexdigest() != "7dc23c5495ca5073d9dac6f0c13ffef3543fdd722b5e1152dec2bed3592347ce":
    raise ValueError("the generated program changed; inspect it before execution")
scope = {"__builtins__": {"len": len, "list": list}, "__name__": "quality_check"}
exec(compile(code, "<inspected merge example>", "exec"), scope)
scope["test_merge_basic"]()
scope["test_merge_overlapping"]()
rng = random.Random(75)
for _ in range(100):
    a = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
    b = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
    before_a, before_b = a[:], b[:]
    assert scope["merge_sorted_lists"](a, b) == sorted(a + b)
    assert a == before_a and b == before_b
print("PASS: one unique program across 6 responses; 2 embedded tests and 100 seeded cases; inputs unchanged.")
