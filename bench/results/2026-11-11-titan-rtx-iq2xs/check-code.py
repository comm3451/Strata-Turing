"""Check the inspected complete merge programs from steady-ab; writes nothing.

All five full responses were reviewed before adding their hashes. Execute only
these recorded programs, in a limited scope, then check their merge functions.
This verifies one coding task, not general coding quality.
"""
import hashlib
import json
from pathlib import Path
import random
import re

reviewed = {
    "1fee95d47085c8097c241bf00b196a170930f54aaaeaf4c117c2889660ecc092",
    "198b8802a810fb0cec427bfa923df9d1af5e2585716245306b731e3b0c3ccdac",
    "f88fd2ce51a5f6ec0bb7917e12e0652dca4e8ec9cefba5ad9a03bd99ff67944c",
    "12238a682ef8efd7b6b96c2344ceeb6ce834714e647dc971c6670b52babe6222",
    "bf3c9ffafed16f6b54b5fb150404aa21ca30ace7866ffc11917bf9b673845836",
}
rows = json.loads(Path(__file__).with_name("steady-ab-results.json").read_text())
answers = [r["response"]["text"] for r in rows if r["case"] in ("code", "long-code")]
assert len(answers) == 12
for text in set(answers):
    assert hashlib.sha256(text.encode()).hexdigest() in reviewed, "Review the new response first"
    code = re.fullmatch(r"```python\n(.*)\n```\s*", text, re.S).group(1)
    scope = {"__builtins__": {"len": len, "list": list}, "__name__": "quality_check"}
    exec(compile(code, "<reviewed merge program>", "exec"), scope)
    for name, func in list(scope.items()):
        if name.startswith("test_"):
            func()
    rng = random.Random(75)
    for _ in range(100):
        a = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
        b = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
        before_a, before_b = a[:], b[:]
        assert scope["merge_sorted_lists"](a, b) == sorted(a + b)
        assert a == before_a and b == before_b
print(f"PASS: {len(set(answers))} reviewed programs across 12 responses; embedded tests and 100 seeded cases each; inputs unchanged.")
