"""Run only the manually reviewed long-context code, guarded by its full hash."""
import hashlib
import json
from pathlib import Path
import random
import re

APPROVED = {'9c43f56d01da9f22c1efae7d597c8bfddfb3a0ba4285f14c23cbc33567371df1'}
rows = json.loads(Path(__file__).with_name('near-code-results.json').read_text())
assert len(rows) == 2 and {r['arm'] for r in rows} == {'baseline', 'candidate'}
checks = []
for row in rows:
    assert row['metrics']['ok'] and row['metrics']['generated_tokens'] < 512
    match = re.fullmatch(r'```python\n(.*)\n```\s*', row['response']['text'], re.S)
    assert match, 'Inspect the changed output before execution'
    code = match.group(1)
    sha = hashlib.sha256(code.encode()).hexdigest()
    assert sha in APPROVED, 'Unreviewed code: inspect it before execution'
    scope = {'__builtins__': {'len': len, 'list': list}, '__name__': 'quality_check'}
    exec(compile(code, '<reviewed long-context merge>', 'exec'), scope)
    scope['test_merge_sorted_lists']()
    rng = random.Random(75)
    for _ in range(100):
        a = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
        b = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
        old_a, old_b = a[:], b[:]
        assert scope['merge_sorted_lists'](a, b) == sorted(a+b)
        assert a == old_a and b == old_b
    checks.append({'arm': row['arm'], 'code_sha256': sha, 'embedded_cases': 2,
                   'random_cases': 100, 'inputs_unchanged': True})
print(json.dumps(checks, indent=2))
