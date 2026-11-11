"""Validate the recorded smoke answers; execute only the reviewed merge AST.

This checks the merge function independently of surrounding generated tests.
It is not a general coding evaluation or proof of quality at long context.
"""
import ast
import json
from pathlib import Path
import random
import re

root = Path(__file__).resolve().parent
reference_source = ast.parse((root.parent / "2026-11-11-titan-rtx-iq2xs/check-quality.py").read_text())
reference = next(ast.literal_eval(n.value) for n in reference_source.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REFERENCE" for t in n.targets))

def algorithm(code):
    module = ast.parse(code)
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "merge_sorted_lists"]
    assert len(functions) == 1
    f = functions[0]
    if f.body and isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant) and isinstance(f.body[0].value.value, str):
        f.body.pop(0)
    for arg in f.args.args:
        assert arg.annotation is None or isinstance(arg.annotation, ast.Name) and arg.annotation.id == "list"
        arg.annotation = None
    assert f.returns is None or isinstance(f.returns, ast.Name) and f.returns.id == "list"
    f.returns = None
    return ast.Module(body=[f], type_ignores=[])

expected = {ast.dump(algorithm(reference), include_attributes=False),
            ast.dump(algorithm(reference[:reference.index("    while i < len(list1):")] +
                               "    merged.extend(list1[i:])\n    merged.extend(list2[j:])\n    return merged\n"),
                     include_attributes=False)}

# Manually reviewed recorded paraphrases: the answers describe mutability
# correctly without using the literal words "가변" and "불변".
reviewed_korean = {
    '리스트는 대괄호 `[]`를 사용하며 값을 변경할 수 있고, 튜플은 소괄호 `()`를 사용하며 한 번 생성하면 값을 변경할 수 없습니다.',
    '리스트는 대괄호 `[]`를 사용하며 값을 수정할 수 있고, 튜플은 소괄호 `()`를 사용하며 한 번 생성하면 값을 변경할 수 없습니다.',
    '리스트는 대괄호 `[]`로 감싸며 **수정 가능(mutable)**하고, 튜플은 소괄호 `()`로 감싸며 **수정 불가능(immutable)**합니다.',
}

def check_code(text):
    match = re.fullmatch(r"```python\n(.*)\n```\s*", text, re.S)
    assert match, "expected a complete fenced program"
    tree = algorithm(match.group(1))
    assert ast.dump(tree, include_attributes=False) in expected, "Unreviewed merge AST: inspect before executing"
    scope = {"__builtins__": {"len": len, "list": list}}
    exec(compile(tree, "<reviewed isolated merge function>", "exec"), scope)
    rng = random.Random(75)
    for _ in range(100):
        a = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
        b = sorted(rng.choices(range(-10, 11), k=rng.randrange(20)))
        old_a, old_b = a[:], b[:]
        assert scope["merge_sorted_lists"](a, b) == sorted(a + b)
        assert a == old_a and b == old_b

counts = {}
for run, count in (("confirm", 30), ("cache-balanced-pilot", 9), ("cache-confirm", 54),
                   ("near-continuity", 7), ("near-code", 1), ("quality", 18)):
    rows = json.loads((root / f"{run}-results.json").read_text())
    assert len(rows) == count and all(row["metrics"]["ok"] for row in rows), run
    checked = 0
    for row in rows:
        text = row["response"]["text"].strip()
        kind = row["case"].split("-")[0]
        if run in ("cache-balanced-pilot", "cache-confirm", "near-continuity"):
            assert row["correct_values"] and re.findall(r"\d+", text) == row["expected_values"]
            if run == "near-continuity" and row["case"] != "parked-seed":
                assert row["metrics"]["prompt_tokens"] > 250000
        elif kind == "code" or row["case"] in ("long-code", "long-code-252k"):
            check_code(text)
        elif run == "quality":
            assert row["metrics"]["generated_tokens"] < 384
            if kind == "json":
                assert json.loads(text) == {"name": "copper", "quantity": 17, "active": True}
            elif kind == "tool":
                assert json.loads(text) == {"name": "get_weather", "arguments": {"city": "Seoul"}}
            elif kind == "math":
                assert re.findall(r"\d+", text)[-1] == "313"
            elif kind == "logic":
                assert text.startswith("No,") and "lamps" in text and "stones" in text
            elif kind == "korean":
                assert text in reviewed_korean or all(word in text for word in ("리스트", "튜플", "가변", "불변")), text
            else:
                raise AssertionError(kind)
        else:
            continue
        checked += 1
    if run == "confirm":
        assert all(row["metrics"]["reused_tokens"] == 0 for row in rows)
    counts[run] = checked
print(json.dumps({"checked_answers": counts, "seeded_merge_cases_per_answer": 100}, indent=2))
