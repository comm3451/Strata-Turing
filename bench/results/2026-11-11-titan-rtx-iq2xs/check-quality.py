"""Check the recorded smoke tasks; this is not a general model-quality benchmark.

Only the reviewed merge algorithm is executed. Docstrings/comments may differ.
Its complete AST must match the reference, with or without builtin list type
annotations. The slice/extend tail was also inspected after the original gate
correctly refused to execute it. No unreviewed program is executed.
"""
import ast, hashlib, json, random, re
from pathlib import Path
REFERENCE = """def merge_sorted_lists(list1, list2):
    merged = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    while i < len(list1):
        merged.append(list1[i])
        i += 1
    while j < len(list2):
        merged.append(list2[j])
        j += 1
    return merged
"""
def tree(code):
    t=ast.parse(code)
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value,str):
            n.body.pop(0)
    return t
expected=ast.dump(tree(REFERENCE),include_attributes=False)
typed=ast.dump(tree(REFERENCE.replace('def merge_sorted_lists(list1, list2):',
                                    'def merge_sorted_lists(list1: list, list2: list) -> list:')),
               include_attributes=False)
extended=ast.dump(tree(REFERENCE[:REFERENCE.index('    while i < len(list1):')] +
                      '    merged.extend(list1[i:])\n    merged.extend(list2[j:])\n    return merged\n'),
                  include_attributes=False)
extended_typed=ast.dump(tree((REFERENCE[:REFERENCE.index('    while i < len(list1):')] +
                            '    merged.extend(list1[i:])\n    merged.extend(list2[j:])\n    return merged\n').replace(
                            'def merge_sorted_lists(list1, list2):',
                            'def merge_sorted_lists(list1: list, list2: list) -> list:')),
                        include_attributes=False)
rows=json.loads((Path(__file__).parent/"quality-results.json").read_text())
assert len(rows)==72
counts={}
for row in rows:
    text=row["response"]["text"].strip();kind=row["case"].split("-")[0]
    assert row["metrics"]["ok"] and row["metrics"]["generated_tokens"]<384
    if kind=="code":
        match=re.fullmatch(r"```python\n(.*)\n```",text,re.S)
        assert match, text
        t=tree(match.group(1))
        assert ast.dump(t,include_attributes=False) in (expected,typed,extended,extended_typed),"Unreviewed program: inspect before executing"
        scope={"__builtins__":{"len":len,"list":list}}
        exec(compile(t,"<reviewed merge algorithm>","exec"),scope)
        rng=random.Random(75)
        for _ in range(100):
            a=sorted(rng.choices(range(-10,11),k=rng.randrange(20)))
            b=sorted(rng.choices(range(-10,11),k=rng.randrange(20)))
            old_a,old_b=a[:],b[:]
            assert scope["merge_sorted_lists"](a,b)==sorted(a+b)
            assert a==old_a and b==old_b
    elif kind=="json":
        assert json.loads(text)=={"name":"copper","quantity":17,"active":True}
    elif kind=="tool":
        assert json.loads(text)=={"name":"get_weather","arguments":{"city":"Seoul"}}
    elif kind=="math":
        # Arithmetic correctness only; several answers add working despite asking for a number.
        assert re.findall(r"\d+",text)[-1]=="313"
    elif kind=="logic":
        assert text.startswith("No,") and "lamps" in text and "stones" in text
    elif kind=="korean":
        # A manual read checked meaning; these markers only detect a missing/truncated explanation.
        explicit = "리스트는 `[ ]`로 표시되며 생성 후 내용을 변경할 수 있고, 튜플은 `( )`로 표시되며 생성 후 내용을 변경할 수 없습니다."
        assert text == explicit or all(word in text for word in ["리스트","튜플","가변","불변"])
    counts[(row["arm"],kind)]=counts.get((row["arm"],kind),0)+1
print(json.dumps({"requests":len(rows),"passed_by_arm_and_task":{f"{a}/{k}":v for (a,k),v in counts.items()},"reviewed_algorithm_ast_sha256":{k:hashlib.sha256(v.encode()).hexdigest() for k,v in [('plain',expected),('list_annotations',typed),('extend',extended),('extend_list_annotations',extended_typed)]}},indent=2))
