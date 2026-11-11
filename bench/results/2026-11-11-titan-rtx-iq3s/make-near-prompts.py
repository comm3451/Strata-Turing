"""Recreate the recorded long retrieval and code prompts inside the checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
from ab_engine import PROMPTS, long_prompt
from qfn_turing_bench import retrieval_prompt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = Path(args.out).resolve()
if os.environ.get("STRATA_TURING_ISOLATED") != "1" or not out.is_relative_to(root):
    ap.error("use tools/turing-run.sh and an output inside this checkout")
prompt = long_prompt(240000)
ending = "Summarize what kinds of words appear in the list, in five sentences."
assert prompt.endswith(ending)
cases = [["long", retrieval_prompt(240000)],
         ["long-code-252k", prompt[:-len(ending)] + "Ignore the background word list above. " + PROMPTS[0]]]
data = json.dumps(cases, ensure_ascii=False, indent=2) + "\n"
manifest = json.loads(Path(__file__).with_name("near-prompts-manifest.json").read_text())
assert hashlib.sha256(data.encode()).hexdigest() == manifest["cases_sha256"]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x") as stream:
    stream.write(data)
print(out)
