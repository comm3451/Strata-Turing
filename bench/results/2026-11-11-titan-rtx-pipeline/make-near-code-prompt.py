"""Recreate the large decode prompt without storing the repeated word list in Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
from ab_engine import PROMPTS, long_prompt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', required=True, help='new JSON file inside this checkout')
args = ap.parse_args()
out = Path(args.out).resolve()
if os.environ.get('STRATA_TURING_ISOLATED') != '1' or not out.is_relative_to(ROOT):
    ap.error('use tools/turing-run.sh and an output inside the checkout')
prompt = long_prompt(240000)
ending = 'Summarize what kinds of words appear in the list, in five sentences.'
assert prompt.endswith(ending)
prompt = prompt[:-len(ending)]+'Ignore the background word list above. '+PROMPTS[0]
payload = json.dumps([['long-code-252k', prompt]], ensure_ascii=False, indent=2)+'\n'
manifest = json.loads(Path(__file__).with_name('near-code-prompt-manifest.json').read_text())
assert hashlib.sha256(payload.encode()).hexdigest() == manifest['cases_sha256']
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('x') as stream:
    stream.write(payload)
print(out)
