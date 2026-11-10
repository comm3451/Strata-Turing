import hashlib,json,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]); cases=json.loads((p/"prompts.json").read_text());rows=[json.loads(x) for x in (p/"results.jsonl").read_text().splitlines()]
rounds=sorted({r["round"] for r in rows});data=[];noise=[]
def load(prefix,i):
    path=p/f"{prefix}-first.{i+1}"  # index 0 is the warmup
    x=np.fromfile(path,dtype=np.float32).astype(np.float64)
    assert x.size and np.isfinite(x).all(),path
    x-=x.max();return x-np.log(np.exp(x).sum()),hashlib.sha256(path.read_bytes()).hexdigest()
def compare(a,b):
    assert a.shape==b.shape
    return {"kl":float(np.sum(np.exp(a)*(a-b))),"top1_equal":bool(a.argmax()==b.argmax()),"max_abs_logp_delta":float(np.max(np.abs(a-b)))}
for rnd in rounds:
    for i,(name,_) in enumerate(cases):
        a,ha=load(f"r{rnd}-baseline",i);b,hb=load(f"r{rnd}-candidate",i)
        data.append({"round":rnd,"case":name,**compare(a,b),"baseline_sha256":ha,"candidate_sha256":hb})
if len(rounds)>1:
    for i,(name,_) in enumerate(cases):
        a,_=load(f"r{rounds[0]}-baseline",i);b,_=load(f"r{rounds[1]}-baseline",i)
        noise.append({"case":name,**compare(a,b)})
def summary(rows):
    k=[r["kl"] for r in rows]
    return {"count":len(rows),"kl_mean":float(np.mean(k)),"kl_max":float(max(k)),"top1_equal":sum(r["top1_equal"] for r in rows)} if rows else None
print(json.dumps({"candidate":summary(data),"baseline_repeat":summary(noise),"comparisons":data,"noise":noise},indent=2))
