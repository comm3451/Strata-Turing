import json,os,signal,subprocess,sys,time
from pathlib import Path
assert os.environ.get("STRATA_TURING_ISOLATED")=="1"
log=Path(sys.argv[1]).resolve()
assert log.is_relative_to(Path.cwd())
p=subprocess.Popen(sys.argv[2:],start_new_session=True)
try:
    with log.open("x") as f:
        while p.poll() is None:
            gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.free,temperature.gpu,power.draw,clocks.sm,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
            mem={k:int(v.split()[0]) for k,v in (line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines()) if k in ("MemAvailable","SwapFree","SwapTotal","AnonHugePages")}
            f.write(json.dumps({"time":time.time(),"gpu":gpu,**mem})+"\n")
            f.flush()
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
    sys.exit(p.returncode)
finally:
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try: p.wait(timeout=20)
        except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
