import os, subprocess
from pathlib import Path

assert os.environ.get('STRATA_TURING_ISOLATED') == '1'
p = Path('.turing/bench/nextopt')
pause = Path('.turing/downloads/PAUSE')
pause.touch()
env = dict(os.environ, LD_LIBRARY_PATH=str(Path('.turing/toolchain/lib').resolve())+':'+str(Path('.turing/toolchain/targets/x86_64-linux/lib').resolve()),
           STRATA_SM75_IQ2XS_FUSED='1')
model = '/home/eesdf/.local/share/strata/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'
try:
    with (p/'fused-sweep.txt').open('x') as log:
        for tokens in [1, 2, 3, 4, 5, 8]:
            for groups in [10, 20]:
                print(f'T={tokens} G={groups}', flush=True)
                subprocess.run(['build/qfn-turing/native_expert_bench', model, '0,2,4,5,9,10,14,19,25,39',
                                str(groups),str(tokens),'0','8','120'],env={**env,'BENCH_CAP':'40'},
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        # Off path for the newly covered format; existing formats must retain their outputs with the new switch.
        for switch in ['0','1']:
            with (p/f'fused-regression-{switch}.txt').open('x') as ref:
                subprocess.run(['build/qfn-turing/native_expert_bench',model,'0,1,8,12,35','20','4','0','8','120'],
                               env={**env,'STRATA_SM75_IQ2XS_FUSED':switch},stdout=ref,stderr=subprocess.STDOUT,check=True)
finally:
    pause.unlink(missing_ok=True)
