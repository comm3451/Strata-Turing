# QFN IQ3_S on TITAN RTX and 64 GB RAM

Measured on 2026-11-11 with one TITAN RTX 24 GiB and Xeon E5-2680 v4
(14 cores / 28 threads), 64 GB DDR4. The model is ISTA-DASLab QFN GSQ-RCO
IQ3_S, with the original IQ4_NL N-gram table for these measurements.
**These are not BF16 N-gram results.**

## Files and engine

The 54,817,524,224-byte IQ3_S first shard was downloaded from pinned revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09` of
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
The downloader verified SHA-256
`4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`
before making the file usable. A new native pack and tokenizer were generated
inside this checkout; experts are read from the GGUF, without a second large
`experts.bin` copy. Expert payload is 46.84 GiB versus IQ3_XXS's 39.97 GiB.

The existing second shard was verified against the published IQ3_S second
shard: 28,800,138,432 bytes, SHA-256
`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.
A symlink inside this checkout reuses it read-only. See the manifests and
`reused-iq4-ngram.json`; the original installation/model files were unchanged.

All inference uses the same CUDA sm_75 executable as the
[IQ3_XXS fusion report](../2026-11-11-titan-rtx-iq2xs/README.md): SHA-256
`1b50bc01e1f3b6957b6fe8a58b577096bb5546278899691e3ab6f2578cfc4c49`,
CUDA 13.0.88/GCC 14.4, installed Python server 0.1.57.
IQ3_S has **no IQ2_XS layers**, so the new type-17 fused kernel is not its
source of acceleration. The profile disables that switch explicitly.

## Startup and memory registration

The first full-arena trials, `iq3s-comparison` and `iq3s-no-thp`, exceeded
the 600-second readiness deadline while starting IQ3_S. Their logs stop
before arena load completion. Both forced and disabled THP failed, with
severe host memory pressure and delayed monitoring; THP alone did not
explain the failure. They contain IQ3_XXS reference answers only, not
successful IQ3_S measurements. The exact kernel/driver cause was not profiled.

`STRATA_ARENA_PIN_GIB=40` allowed IQ3_S to start with all experts in RAM.
The first working configuration registers a prefix of 41 layers, about
39 GiB. `STRATA_PIN_BY_NEED=1` instead registers experts by profile priority,
covering all estimated nonresident experts: 32.05 GiB across all 48 layers,
then other experts until the 40 GiB cap. The observed total is 40,958 MiB
across 7,577 registered ranges. All of this is per-process; no system pinning
limits, THP policy or packages were changed.

Both successful arms use `STRATA_NO_ARENA_THP=1`. The unregistered remainder
is ordinary anonymous RAM; its `mlock` attempt fails at the existing limit,
which is retained in the raw logs. Do not call the entire model locked or
assume the same memory margin under other workloads.

## Repeated tuning comparison

`iq3s-pin-need40` is the initial one-pair pilot. The independent confirmation
uses three alternating pairs (A/B, B/A, A/B), three 256-token warmups per arm,
and code/Korean/31,610-token input cases. Both arms use the same IQ3_S files,
engine, greedy sampling, 262,144 capacity, INT8 KV with 32,768 resident cells,
13 workers, MTP 4/min-p 0.5, 2 GiB VRAM reserve, 32K prefill chunks and zero
parked/prompt cache. All 18 measured requests have zero reused tokens.
Only `STRATA_PIN_BY_NEED=1` differs. Downloads were paused throughout.

| Metric | Prefix pinning median | Pin by need median | Change |
|---|---:|---:|---:|
| Short code decode | 76.0 tok/s | 80.1 tok/s | +5.4% |
| Korean decode, 512-token cap | 69.4 tok/s | 71.1 tok/s | +2.4% |
| 31,610-token prefill | 2,077.5 tok/s | 2,131.5 tok/s | +2.6% |
| Code completion time | 6.981 s | 6.243 s | -10.6% |
| 31,610-token client TTFT | 15.286 s | 14.903 s | -2.5% |

All three paired code/Korean decode and long-prefill comparisons favored
pin by need. Code answers differ: prefix produces 468/509 tokens, candidate
477 tokens. The completion-time improvement includes shorter answers and
is not a pure kernel-speed ratio. Placement between CPU/GPU can change
rounding and output. Korean performance answers are capped, so their rate
does not establish answer quality. The recorded complete code functions
are checked separately with 100 seeded merge cases.

The 2-second samples show peak VRAM 22,198 MiB and minimum available host
RAM 4.67 GiB during confirmation. Swap already contained about 260 MiB;
the used-size range was 262,640–267,848 KiB. This is not a page-in/out counter.

These are IQ3_S tuning results, not a controlled IQ3_XXS-versus-IQ3_S quality
benchmark. The successful IQ3_XXS report used a different physical page
allocation. Neither report establishes a hardware maximum or the C++
accuracy improvement claimed in the community article.

## Production profile and near-limit checks

`--titan-iq3s` selects the measured pin-by-need configuration and keeps
262,144 total context tokens. It reserves a smaller parked conversation
cache, 1,024 MiB with a 4,096 MiB free-RAM floor. The near-limit and answer
checks use that production profile, with the unused IQ2_XS switch disabled.
`make-near-prompts.py` regenerates the long retrieval/code inputs and checks
their recorded SHA-256. All 32 previous profile-option combinations retain
their behavior (`profile-checks.json`).

Both near-limit requests succeeded with zero reused tokens:

| Actual input | Result | Prefill | Client TTFT | Decode |
|---|---|---:|---:|---:|
| 252,146 tokens | Three markers recovered exactly | 2,048.2 tok/s | 123.390 s | 23 tokens; too short for a sustained rate |
| 252,100 tokens | Complete 587-token merge program | 2,026.7 tok/s | 124.658 s | 82.4 tok/s |

The long program's extracted merge function matches the reviewed algorithm.
Peak VRAM was 22,178 MiB and minimum available RAM 4.56 GiB. Already used
swap fell slightly, from 262,640 to 261,368 KiB; swap traffic was not measured.
This establishes these two real near-256K requests on this node, not general
long-context quality or two simultaneous full-context engine slots.

All **18 separate basic content checks** pass: merge code, Korean explanation,
arithmetic, JSON, tool-call JSON and logic at about 0.6K/5K/20K input lengths.
Every answer finishes below the 384-token cap. The three Korean explanations
were read manually; two correctly use wording other than the initial marker
words and are checked by exact reviewed text. Arithmetic checks the result,
not strict formatting: some answers show working despite asking for a number.
Tool-call JSON is a syntax/content check, not an actual external tool execution.

`check-answers.py` validates six performance code functions, the near-limit
retrieval/program and the 18 basic answers. It executes only merge functions
whose AST matches the reviewed algorithm, then checks 100 seeded inputs and
input preservation for each. Surrounding generated tests are not executed.
The BF16 download ran during the separate 18-answer check, so those timings
are excluded from performance claims. There is no broad model-quality score.

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_profile.py \
  --source .turing/bench/nextopt/iq3s-pin-need40.json \
  --exe "$PWD/build/qfn-turing/strata" \
  --out .turing/launch/qfn-iq3s.json --titan-iq3s
```

The source must already reference an IQ3_S model, pack, tokenizer, N-gram
table and compatible MTP pack. This command does not silently change an
IQ3_XXS model path or select a BF16 table. Existing API authentication is
preserved in the private output config.

## BF16 N-gram preparation

The requested original BF16 table is a separate **102,400,491,712-byte**
download from pinned revision `9af845f821fec6bf2a9ddbacd3778930ba80157f` of
[apetersson/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/apetersson/Qwen3.8-Flash-Next-GGUF).
Expected SHA-256 is
`11828e522266cf7aa9579dea8a0ab375be7afcadede4bd9b9fcfb0722d952ca1`.
It remains a `.part` file until the complete checksum passes. Background
download is resumed after performance testing; correctness-only runs may
overlap it, and their timings must not be used as performance evidence.

Its 4 KiB downloaded header was read with the engine's C++ GGUF reader:
one tensor, three metadata entries, data start 192. The tensor is
`per_layer_token_embd.weight`, BF16, shape `[160, 320001536]`. It has split
number 1/count 2 but no `general.architecture`. The loader now recognizes a
table-only GGUF by its sole tensor as well as by `architecture=strata-ple`,
so it does not append this table to the dense-shard list and duplicate a
native shard number. The CUDA build and regular IQ3_XXS/IQ3_S inference pass;
this header check is **not full BF16 inference validation**. No BF16 speed,
memory or quality improvement is claimed before that separate run.
