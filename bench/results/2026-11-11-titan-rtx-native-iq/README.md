# TITAN RTX QFN: native IQ CUDA build and long-prompt profile

2026-11-11, one TITAN RTX (24,576 MiB), Xeon E5-2680 v4 (14 cores / 28
threads), 64 GB DDR4. Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, the owner's existing
two GGUF shards and native expert pack, read-only throughout.

The new sm_75 engine and fast profile improved **31,610-token cold prefill by
29.7%** against the installed 0.1.57 engine in three alternating pairs. TTFT
fell from **21.319 to 16.448 seconds**. Short code and Korean decode medians
were unchanged. This combined result includes the engine version, FP16
projections and QSA options; the IQ source patch alone gave **+4.3%** at that
length on the same new binary.

## What changed

The native-IQ dispatch adapts [c3c's upstream PR #3277](https://github.com/Niko1221/Strata/pull/3277),
head `4450012270c81f89bbf415cc77ab04adf731e5fb`, to this fork. The widest IQ codebooks at
WW=4 exceed Turing's shared-memory limit. Each admitted format is checked for
shared-memory use and occupancy, then uses the qualified WW=2 / 128-row shape.
The old Q2_0 switch remains separate; unsupported formats keep their fallback.
All 48 layers of this pack, across ten observed format pairs, are eligible.

The native IQ switch requires `STRATA_PF_FUSED=1`, `STRATA_PF_FUSED_SM75=1` and
`STRATA_PF_FUSED_NATIVE_SM75=1`. The fast profile additionally sets:

```text
STRATA_BF16_TC=1
STRATA_SELECT_SIMT=2
STRATA_PA_V2T=1
```

These change prompt arithmetic and can change answers. They do not change the
model files or KV quantization. They remain explicit configuration options,
outside the existing `STRATA_SM75_PERF=1` group. Decode dispatch was not changed
by the IQ patch. The MIT license and upstream attribution are retained.

## Build and method

- Newly compiled engine: **0.1.59.1**, CUDA **13.0.88**, GCC **14.4.0**, driver
  **595.99.02**, `75-real;75-virtual`, Flash-Next geometry, portable AVX2 CPU
  backend. Native source commit: [6dbafa61](https://github.com/comm3451/Strata-Turing/commit/6dbafa6115ccc509795c2d97788d00039934ee1d).
- New executable SHA-256:
  `875efc6063f1977648bea649664b21e3745a278406c226e7a06b8d8801ab3420`.
- Installed 0.1.57 executable SHA-256:
  `9980139c6119dec0de3801aa5233b7f33b50bc814917cae24a7a098fcafbbe63`.
- Same installed 0.1.57 Python/server for both engines; explicit MTP, depth 4,
  confidence 0.5, 13 workers, automatic PCIe share and expert cache, expert
  mode 8, SM75 series on, 2048 MiB reserve and elastic vision configuration.
- **262,144 total context**, INT8 KV, **32,768 resident cells**, **one slot** in
  every performance arm. Cold-input tests disable prompt and conversation
  caches; every measured request reports zero reused tokens.
- A fresh server per arm, 16-token warmup, A/B then B/A then A/B. Same prompts,
  greedy sampling, thinking disabled and 512-token output limit. Long inputs
  contain seeded synthetic word lists; they are not a representative corpus.
- All tools, downloads, builds, configs, caches and logs were inside the
  checkout. `tools/turing-run.sh` mounted ordinary external files read-only.
  No driver, system package, clock or power setting was changed.

[build.json](build.json) records the initial build base plus source-file hashes:
the binary was compiled with the patch before its commit. The source hashes
match the validated native-kernel commit. The exact toolchain package list and
bootstrap/build scripts are in `tools/`; the binary's runtime libraries remain
in `.turing/toolchain`. Binaries and model files are not committed.

## End-to-end result: installed engine → new fast profile

Medians of three requests per arm/case:

| Work | Installed | New fast | Change |
| --- | ---: | ---: | ---: |
| Code decode, tok/s | 76.6 | 76.6 | 0.0% |
| Korean decode, tok/s | 93.4 | 93.4 | 0.0% |
| 4,349-token prefill, tok/s | 1158.8 | 1205.8 | +4.1% |
| 4,349-token TTFT, s | 3.791 | 3.644 | -3.9% |
| 31,610-token prefill, tok/s | 1487.2 | 1929.0 | +29.7% |
| 31,610-token TTFT, s | 21.319 | 16.448 | -22.8% |

All 24 requests succeeded. The code answer was 478 tokens and identical in
all arms. The Korean throughput case reached its 512-token limit; the separate
quality cases below completed. Long-summary responses can differ, so their
decode rates are not presented as fixed-work speedups.

The two-second sampler across this comparison observed at most **22,182 MiB
GPU use**, at least **1,840 MiB GPU free**, at least **12.56 GiB host available**,
and at most **70°C**. These are sampled extrema, not allocator peaks. About
191 MiB of swap was already in use before the run and remained allocated;
that alone does not establish active swapping during inference.

Raw responses and timings: [final-ab-results.json](final-ab-results.json).
Configurations, binary/server/harness hashes: [final-ab-metadata.json](final-ab-metadata.json).
Memory samples: [final-ab-memory.jsonl](final-ab-memory.jsonl). Per-arm engine
logs are the `final-ab-r*-engine.txt` files beside this report.

## Near-256K input

One additional installed/new pair used **252,146 input tokens**, zero reused
tokens, and three access-code markers spread through a seeded word list. Both
returned exactly `729183, 416927, 853204` in 23 generated tokens.

| Work | Installed | New fast | Change |
| --- | ---: | ---: | ---: |
| Prefill, tok/s | 1361.8 | 1733.7 | +27.3% |
| TTFT, s | 185.432 | 145.714 | -21.4% |

This is **one pair and one simple retrieval task**, not a repeated long-context
quality evaluation. Capacity was 262,144; the actual input was 252,146. Across
both runs the two-second sampler observed at most 22,150 MiB GPU use, at least
1,872 MiB GPU free, at least 12.45 GiB host available and at most 77°C.
[Responses](near-256k-results.json), [configuration](near-256k-metadata.json),
[memory samples](near-256k-memory.jsonl).

## Production cache and two queued clients

The new fast engine also completed nine short requests with the actual
7,168 MiB conversation-cache budget / 4,096 MiB floor restored. This test used
one engine slot, clients 1 then 2, three rounds each and a 256-token answer cap.
API authentication was removed only in the loopback test copy.

| Clients | Median total tok/s | TTFT p50 / p95, s | Completion p50 / p95, s | Errors |
| --- | ---: | ---: | ---: | ---: |
| 1 | 67.1 | 0.72 / 0.97 | 3.81 / 5.12 | 0 |
| 2 | 66.6 | 2.46 / 4.78 | 5.81 / 7.96 | 0 |

Total rate includes prompt processing and queue time. Two callers share a
queue; they do not decode simultaneously. This is not a test of two long
252K requests or a reason to enable two engine slots. The post-warmup snapshot
had 22,044 MiB GPU use / 1,978 MiB free and 12,867 MiB host available.
[Results](concurrency-results.json), [configuration](concurrency-metadata.json),
[load log](concurrency-load.txt), [snapshot](concurrency-memory.txt).

## Isolating the candidates

The IQ switch alone, same new binary and BF16 projection path, three pairs:

| Work | MMQ | Native IQ | Change |
| --- | ---: | ---: | ---: |
| 4,349-token prefill, tok/s | 1157.6 | 1184.5 | +2.3% |
| 31,610-token prefill, tok/s | 1485.1 | 1548.5 | +4.3% |
| Code decode, tok/s | 76.5 | 76.5 | 0.0% |
| Korean decode, tok/s | 93.5 | 93.6 | +0.1% |

[Native-IQ raw data](native-ab-results.json), [configuration](native-ab-metadata.json).
On synthetic individual expert layers, fused/MMQ speed ratios were measured
at 2,048 and 8,192 tokens; see [kernel-timing.txt](kernel-timing.txt). Those are
not whole-model speedups.

`STRATA_PREFILL_ROUTED_STREAM=1` on top of native IQ + FP16 gave only +0.3% at
4K and +0.8% at 32K (three pairs). Responses matched within every pair. It is
left off in the selected profile because the gain was small in this workload.
[Results](routed-ab-results.json), [configuration](routed-ab-metadata.json).

The QSA scorer/attention addition gave 1846.8 → 1909.7 tok/s at 31,610 tokens
in **one exploratory pair** (+3.4%). Its individual contribution needs more
repetitions; the complete profile above was subsequently tested for three
pairs. [Pilot](attention-pilot-results.json).

## Numerical and output checks

- Native IQ: **78** reference/edge comparisons, all 12 IQ format pairs plus
  Q2_0, at 256, 33, 1, 130, 517 and 640 tokens. All passed, including empty
  experts, zero inputs, scattered two-byte-aligned blobs and tile tails. Max relative
  RMS against MMQ was **0.0004522**; max worst-row relative error **0.005599**.
  [Full output](parity-edge.txt), [summary](parity-summary.json).
- Eight additional real-shape comparisons at 2,048/8,192 tokens passed the
  existing MMQ and FP32-reference accuracy gates. Three dispatch tests passed:
  new flag off, on, and on without the older SM75 master gate.
- Both tiled scorer variants passed the FP64 gate at 131,072 active tokens
  under 262,144 capacity. All **256/256 selections** matched in those synthetic
  cases. [SIMT 1](select-simt-1.txt), [SIMT 2](select-simt-2.txt).
- V1 and V2T prompt attention passed the FP64/FP32 checks for INT8 and FP16
  pools, including short-context tails. The unsupported Q4 pool cases were
  explicitly skipped on Turing. At 32K/256 queries, the INT8 prompt kernel
  took approximately 1.840 → 1.415 ms. [V1](attention-v2t-0.txt), [V2T](attention-v2t-1.txt).

For model-level smoke checks, 18 fixed prompts covered code, Korean, arithmetic,
JSON, elementary logic and tool-call-shaped JSON at roughly 600, 2,400 and 5,000
tokens. Two alternating pairs used the same new binary: MMQ/BF16/default QSA
against the selected fast configuration. **All 72 requests completed without
truncation and passed the recorded content checks.** These are simple tasks,
not a general reasoning, coding or tool-use evaluation.

Across the 36 candidate/reference first-token distributions, mean KL was
**0.005216**, max **0.025324**, with **33/36** equal top-1 tokens. Repeating the
baseline itself gave mean KL **0.003631**, max **0.016446**, and **18/18** equal
top-1 tokens. Scheduling/adaptation noise is present; do not subtract the two
KL means to claim a net quality score. The candidate is not byte-identical.
This does not establish unchanged perplexity or general long-context quality.

[Prompts](quality-prompts.json), [responses](quality-results.json),
[checks](quality-checks.json), [KL and logit-file hashes](quality-logits.json).
Raw FP32 logit files remain in `.turing/bench/newcuda/quality` locally.
`compare-logits.py` recomputes the KL from that raw harness directory. The
content checker only executes a manually reviewed merge algorithm after an
exact AST check (docstrings/comments excluded); it tests 100 seeded list pairs
and input preservation. The short throughput answer has a separate hash guard:

```sh
python3 -B bench/results/2026-11-11-titan-rtx-native-iq/check-quality.py
python3 -B bench/results/2026-11-11-titan-rtx-native-iq/check-code.py
python3 -B bench/results/2026-11-11-titan-rtx-native-iq/summarize.py
```

CUDA was built and run on this node. HIP, SYCL, other GPUs, full model perplexity
and vision inference were not tested in this change.

## Reproduce and launch

See [docs/TURING.md](../../../docs/TURING.md) for the isolated build, opt-in
profile generator and benchmark commands. The local launch helper accepts:

```sh
# Native-IQ change only, BF16 projections and default QSA:
bash .turing/launch/start-qfn.sh turing
# The measured combined profile:
bash .turing/launch/start-qfn.sh turing-fast
```

The launch JSON files preserve the owner's authentication, original model and
vision paths, and conversation-cache budget. They have mode 0600 and stay
ignored. The performance JSON fixtures deliberately disable caches and API
authentication on loopback; they are not production launch files. To reproduce
the comparisons, use `qfn_turing_bench.py` with the same settings, fixed server,
`--rounds 3`, and the prompts defined by `ab_engine.long_prompt(4096)` and
`long_prompt(30000)`. The installed/new comparison additionally requires
`--compare-builds`. Exact sanitized settings are in the metadata files.

The original PR and [NVIDIA's Turing guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
informed the implementation. All performance numbers above are measurements
of this node, not the upstream contributor's RTX 2080 Super Max-Q results.
