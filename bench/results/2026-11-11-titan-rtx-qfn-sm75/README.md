# TITAN RTX + Xeon E5-2680 v4: QFN IQ3_XXS tuning

Measured on 2026-11-11 (Asia/Seoul) on the owner's inference node. These are
targeted experiments, not the Strata standard benchmark and not measurements
of a newly compiled Strata-Turing engine.

The conservative profile keeps 13 CPU workers, expert mode 8, automatic PCIe
sharing and the current MTP/cache policy. The tested fixed-share and combined
tuner candidates did not improve the mixed workload enough to replace them.
FP16 tensor-core prefill is offered separately: it improved 4K prefill by 4.0%
in three rounds, with different rounding and some different replies.
A single near-limit MTP pair read 252,146 tokens: FP16 reduced TTFT from
184.580 to 156.039 seconds, and both paths retrieved all three markers correctly.

## Hardware and executable

- One NVIDIA TITAN RTX, 24,576 MiB VRAM, compute capability 7.5; driver 595.99.02.
- Xeon E5-2680 v4, 14 physical cores / 28 logical CPUs, one socket, AVX2.
- Two 32 GB DDR4 DIMMs (owner's specification); Linux reports 60.4 GiB usable.
  Memory channel population and DRAM frequency were not measured.
- Installed Strata engine 0.1.57. Its BUILD.json identifies
  `f-0214-int 0bdcef4682c5`, CUDA 13.0, portable CPU build and architectures
  75/86/89/120. Exact executable SHA-256 is in the metadata JSON files.
- Original `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` two-shard model and native pack,
  reused read-only. Full model-file hashes were not computed.
- Context capacity 262,144; INT8 KV; 32,768 resident KV cells unless stated otherwise.

All experiments ran with the ordinary filesystem read-only outside the checkout,
through [turing-run.sh](../../../tools/turing-run.sh). No driver, system package,
clock setting or original model/config file was changed. Logs and data here are
sanitized copies; local raw artifacts remain in the checkout's ignored `.turing/`.

## Baseline provenance matters

The first experiments used the initial config under the installed Strata folder.
That config had no explicit drafting policy or expert-mode overrides, so the
0.1.57 server resolved its old MTP arguments to DFlash2. Vision was disabled for
that exploratory text experiment; the VRAM reserve was 700 MiB.

The owner's **recently used** launch config was subsequently found in the launch
state directory. It explicitly selects **MTP**, `STRATA_SM75_PERF=1` and
`STRATA_EXP_MODE=8`. Its latest launch log uses a **2048 MiB** VRAM reserve and
a 7168 MiB conversation-cache budget. The image encoder is elastic and starts
only for an image. The earlier DFlash2 numbers therefore are **not gains over the
owner's usual configuration**.

The MTP comparisons use the recent launch's arguments, preserve the image
configuration (no images sent), and disable prompt/conversation checkpoints in
both arms to measure cold input. Saved tuning presets are disabled in every
experiment. Exact local paths are redacted in the per-experiment metadata.

## MTP: keep the forced expert layout

Baseline: `STRATA_EXP_MODE=8`. Candidate: remove that override and use the
Turing automatic layout. Both have `STRATA_SM75_PERF=1`. The automatic path
loses on this model, GPU and MTP workload, so the override is retained.

Three fresh-engine rounds per arm, order A/B, B/A, A/B. Each arm warms up once,
then reads the same code, Korean and synthetic 4K prompts. Temperature zero,
thinking off, maximum 256 output tokens. Actual output counts: 256, 256, 114.
No prompt tokens were reused.

| Case | Baseline decode tok/s | Automatic layout tok/s | Change |
| --- | ---: | ---: | ---: |
| Code, 34 input tokens | 69.9 | 69.2 | -1.0% |
| Korean, 37 input tokens | 87.6 | 83.6 | -4.6% |
| After 4,349 input tokens | 64.3 | 62.8 | -2.3% |

Medians of three runs. Full response text is retained in
[mtp-layout-results.json](mtp-layout-results.json); configuration and binary
identity are in [mtp-layout-metadata.json](mtp-layout-metadata.json).
The code response hits the output cap; this is not a coding-task pass rate.
Text equality does not establish equal logits or general model quality.

## CPU/GPU exploration and a rejected combined candidate

The installed calibrator explored PCIe shares, draft confidence, CPU workers and
expert-cache adaptation in 725 seconds. Its three English prompts are different
from the code/Korean/long corpus below. The PCIe copy probe measured 9.8 GB/s;
the runtime expert-pool estimate was about 22 GB/s of expert bytes. These are
engine measurements, not a DRAM bandwidth benchmark or a negotiated PCIe width.

| CPU workers | Decode tok/s in the tuner |
| --- | ---: |
| 13, plus the host on the remaining physical core | 87.1 |
| 9 | 81.5 |
| 6 | 75.4 |
| 3 | 60.9 |

Two samples per worker count after one warmup, at PCIe share 0.10 and MTP
confidence 0.70. Fewer workers lost; retain 13 on this machine. The engine pins
workers to logical CPUs 1–13 and the host to CPU 0; using 28 worker threads is
not supported by these results.

The initial automatic share was 0.10, but it refined to 0.25 during generation.
The installed calibrator compared fixed 0.10 against candidates rather than
this real automatic policy. Its confirmation and final 89.9 tok/s are therefore
**exploratory**, not a measured gain over the owner's profile. The fork fixes
that comparison in `tools/calibrate.py` and covers the refinement with regression
tests. [Raw tuner report](exploratory-calibrate-mtp-result.json),
[console](exploratory-calibrate-mtp-console.txt),
[engine log](exploratory-calibrate-mtp-engine.txt),
[config](exploratory-calibrate-mtp.json).

The tuner suggested share 0.10, confidence 0.70, and cache adaptation
`--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`. A separate three-round
comparison against the actual MTP configuration **rejects this combination for
general use**:

| Case, maximum 512 output tokens | Current decode tok/s | Combined candidate | Change |
| --- | ---: | ---: | ---: |
| Code | 76.9 | 86.0 | +11.8% |
| Korean | 94.0 | 91.1 | -3.1% |
| After 4,349 input tokens | 60.7 | 54.4 | -10.4% |

All 18 requests succeeded. Code responses matched (478 tokens). Korean responses
hit the 512-token cap and differed between arms. Long responses also differed,
including between baseline repetitions (116–122 versus 115 tokens). These rates
are not an equal-output performance proof or a quality-equivalence result.
The combined candidate's English tuning score did not predict the mixed workload.
[Responses and timings](mtp-combined-results.json),
[metadata](mtp-combined-metadata.json).

Keeping the default draft/cache settings and changing **only** `--pcie-frac 0.10`
also did not establish a useful general decode improvement. In another three
interleaved rounds at maximum 512 output tokens:

| Case | Automatic share decode tok/s | Fixed 0.10 decode tok/s |
| --- | ---: | ---: |
| Code | 77.2 | 77.8 |
| Korean | 94.1 | 94.9 |
| After 4,349 input tokens | 60.8 | 58.9 |

Korean TTFT fell from 0.331 to 0.276 seconds, but the two short-case decode gains
were below 1%, and long responses differed (122 versus 113 tokens). Retain
automatic sharing in the general profile. The fixed fraction is an experiment,
not a universal improvement on this CPU/GPU. [Data](mtp-pcie-results.json),
[metadata](mtp-pcie-metadata.json).

## FP16 tensor-core prefill: a separate opt-in profile

Only `STRATA_BF16_TC=1` changes here; PCIe sharing, MTP and cache adaptation
remain automatic/default as in the recent configuration. Three interleaved
rounds, maximum 512 output tokens:

| Metric | Current path | FP16 tensor-core projections |
| --- | ---: | ---: |
| 4,349-token input, prefill tok/s | 1162.3 | 1208.5 |
| Same input, client TTFT | 3.779 s | 3.637 s |
| Short code, decode tok/s | 76.9 | 77.0 |
| Short Korean, decode tok/s | 93.6 | 94.1 |

The measured prefill gain is **4.0%** on this node. Short-case decode is
essentially unchanged. Do not apply other Turing cards' reported 20–34% gains
to this machine and IQ3_XXS pack.

The full code response matched across all six runs. Korean/long responses were
not identical across the sample set; baseline repetitions also varied. The
FP16 long response had 111 tokens versus 114–122 in the baseline. This path
converts operands, clamps finite values outside FP16 range and changes rounding;
it remains a separate opt-in profile, not a bitwise-preserving default.
[Responses and timings](mtp-fp16-results.json), [metadata](mtp-fp16-metadata.json).

Across the combined, fixed-share and FP16 experiments, all 18 complete code
responses contained the same merge program. That one inspected program passed
its two embedded tests plus 100 deterministic cases including empty inputs,
duplicates and negative numbers; inputs were unchanged. Reproduce with
[check-code.py](check-code.py); [record](code-quality.json). This is one coding
task, not a broad quality benchmark. The Korean answers hit the output cap and
contain imperfect generalizations in both paths; no overall quality score is
claimed.

### Near-limit MTP input

One ordered baseline/candidate pair used the same 252,146-token three-marker
prompt, maximum 64 output tokens. Both returned exactly
`729183, 416927, 853204` (23 tokens).

| Projection path | Prefill tok/s | Client TTFT | Retrieval |
| --- | ---: | ---: | --- |
| Current BF16 path | 1368.1 | 184.580 s | all three correct |
| FP16 tensor cores | 1618.8 | 156.039 s | all three correct |

This pair observed +18.3% prefill throughput and 28.541 seconds less TTFT
(-15.5%). It is **one pair, not three-round evidence of a sustained gain**.
The context capacity was 262,144 in both arms; setting that capacity alone was
not the test. There are too few output tokens to claim a sustained decode gain.
[Responses/timings](mtp-near-256k-results.json),
[metadata](mtp-near-256k-metadata.json).

A [single snapshot during baseline prefill](mtp-near-256k-r0-baseline-memory.json)
showed 22,068 MiB GPU memory used, 1,954 MiB free, and 12.6 GiB host memory
available. The snapshot also records 70 C / 241.29 W / 1845 MHz SM clock; these
are point readings, not peaks or a power/thermal trace. Host swap use at that
point was the same 24 KiB already present before the experiments.

## Concurrent requests: keep the one-slot queue for this profile

The conservative MTP configuration, without FP16 projections, was started once
per slot count. After a short hello warmup, `tools/serve_load.py` sent one client
and then two simultaneous clients, three bursts each, maximum 256 output tokens
and fixed nonce `turing-20261111`. Prompts were short and cold. These are
end-to-end load figures, not engine-only decode rates.

| Server slots | Simultaneous clients | Aggregate tok/s | TTFT p50 / p95 | Latency p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 64.4 | 0.604 / 0.673 s | 3.978 / 4.923 s |
| 1, queued | 2 | 67.2 | 2.430 / 4.467 s | 5.678 / 7.666 s |

All nine requests completed without errors. [Load results](concurrency/slots-1-results.json),
[console](concurrency/slots-1-load.txt), [metadata](concurrency/metadata.json).
Full answers, configuration and engine/server logs are retained in that folder.

The `parallel: 2` / `--batch-mtp` attempt **did not become ready within 600
seconds**, so no requests or throughput were measured for it. The server also
started the vision encoder eagerly in this mode, despite receiving no images.
The engine reported two 0.95 GiB slot sessions and two 50 MiB draft states,
then stopped making visible startup progress after GPU detection. See the
[engine log](concurrency/slots-2-engine.txt) and
[server log](concurrency/slots-2-server.txt).

After the failed attempt, host swap use was about 216 MiB versus 24 KiB before
the experiments. This indicates memory pressure during the attempt, but the
captured logs do not establish an OOM kill or the exact failure cause. The test's
processes were stopped; no GPU compute processes remained and about 58.7 GiB
host memory was available after cleanup. No system memory settings were changed.

Keep one slot with queued requests for this node's general 256K profile. This
does not prove that every two-slot configuration is impossible; memory budgets
and a text-only batch setup would need their own investigation. The successful
short-request load test also does not prove two full 256K sessions fit.

## Exploratory SM75 kernel check (DFlash2)

This separate experiment toggles only `STRATA_SM75_PERF` between 0 and 1 on
the initial config described above. Maximum 128 output tokens, three rounds
per arm. Full response text matched for each of the three cases across all
six runs. This is a limited output check, not a numerical parity proof.

| Case | Off | On | Metric |
| --- | ---: | ---: | --- |
| Code | 75.7 | 77.0 | decode tok/s |
| Korean | 67.4 | 67.9 | decode tok/s |
| 4,349-token input | 1174.7 | 1202.5 | prefill tok/s |
| Decode after that input | 61.6 | 62.7 | decode tok/s |

[Short-run data](short-results.json), [metadata](short-metadata.json).

A separate **single pair** used a 252,146-token prompt with three access codes
placed at different positions. Both arms returned all three codes exactly.

| SM75 series | Prefill tok/s | Client TTFT | Returned codes |
| --- | ---: | ---: | --- |
| Off | 941.8 | 267.987 s | correct |
| On | 1362.9 | 185.281 s | correct |

[Near-limit data](exploratory-near-256k-results.json),
[metadata](exploratory-near-256k-metadata.json). Only 23 output tokens were
generated, so do not treat their decode timing as a sustained-throughput result.
One ordered pair is a smoke measurement, not a statistically established speedup.
The owner already enabled this switch, so this result does not justify claiming
an additional improvement to the recent MTP profile.

## Reproduction and limits

Use [qfn_turing_bench.py](../../../tools/qfn_turing_bench.py) with the metadata's
arguments and a local config restored with the correct read-only model paths.
`--candidate-config` compares two configs on the same executable; without it
the tool toggles the SM75 series. Run through `tools/turing-run.sh` with the
installed 0.1.57 server and its bundled Python 3.12 environment. The corpus is
deterministic in the script; individual outputs and timing samples are retained.

No source build, CUDA kernel-level parity suite, broad perplexity evaluation,
or sustained power/thermal/clock trace has been completed. A matching response
on these prompts is not a quality guarantee. Upstream measurements from other
GPUs are not counted as results on this node.
