# TITAN RTX + Xeon E5-2680 v4: QFN IQ3_XXS tuning

Measured on 2026-11-11 (Asia/Seoul) on the owner's inference node. These are
targeted experiments, not the Strata standard benchmark and not measurements
of a newly compiled Strata-Turing engine.

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
power measurement or thermal/clock trace has been completed. A matching response
on these prompts is not a quality guarantee. Upstream measurements from other
GPUs are not counted as results on this node.
