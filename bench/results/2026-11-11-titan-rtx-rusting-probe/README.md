# TITAN RTX QFN: RustingStrata compatibility and speed probe

2026-11-11. Xeon E5-2680 v4, 64 GB DDR4, TITAN RTX 24 GiB, driver 595.99.02.
The owner's Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS shards were read-only. This is a
small comparison of usable engines on this node, **not a test of Rust versus
C++ as languages** and not a full RustingStrata tuning or quality evaluation.

## Versions and compatibility

The official [v1.2.1 release](https://github.com/RustingStudio/RustingStrata/releases/tag/v1.2.1),
source `d5afeb0386c9b42e89e29f74a876cf23df262a05`, was downloaded and its SHA-256
checked against the GitHub asset digest. It failed to load this model: its CUDA
initialization classified sm_75 as Volta and rejected NVRTC 13.0. This was a
release-version issue, not evidence that Turing requires CUDA 12.
[Release startup log](release-server.txt), [request failure](release-results.json).

The newer source at
[`e1c2d6e1`](https://github.com/RustingStudio/RustingStrata/tree/e1c2d6e1722a33ca75da9757761c9df6246cba60)
includes a Turing path and **loaded successfully**. It was compiled without
source changes using Rust 1.99.1, `-C target-cpu=broadwell`, GCC 14.4, four build
jobs and `cargo build --release --locked --bin rustingstrata`. CUDA kernels were
compiled by NVRTC 13.0.88 for sm_75 at runtime. The resulting binary SHA-256 is
`579c370b6446c4087cfe49addccbd96419ab773decf49abd4aceca7b9d23cf05`.
[Build metadata](build.json), [build log](build.txt), [successful startup](latest-server.txt).

All downloads, Rust/toolchain installations, cache files, builds and test outputs
stayed in this checkout under the read-only filesystem wrapper. System packages,
shell profiles, home configuration and existing Strata files were not changed.

## QFN-specific differences

- Both the release and the newer source disable **MTP for hyper-connection
  models**, including this QFN. The successful run explicitly reported that
  `--mtp 4` could not take effect. See
  [`resolve_flags`](https://github.com/RustingStudio/RustingStrata/blob/e1c2d6e1722a33ca75da9757761c9df6246cba60/src/main.rs#L323)
  and the startup log. The selected Strata configuration uses working MTP4.
- Multiple GPU decode slots are rejected for hyper-connections, PLE or QSA in
  [`GpuModel::load_part`](https://github.com/RustingStudio/RustingStrata/blob/e1c2d6e1722a33ca75da9757761c9df6246cba60/src/gpu_model.rs#L272).
  This is a source finding; the Rust parallel mode was not run here. A one-slot
  server queue is distinct from decoding several requests together.
- RustingStrata has separate AVX2 CPU expert kernels, expert placement, CUDA
  kernels and an HTTP server. It is more than a wrapper around Strata's engine.
  Its Turing path uses FP16 prefill GEMM; its own INT8 GEMM implementation is
  gated to sm_80+. That software gate is **not** a claim that Turing hardware
  lacks INT8 tensor cores. See
  [CUDA dispatch](https://github.com/RustingStudio/RustingStrata/blob/e1c2d6e1722a33ca75da9757761c9df6246cba60/src/cuda/mod.rs#L232).
- At 262,144 capacity the Rust plan reported 3,264 MiB KV and 8,341/24,576
  resident experts. It reads the original GGUF through memory mapping. Strata
  uses its native pack and a 32,768-cell resident KV pool at the same total
  capacity. Placement and arithmetic therefore differ even with the same model.

The project's advertised approximately 1.7x decode result compares another
model on RTX 3060 / Ryzen 7600X with llama.cpp. It does not establish a speedup
over this Strata configuration on a Xeon/TITAN RTX.
[Upstream benchmark context](https://github.com/RustingStudio/RustingStrata/blob/e1c2d6e1722a33ca75da9757761c9df6246cba60/README.md).

## Matched-input probe

Both engines used 262,144 capacity, an 8-bit KV format, 13 CPU workers/threads,
2,048 MiB VRAM reserve, one slot and no parked-conversation cache. Rust used a
1,024-token prefill chunk, a reasonable QFN starting point from its source;
Strata used its measured automatic prefill. These are not identical algorithms.
Rust's requested MTP4 was disabled by its engine; Strata's remained active.

One initial run per engine sent a warmup, the same 34-token code question,
31-token Korean question and 4,349-token seeded word list. Output caps were
128, 128 and 32 respectively. All measured prompts reported zero reused tokens.
The code response was **identical for all 128 generated tokens**; other responses
differed. All three measured responses reached their output cap, so they do not
establish completed-answer quality. Warmup is excluded: Rust's first warmup
included runtime kernel compilation and model loading.

| Work | Rust latest | Strata fast |
| --- | ---: | ---: |
| Code TTFT, s | 3.147 | 0.351 |
| Code decode, client tok/s | 12.1 | 69.2 |
| Korean TTFT, s | 1.629 | 0.273 |
| Korean decode, client tok/s | 15.9 | 77.5 |
| 4,349-token TTFT, s | 21.016 | 3.734 |
| 4,349-token prefill, engine tok/s | 207.9 | 1188.4 |

Client decode is `(generated_tokens - 1) / (total_seconds - first_seconds)`
for both engines. Prefill rates come from each engine's own timings/logs.
This initial comparison is **one pair**, with different expert-cache histories;
it should not be presented as a precise general speed ratio.
[Rust requests](latest-results.json), [settings](latest-metadata.json),
[Strata requests](strata-results.json), [settings](strata-metadata.json).

The initial Rust sampler observed up to 22,186 MiB GPU use, at least 1,836 MiB
GPU free and up to 45°C. Host `MemAvailable` is not a measure of model RSS or
physical reads: memory-mapped file pages can count as reclaimable. The OS page
cache was not cleared or otherwise controlled. Raw memory samples for both
engines are included; they do not isolate the causes of the speed difference.

## Repeated Rust run after initialization

To check the effect of expert adaptation and page-cache warmup, a second Rust
server reused its routing profile and compiled-kernel cache. It ran a warmup
and then code → Korean → 4K **three times in the same process**, with the same
caps and settings. All nine measured requests completed and reported zero
prompt tokens reused. Every code response matched the same 128-token Strata
answer. The three rounds were:

| Work | Round 1 | Round 2 | Round 3 |
| --- | ---: | ---: | ---: |
| Code, client decode tok/s | 13.0 | 18.0 | 18.1 |
| Korean, client decode tok/s | 15.2 | 19.4 | 19.8 |
| 4K prefill, engine tok/s | 220.4 | 711.9 | 719.3 |
| 4K TTFT, s | 19.836 | 6.178 | 6.119 |

Warm state clearly matters: **the initial 21-second Rust TTFT is not its warm
performance**. Even the last two rounds did not beat the matched Strata probe
above (69.2 code tok/s, 3.734-second 4K TTFT). This is still an exploratory
comparison, not three alternating pairs of engines, and it does not separate
OS caching from expert adaptation. It supports retaining the current engine
without asserting a universal ratio or a fully tuned Rust ceiling.
[Repeated requests](latest-warm-results.json), [server timings](latest-warm-server.txt),
[configuration](latest-warm-metadata.json), [computed statistics](summary.json).
Run `python3 -B summarize.py` in this report directory to recompute statistics.

## Decision and limits

Retain the measured Strata fast profile for this node. A Rust implementation
could improve particular scheduling or kernel paths, but the engine tested
here does not provide QFN's existing MTP path and has not demonstrated a benefit
over it. This is not a claim that every RustingStrata configuration is slower.
The Rust probe did not test 252K actual input, multiple clients, perplexity,
vision, tool calling, or a CPU-thread/chunk-size tuning sweep.

Uses RustingStrata by Vasyl Trefilov. It was tested as a separate executable;
no RustingStrata implementation was incorporated into this fork. Its
[custom license](https://github.com/RustingStudio/RustingStrata/blob/e1c2d6e1722a33ca75da9757761c9df6246cba60/LICENSE.md)
remains with its local source checkout and differs from this engine's MIT license.
