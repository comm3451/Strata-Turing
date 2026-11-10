# QFN on TITAN RTX

This fork targets Qwen3.8-Flash-Next IQ3_XXS on one TITAN RTX with a
Xeon E5-2680 v4 and 64 GB DDR4 (two 32 GB DIMMs). The required context is
262,144 tokens, including the prompt and generated answer. Single-user latency
and answer quality come first. Two concurrent requests are a separate measurement.

## Keep work inside the checkout

The working directory is `/home/eesdf/Strata-Turing`. The existing Strata install
and models under `/home/eesdf` may be read, but must not be modified.
Run experiments through:

```sh
bash tools/turing-run.sh COMMAND [ARG ...]
```

This Linux wrapper uses bubblewrap to mount ordinary files outside the checkout
read-only. Temporary files, CUDA/Python package caches and XDG state go under
`.turing/`, which Git ignores. GPU devices and networking remain available.
The wrapper requires user namespaces and `bwrap`; it fails rather than running
without the read-only mounts. It does not install packages or modify `HOME`.
Give log, config, session and report paths inside the checkout explicitly.
Do not use a home-directory installer or system package manager for this fork.

## Build

The `qfn-turing` CMake preset selects Flash-Next geometry, `sm_75` native code
plus compute-75 PTX, and Strata's AVX2 CPU baseline. It inherits the release
CUDA build options and keeps the quantization kernels available. It limits
build parallelism to four jobs as an initial build-memory precaution; this is
unrelated to inference worker count. Existing presets are unchanged.

With a compatible C++ compiler, CMake 3.24+, Ninja and CUDA Toolkit available:

```sh
bash tools/turing-run.sh cmake --preset qfn-turing
bash tools/turing-run.sh cmake --build --preset qfn-turing
```

The preset is a build target, not evidence of a speedup. Record the compiler,
toolkit, driver, source commit and executable hash with each measured build.

## First comparisons

The fork starts from Strata commit `dbb2d168`. Upstream already supports Turing
and has an optional group of Turing kernels. Basalt is useful for architectural
ideas, but its current build requires Blackwell sm_120 and cannot run on TITAN RTX.

1. Reproduce the installed engine's current configuration at 262,144 context.
   Record the actual model files, KV format and resident capacity, draft mode,
   worker count, expert slots, vision state and memory budgets.
2. Compare `STRATA_SM75_PERF=0` with `STRATA_SM75_PERF=1`, restarting the engine
   between arms. Individual kernel overrides must match between arms. Upstream
   describes this group as bitwise equivalent; confirm output on this node.
3. Measure CPU expert workers around the engine's physical-core default, rather
   than choosing 28 workers from the logical CPU count. Measure memory bandwidth
   and CPU/GPU expert sharing before changing it. Two DIMMs do not establish the
   active memory-channel layout or its bandwidth.
4. Examine the 256K memory budget: GPU-resident KV versus host-backed KV, expert
   cache, prefill workspace, drafting and optional vision. Compare `--kv-grow`
   only when its documented preconditions hold. Keep INT8 KV as the reference;
   changing KV quantization requires a separate quality comparison.
5. Compare FP16 projection conversion (`STRATA_BF16_TC`) and alternate attention
   paths separately. These may change rounding and output. Do not include them
   silently in a supposedly bitwise-preserving profile.
6. Once the solo profile is measured, evaluate two requests. Record per-request
   time to first token and latency as well as aggregate throughput and the slots
   actually admitted. Keep the 256K requirement visible; a short-prompt batch
   measurement does not prove two full 256K sessions fit.

Use interleaved A/B rounds with identical prompts and output limits, a warmup,
and medians from at least three rounds. Separate cold prefill, cached follow-ups
and steady decode. Include short, 32K, 128K and near-limit prompts as practical;
reserve room for generation and record the actual token counts. Merely setting
`--max-context 262144` is not a 256K prompt test.

For quality, compare fixed-seed greedy outputs for scheduling-only changes and
use long-context retrieval, summarization, Korean text and coding/tool-use tasks
for changes to arithmetic. A single matching response is only a smoke check.
Store raw logs and local configurations under `.turing/`; publish sanitized,
reproducible measurement summaries with their limitations.

## References

- [Upstream Turing switches](DETAILS.md#rtx-20-turing-sm_75-kernels-opt-in),
  [dispatch code](../include/strata/core/sm75_series.hpp),
  [precision-changing options](OLDER_GPUS.md#quadro-rtx-8000-and-other-turing-cards).
- [Memory modes](DETAILS.md#memory-modes), [batching](BATCHING.md),
  [community benchmark reporting](COMMUNITY_BENCHMARKS.md).
- [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus) and
  [Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).
- [Basalt source and build requirements](https://github.com/jesdga95/basalt).
