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

This fork also includes a project-local CUDA 13.0 / GCC 14.4 toolchain recipe:

```sh
bash tools/bootstrap_turing_toolchain.sh
bash tools/build_qfn_turing.sh
```

Both scripts enter `turing-run.sh` automatically. The bootstrap verifies the
micromamba archive and installs the exact packages in
`tools/turing-toolchain-linux-64.txt` into `.turing/toolchain`. It uses an empty
local Conda prefix with `install`, avoiding `create`'s home-directory environment
registration. It never activates a shell environment or changes a driver.
The engine is `build/qfn-turing/strata`; its toolchain libraries stay in `.turing/`.

## Native IQ prompt kernels on Turing

The CUDA dispatch adapts [c3c's upstream PR #3277](https://github.com/Niko1221/Strata/pull/3277),
revision `4450012270c81f89bbf415cc77ab04adf731e5fb`, to this fork's newer kernel
code. It admits six IQ gate/up formats with Q2_0 or IQ4_NL down on sm_75.
Each specialization must pass CUDA shared-memory and occupancy checks. IQ layers
use the qualified 128-row shape (`WW=2`) to fit Turing's 64 KiB shared-memory
limit. Other formats retain their previous fallback.

The new path requires all three switches:

```text
STRATA_PF_FUSED=1
STRATA_PF_FUSED_SM75=1
STRATA_PF_FUSED_NATIVE_SM75=1
```

It is separate from `STRATA_SM75_PERF` and the older Q2_0 opt-in. Set the new
variable to `0` to compare MMQ using the same binary. It changes prompt expert
arithmetic; answers need not be byte-identical. Decode dispatch is unchanged.
The test matrix covers all 12 IQ format pairs, Q2_0, and token counts around
tile boundaries. Dispatch tests check both opt-in gates and unsupported pairs.

Run the focused checks on a Turing GPU after building:

```sh
bash tools/turing-run.sh .turing/toolchain/bin/ctest --test-dir build/qfn-turing \
  -R 'prefill_fused_iq_(test_sm75|sm75_)' --output-on-failure
```

The source retains the upstream MIT license and attribution. The original PR's
RTX 2080 Super Max-Q speedup is not a measurement of this TITAN RTX.

## Local launch profiles

`tools/qfn_turing_profile.py` creates a separate JSON config for this exact node.
It reads the recent QFN IQ3_XXS MTP config and preserves its model paths,
tokenizer, vision, API key, network settings and conversation-cache budget.
The output must be new and inside this checkout; it is created with mode 0600.
The engine log and working directory are moved inside the checkout.

The conservative profile keeps INT8 KV with 32,768 resident cells, 262,144 total
context, MTP depth 4 / confidence 0.5, automatic PCIe sharing and the default
expert-cache adaptation. It pins the worker count to 13, keeps the existing
SM75 series and expert mode 8, and uses the 2048 MiB reserve from the recent
resolved launch. Forced share 0.10 and the combined tuner candidate were not
adopted because the mixed-workload comparison did not support them.

Use the same Python/server/engine version as the measurements. For example,
from the checkout, with your actual paths substituted:

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_profile.py \
  --source /path/to/recent-qfn-launch.json --exe /path/to/strata \
  --out .turing/launch/qfn-balanced.json
bash tools/turing-run.sh /path/to/python /path/to/serve/server.py \
  --engine strata --config .turing/launch/qfn-balanced.json
```

`--fp16-prefill` creates an **opt-in** alternative with `STRATA_BF16_TC=1`.
This converts BF16 projection operands to FP16 for Turing tensor cores, with
FP32 accumulation. Rounding and replies can change; values outside FP16 range
are clamped by the existing engine. It does not change the IQ3_XXS model files.
Keep this separate from the conservative profile and inspect the measurements
before choosing it. The original profiles were measured with the installed
0.1.57 engine; the newly compiled fork has its own results below.

The measured FP16 gain was 4.0% prefill throughput on a 4,349-token input (three
rounds). One 252,146-token pair reduced TTFT from 184.580 to 156.039 seconds;
both paths retrieved three markers correctly. The longer case needs more
repetitions and broader quality evaluation. This is an opt-in speed/rounding
tradeoff, not a universal quality-preserving acceleration.

Keep `parallel: 1` for now. Two short simultaneous requests worked through its
queue; a `parallel: 2` / `--batch-mtp` attempt at 262,144 capacity failed to
become ready within 600 seconds and was stopped. The server started vision
eagerly in that mode, and host swap use increased. Do not turn on two slots
automatically based on the short-request queue test.

## Profiles for the newly compiled engine

`--native-iq-prefill` opts into the new IQ kernels and requires this fork's
compiled engine. On the same TITAN RTX build, three alternating pairs measured
4,349-token prefill at 1157.6 → 1184.5 tok/s (+2.3%) and 31,610-token prefill at
1485.1 → 1548.5 tok/s (+4.3%). Short code/Korean generation was essentially flat.

The faster profile also adds `--fp16-prefill --turing-attention`. The latter
selects the FP32 tiled QSA scorer (`STRATA_SELECT_SIMT=2`) and the Turing v2
prompt attention kernel (`STRATA_PA_V2T=1`). It retains IQ3_XXS weights, INT8 KV,
262,144 total capacity, 32,768 resident KV cells, 13 workers and one request slot.
It changes summation order and can change generated tokens. These are explicit
options in the generated config; the engine's global defaults are unchanged.

One 252,146-token retrieval pair measured 1361.8 → 1733.7 prefill tok/s and
185.432 → 145.714 seconds to the first response; both retrieved all three
markers. With the production conversation-cache budget restored, nine short
requests from one/two clients completed without errors through one slot's queue.
Neither result establishes general 256K answer quality or two-slot capacity.

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_profile.py \
  --source /path/to/recent-qfn-launch.json \
  --exe "$PWD/build/qfn-turing/strata" --out .turing/launch/qfn-turing-fast.json \
  --native-iq-prefill --fp16-prefill --turing-attention
```

See the [new CUDA build measurements](../bench/results/2026-11-11-titan-rtx-native-iq/README.md)
for the comparison against the installed engine, numerical checks and quality
limits. `STRATA_PREFILL_ROUTED_STREAM=1` was tested but is not selected: its
median prefill gains were only 0.3% at 4K and 0.8% at 32K on this node.

### TITAN RTX pipeline profile

`--titan-throughput` includes the three fast switches above and adds the
measured CPU/GPU scheduling group, cost-aware MTP (still capped at four
tokens), and `--prefill auto:32768`. The larger prompt chunk reuses streamed
expert weights across more tokens. It borrows more of the expert cache for
temporary GPU buffers; it does not reduce the 262,144-token context limit.

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_profile.py \
  --source /path/to/recent-qfn-launch.json \
  --exe "$PWD/build/qfn-turing/strata" --out .turing/launch/qfn-turing-tuned.json \
  --titan-throughput
```

This profile is specific to this 24 GB GPU / 64 GB RAM node. It uses
`STRATA_ADAPT_OVERLAP=1`, `STRATA_SPEC_COST=1`, `STRATA_CPU_IQ_ALL_EXACT1=1`,
`STRATA_SIDE_MIXER=2`, `STRATA_B6_MMVQ_ROWS=1`, `STRATA_ATTN_DAILY=1` and
`STRATA_QSA_ROWS=1`. It explicitly restores `STRATA_ADAPT_NOWAIT=0` and
`STRATA_ADAPT_LAG=1`, which are required by the overlap schedule. Check the
engine log for `adaptive overlap` being active: explicit asynchronous-tier,
pipeline-window, DMA-batch or resident-expert overrides can prevent it.
Other hardware and memory budgets need their own comparison.

The [pipeline measurements](../bench/results/2026-11-11-titan-rtx-pipeline/README.md)
compare it with the previous fast profile using the **same executable**.
Three alternating pairs improved short code decode by 14.0%, Korean decode
by 3.2%, and 31,610-token prefill by 15.9%. A separate pair after 252,100 input
tokens improved decode of the same 466-token program from 76.4 to 84.4 tok/s.
They include larger-window MTP candidates, fresh-engine repetitions, near-256K
retrieval, answer checks and memory observations. These are measured workload
improvements, not proof of the hardware's absolute maximum throughput.

`tools/qfn_mtp_sweep.py` can explore draft windows and confidence thresholds
without restarting for every request. Its cache adapts across requests and
it uses the calibrator's direct token template, so use it to select candidates;
confirm a winner with the HTTP A/B harness below. Do not report a warmed
repeated answer as an independent configuration speedup.

## Further IQ3_XXS fusion and memory options

`--titan-fused` includes `--titan-throughput` and adds the CUDA-only
`STRATA_SM75_IQ2XS_FUSED=1`, CPU `STRATA_Q2_BITPLANE=1`, and shared prefill
scratch `STRATA_PF_HC_REGION=1`. It keeps the engine's normal arena huge-page
policy and does not change OS settings. Do not force `STRATA_ARENA_THP=1`
on fragmented RAM. Later IQ3_S trials timed out during arena startup with
and without THP, and succeeded when CUDA registration was capped at 40 GiB;
THP alone did not explain that failure. All old profile options keep their
previous behavior.

On this node, three warmed alternating pairs on the same binary measured
short code decode 102.2 to 103.7 tok/s, Korean 91.5 to 93.7 tok/s, and 31,610-token
prefill 2,206.3 to 2,232.8 tok/s. Both arms requested huge pages and used the
same memory budgets. This is a small additional improvement. A 252,146-token
retrieval pair passed in both arms but prefilled slightly slower with the new
options, 2,131.6 versus 2,108.8 tok/s. It is not faster in every case.

The [fusion report](../bench/results/2026-11-11-titan-rtx-iq2xs/README.md) includes
120 bitwise kernel checks, 72 basic answer checks, complete outputs, memory
samples and a separate two-client queue test. IQ3_S has no IQ2_XS layers, so
the new fused kernel does not directly accelerate that model's experts.

## IQ3_S on this 64 GB node

The ISTA-DASLab IQ3_S pack needs a 46.84 GiB expert arena. Full-arena starts
timed out here both with and without THP. Limiting CUDA registration to
40 GiB allowed it to start. `--titan-iq3s` uses that limit, registers the
estimated hot misses first with `STRATA_PIN_BY_NEED=1`, disables arena THP
requests, and keeps 262,144 context. The parked conversation cache is
1 GiB with a 4 GiB free-RAM floor. System settings are unchanged.

Three alternating pairs on the same IQ3_S files measured prefix pinning
versus pin by need: code decode 76.0 to 80.1 tok/s, Korean 69.4 to 71.1 tok/s,
and 31,610-token prefill 2,077.5 to 2,131.5 tok/s. Code output lengths differ;
these are workload measurements, not a pure kernel comparison. The normal
IQ4_NL N-gram table was used. BF16 N-gram download/compatibility preparation
does not establish BF16 speed or quality.

The [IQ3_S report](../bench/results/2026-11-11-titan-rtx-iq3s/README.md) records
the failed starts as well as working configurations, exact model hashes,
raw answers and memory observations. The profile source must already
reference the IQ3_S model, pack, tokenizer and chosen N-gram table; it does
not change model files implicitly.

## RustingStrata candidate

RustingStrata was built at `e1c2d6e1722a33ca75da9757761c9df6246cba60` for Broadwell
and run on the same model at 262,144 capacity. It loaded on sm_75, but currently
disables MTP for QFN's hyper-connections. After cache adaptation, its repeated
128-token code responses decoded at about 18 tok/s; the matched Strata fast
probe decoded the identical response at 69.2 tok/s. Its warm 4K TTFT was about
6.2 seconds against 3.7 seconds in the Strata probe.

Keep the Strata engine selected. This is a limited engine comparison, not a
language benchmark or a full tuning sweep. See the
[Rust compatibility and repeated-run report](../bench/results/2026-11-11-titan-rtx-rusting-probe/README.md)
for the official-release failure, latest-source success, exact configurations,
warmup effects and raw responses. RustingStrata was tested as a separate
executable; no code from its custom-licensed implementation was imported.

## Run the A/B measurement

`tools/qfn_turing_bench.py` compares the existing `STRATA_SM75_PERF` switch at
0 and 1 with a fresh server for each arm, alternating order over three rounds.
It records complete response text, engine timings and client time to first token.
Its default cases are a coding prompt, a Korean prompt and a synthetic long input.

Prepare a **separate local JSON config** under `.turing/` with an absolute `exe`,
the existing read-only model/tokenizer paths and local output paths. Start from
the **recent launch config and resolved engine command**, not an old installer
config. Set `drafting: "mtp"` explicitly for the owner's current workload; keep
its vision configuration and VRAM reserve equal between arms (no images are
sent). Use `parallel: 1` and replace the cache options with `--prompt-cache 0
--conversation-cache-mib 0` for cold-input comparisons. Do not modify the original
config. Use the same server and Python environment for both arms:

The benchmark binds to loopback and requires a separate config without an
`api_key`. Keep the generated launch profiles, which preserve authentication,
in the ignored `.turing/` directory.

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_bench.py \
  --config .turing/bench/base.json --server /path/to/serve/server.py \
  --out .turing/bench/sm75-ab --rounds 3 --long-words 4096 --max-new 128
```

The output directory must be new. To check a near-limit prompt, use another
output directory with `--only-long --retrieval --long-words 240000 --max-new 64`.
Use the **reported token count**, not the word count, when describing its length.
The three-marker retrieval is a smoke test, not a general long-context quality
evaluation. Inspect the engine command in the log: newer servers may resolve
an old MTP config to DFlash2 automatically. The harness does not freeze that
policy across server versions, so keep the server version fixed.

To compare a CPU/GPU tuning candidate against the real current configuration,
pass `--candidate-config .turing/bench/candidate.json`. Both configs must use the
same executable; their SM75 settings are preserved. Without this argument the
harness tests only the SM75 switch. Keep all changes outside the candidate's
intended tuning settings identical.

An explicit `--compare-builds` permits different executables and records both
SHA-256 hashes. Keep compiler/version comparisons separate from a kernel flag
comparison on one binary. `--cases` accepts a local JSON list of `[name, prompt]`
pairs, and `--dump-first-logits` saves the FP32 first-token logits per arm (file
0 is the warmup). These are diagnostics, not a general quality score.

## Automatic calibration and this CPU/GPU pair

`tools/calibrate.py` now confirms candidates against requests with no tuning
override. The engine can refine its PCIe share after startup, so forcing the
old share from startup `INFO` does not measure its current automatic policy.
Confirmation alternates A/B, B/A, A/B, including when a candidate equals that
startup value. If automatic sharing wins, worker-count tests keep it automatic.
The sweep explicitly includes 0.10, a useful candidate on this node.

The installed 0.1.57 calibrator was used to explore this machine before that
fix. Its confirmation numbers are retained as exploratory data, not as gains
over the owner's actual automatic configuration. Its English corpus is also
insufficient for selecting the owner's mixed Korean/code/long-input profile;
all proposed settings need a separate workload comparison. See the
[node measurements](../bench/results/2026-11-11-titan-rtx-qfn-sm75/README.md).

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
