# TITAN RTX: IQ2_XS fused experts and CPU/prefill options

Measured on 2026-11-11 with one TITAN RTX 24 GiB, Xeon E5-2680 v4
(14 physical cores, 28 threads), and 64 GB DDR4. QFN ISTA-DASLab IQ3_XXS,
262,144 total context capacity, INT8 KV with 32,768 GPU-resident cells,
13 CPU expert workers, MTP window 4/min-p 0.5, and 2 GiB VRAM reserve.
Vision stayed configured but no images were sent. No OS settings were changed.

This adds a CUDA-only, opt-in IQ2_XS gate/up + SwiGLU + q8_1 fused path for
QFN shapes on actual sm_75 devices. Ten of this IQ3_XXS model's 48 layers use
that format. The selected profile also enables the existing CPU Q2 bit-plane
path and shared prefill scratch for hyper-connections. These are small further
gains over the [previous pipeline profile](../2026-11-11-titan-rtx-pipeline/README.md),
not a large decode speed increase or a measured hardware ceiling.

## Controlled performance comparison

`steady-ab` is the main comparison: three alternating pairs (A/B, B/A, A/B),
five prompts each, same model files, executable, server, sampling and memory
budgets. Both arms request transparent huge pages for the expert arena with
`STRATA_ARENA_THP=1`. Three separate 256-token binary-search warmups precede
each arm. Neither parked conversation cache nor prompt cache is enabled;
all 30 measured requests report zero reused tokens.

The candidate changes only `STRATA_SM75_IQ2XS_FUSED=1`,
`STRATA_Q2_BITPLANE=1`, and `STRATA_PF_HC_REGION=1` from the reference arm.
The executable SHA-256 is
`1b50bc01e1f3b6957b6fe8a58b577096bb5546278899691e3ab6f2578cfc4c49`.
It was built for sm_75 with CUDA 13.0.88 and GCC 14.4. Both arms use the same
installed Python server 0.1.57; hashes, arguments and complete configs are in
[steady-ab-metadata.json](steady-ab-metadata.json).

| Workload | Reference median | Candidate median | Change |
|---|---:|---:|---:|
| Short code decode, same complete 493-token answer | 102.2 tok/s | 103.7 tok/s | +1.5% |
| Korean decode, 512-token cap | 91.5 tok/s | 93.7 tok/s | +2.4% |
| 4,349-token prefill | 1,202.7 tok/s | 1,224.1 tok/s | +1.8% |
| 31,610-token prefill | 2,206.3 tok/s | 2,232.8 tok/s | +1.2% |
| Code after 4,393 input tokens | 89.7 tok/s | 91.1 tok/s | +1.6% |

Short code won two of three pairs; Korean and both long prefill cases won
three of three. The last row has different 367–374-token answers, and the
candidate lost two of three paired runs despite the higher median. Do not
describe these results as a uniform gain. Korean performance answers reach
the cap; separate uncapped short-answer checks below cover basic correctness.
Client TTFT, elapsed time and all per-run metrics are in
[summary.json](summary.json) and [steady-ab-results.json](steady-ab-results.json).

The 2-second memory samples show peak VRAM 22,194 MiB and minimum available
host RAM 11.93 GiB. Swap already contained about 175 MiB; its used-size range
changed by only 4 KiB, which is not a measurement of page-in/page-out traffic.
Post-warmup engine `smaps_rollup` records confirm roughly 39.7–40.0 GiB of
anonymous huge pages in every arm. Later IQ3_S trials exceeded the 600-second
readiness deadline during arena startup, both with and without THP. Capping
CUDA registration at 40 GiB allowed startup; THP alone did not explain the
failure. The published launch profile does **not** force THP and retains
the engine's fragmentation guard. The table describes the recorded huge-page
condition, not a promise of these rates with a different page allocation.

## Near 256K and two clients

Both configurations correctly recovered all three markers from **252,146
input tokens**. This separate pair used the production 7 GiB parked-cache
budget, otherwise matched. Reference/candidate prefill was **2,131.6 / 2,108.8
tok/s**, and client TTFT **118.56 / 119.85 seconds**: no improvement at this
length. The 23-token retrieval answer is too short to estimate sustained
decode throughput. Peak VRAM was 22,174 MiB; minimum available RAM 11.87 GiB.
This is a retrieval smoke test, not general quality at 256K. Recreate it with
the harness's `--only-long --retrieval --long-words 240000` options.

The selected configuration was tested separately with one engine slot and
one or two concurrent HTTP clients, three rounds each, 256 output tokens.
There were no request errors. Two clients queue on the one slot:

| Clients | Aggregate output | Median TTFT | Median completion time |
|---|---:|---:|---:|
| 1 | 68.3 tok/s | 0.445 s | 3.750 s |
| 2 | 70.1 tok/s | 2.279 s | 5.434 s |

These short load prompts differ from the performance table. Two clients do
not double throughput, and this does not establish that two full 256K engine
slots fit. See the `concurrency-*` files for outputs, configuration and logs.

## Arithmetic and output checks

- All **120 real-layer checks** pass bitwise: ten IQ2_XS layers, token counts
  1/2/3/4/5/8, 10/20 expert groups, capacity 40. Compare mode 0 against fused
  mode 8. Five format regression cases also keep identical output hashes
  with the switch off and on. See `run-fused-checks.py` and `fused-*.txt`.
- The existing CPU Q2 bit-plane parity test passes (`cpu-parity.txt`).
- **72 basic answer checks pass**: code, Korean, arithmetic, JSON, tool-call
  JSON and logic, at about 0.6K/5K/20K input lengths, two rounds in each arm.
  The reviewed merge functions pass 100 seeded cases without mutating input.
  The initial strict checker refused a safe slice/extend variant and a
  correctly worded Korean explanation; these were inspected before allowing
  their exact AST/text. No unreviewed generated code is executed.
- The 12 performance code responses contain five inspected complete programs.
  Their embedded tests and 100 seeded merge cases each pass (`check-code.py`).
- First-token logits are not globally byte-identical. Over 36 A/B comparisons,
  KL mean/max are 0.004446/0.029785 and top-1 agrees in 35/36. Repeated baseline
  runs have KL mean/max 0.005433/0.023070 and agree in 18/18. Dynamic placement
  and arithmetic vary; subtracting these means does not establish a quality
  gain. The comparison skips all three warmup logit dumps.

Downloads ran during the 72-answer quality check, so those timings must not
be used as speed evidence. Downloads were paused for performance runs.
There is no general coding benchmark, perplexity evaluation or proof of
unchanged model quality here. HIP/SYCL were not built on this CUDA node; the
new CUDA templates and dispatch are excluded from HIP.

## Candidate selection and reproduction

`bitplane-pilot`, `expdec-pilot`, and `hcregion-pilot` are one-pair exploratory
runs. EXPDEC's real-data microchecks pass, but its end-to-end results were
mixed, so it is not in the selected profile. The shared scratch option reduced
the logged 32K prompt expert-loan allocation from 6,617 slots / 10.67 GiB to
5,417 slots / 8.74 GiB. This is allocation evidence, not a separate speedup.

`final-ab` was an earlier startup experiment, despite its name. Its first
reference arena did not request THP, and startup calibration differed. Keep
it as diagnostic data; use `steady-ab` for the table above.

The new profile is explicitly selected with `--titan-fused`. All 16 combinations
of the pre-existing generator options were compared with commit `838b741f`
and produce unchanged configs. The new profile's tuning args/env match the
tested production candidate except for the deliberately omitted forced THP
request. Global defaults are unchanged.

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_profile.py \
  --source /path/to/current-qfn-config.json --exe "$PWD/build/qfn-turing/strata" \
  --out .turing/launch/qfn-turing-fused.json --titan-fused

# Use the same Python/server version and local baseline/candidate configs.
bash tools/turing-run.sh /path/to/python tools/qfn_turing_bench.py \
  --config .turing/bench/nextopt/new-base-thp.json \
  --candidate-config .turing/bench/nextopt/selected-thp.json \
  --server /path/to/serve/server.py --out .turing/bench/reproduce-fused \
  --rounds 3 --max-new 512 --warmup-rounds 3 --warmup-tokens 256 \
  --cases bench/results/2026-11-11-titan-rtx-iq2xs/steady-ab-prompts.json

bash tools/turing-run.sh python3 -B bench/results/2026-11-11-titan-rtx-iq2xs/check-code.py
bash tools/turing-run.sh python3 -B bench/results/2026-11-11-titan-rtx-iq2xs/check-quality.py
bash tools/turing-run.sh python3 -B bench/results/2026-11-11-titan-rtx-iq2xs/summarize.py
```

The downloader has its own manifest with pinned revisions, expected sizes
and SHA-256 hashes. IQ3_S uses different expert formats and has no IQ2_XS
layers; do not attribute IQ3_S throughput to the new type-17 fused kernel.
