# TITAN RTX QFN: decode scheduling and 32K prefill chunks

2026-11-11, Xeon E5-2680 v4 (14 cores / 28 threads, AVX2), TITAN RTX
(24,576 MiB), 64 GB DDR4. Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, using the
owner's existing model shards and native expert pack as read-only inputs.

Against this fork's previous fast profile, the new pipeline profile improved
code decode by **14.0%**, Korean decode by **3.2%**, and 31,610-token prefill
by **15.9%** in three alternating pairs. Both arms used the **same engine
binary**. These are additional gains over the previously published native-IQ,
FP16-projection and Turing-attention profile, not a comparison with stock Strata.

## Selected profile

`tools/qfn_turing_profile.py --titan-throughput` includes the earlier three
fast options and adds:

```text
--prefill auto:32768
STRATA_CPU_IQ_ALL_EXACT1=1
STRATA_SIDE_MIXER=2
STRATA_B6_MMVQ_ROWS=1
STRATA_ATTN_DAILY=1
STRATA_QSA_ROWS=1
STRATA_ADAPT_OVERLAP=1
STRATA_ADAPT_NOWAIT=0
STRATA_ADAPT_LAG=1
STRATA_SPEC_COST=1
```

The CPU/GPU work and adaptive expert copies can overlap; the cost-aware MTP
policy chooses drafts within the existing four-token cap. The larger prompt
chunk reuses streamed expert weights across more tokens. Startup confirms a
32,768-token chunk with a 512-slot streaming ring; it can borrow roughly
10.67 GiB of the existing GPU expert cache for prompt buffers. The memory
budget and context capacity are not reduced.

This is an explicit node profile, not a change to the engine's global defaults.
All eight combinations of the previous profile options were compared with
the prior generator and remained unchanged. The launch helper retains the
previous configurations for comparison and rollback.

## Method and repeated results

Engine **0.1.59.1**, CUDA **13.0.88**, GCC **14.4.0**, driver **595.99.02**,
`75-real;75-virtual`, AVX2. Binary SHA-256:
`875efc6063f1977648bea649664b21e3745a278406c226e7a06b8d8801ab3420`.
The binary is unchanged from the
[native-IQ build report](../2026-11-11-titan-rtx-native-iq/README.md).
The installed 0.1.57 Python/server was used throughout.

Both arms retained **262,144 total context**, INT8 KV, **32,768 resident KV
cells**, 13 CPU workers, automatic PCIe sharing and expert cache, mode 8,
MTP depth 4 / confidence 0.5, 2,048 MiB VRAM reserve and one engine slot.
Vision configuration remained present, with no image requests. Performance
and quality tests disabled prompt/conversation caches; every request read
its full input. Greedy sampling and thinking-disabled HTTP calls were used.

The performance comparison starts a fresh engine for each arm, warms it with
16 tokens of "Say hello", then runs five fixed prompts with a 512-token answer
cap. Order: A/B, B/A, A/B. Medians of three requests per arm/case:

| Work | Previous fast | Pipeline profile | Change |
| --- | ---: | ---: | ---: |
| Code decode, tok/s | 76.2 | 86.9 | +14.0% |
| Korean decode, tok/s | 93.2 | 96.2 | +3.2% |
| 4,349-token prefill, tok/s | 1204.9 | 1206.3 | +0.1% |
| 31,610-token prefill, tok/s | 1929.9 | 2236.7 | +15.9% |
| 31,610-token TTFT, s | 16.444 | 14.192 | -13.7% |

All 30 requests succeeded with zero reused tokens. The short code response
was identical and complete at 478 tokens in all six runs. Its baseline range
was 66.4–76.4 tok/s, versus 85.0–86.9 for the candidate; the first baseline
had a slower measured CPU expert cost and chose more PCIe work. Automatic
sharing was retained in both arms, so this variation is part of the result.
The gain held in each of the three pairs.

The Korean throughput response reached its 512-token cap; separate completion
checks below use shorter tasks. Long-summary and long-code responses can differ,
so their decode rates are recorded without calling them identical-work gains.
Long prompts are seeded synthetic word lists, not a representative corpus.

[Responses and timings](final-ab-results.json), [configurations and hashes](final-ab-metadata.json),
[prompts](final-ab-prompts.json), [memory samples](final-ab-memory.jsonl).

Across the repeated performance runs, two-second samples observed at most
**22,194 MiB GPU use**, at least **1,828 MiB GPU free**, at least **12.04 GiB
host available**, and at most **68°C**. Approximately 209 MiB of swap was
already allocated at the beginning and the recorded allocation did not rise.
That is not a measurement of page-in/page-out traffic. Sampled extrema are
not allocator peaks.

## Answer checks

Eighteen prompts cover code, Korean, arithmetic, JSON, elementary logic and
tool-call-shaped JSON at **631–645**, **5,043–5,057** and **20,427–20,441**
input tokens. The largest group exercises the changed prefill chunk shape.
Two alternating pairs completed **72/72 requests**, without truncation, and
passed the recorded content checks. These are simple smoke tasks, not a broad
coding, reasoning or actual tool-execution evaluation.

The code checker executes only an exact, reviewed AST after removing comments
and docstrings. One answer added builtin `list` type annotations; the original
gate refused to execute it. That answer was inspected, and only that exact
annotated form was added as a second allowed AST. Both forms implement the
same merge algorithm. Every checked function passed 100 seeded list pairs and
input-preservation checks. The separate performance-code check passed two
embedded tests plus 100 seeded pairs, guarded by the previously reviewed code
hash. All eight unique short Korean explanations were read manually.

For 36 candidate/reference first-token comparisons, mean KL was **0.005967**,
max **0.037801**, with **34/36** equal top-1 tokens. Repeating the baseline
gave mean KL **0.006434**, max **0.027971**, and **16/18** equal top-1 tokens.
CPU/GPU placement and adaptation vary even within the baseline. Do not
subtract these means to infer a quality gain. The candidate is not byte-identical;
first-token checks do not establish full-sequence or general long-context quality.

[Prompts](quality-prompts.json), [full answers](quality-results.json),
[content checks](quality-checks.json), [logit comparisons and hashes](quality-logits.json).
The raw FP32 logit files stay in `.turing/bench/maxspeed/quality` locally.
The quality run's two-second samples observed at most 22,188 MiB GPU use,
at least 12.03 GiB host available and at most 75°C.

## Near-256K retrieval

One fresh-engine pair read **252,146 tokens** at 262,144 capacity. The
7,168 MiB conversation-cache budget / 4,096 MiB floor was restored in both
arms; prompt caching remained disabled, and reuse was zero. Both returned
exactly `729183, 416927, 853204`, the three markers spread through the document.

| Work | Previous fast | Pipeline profile | Change |
| --- | ---: | ---: | ---: |
| Prefill, tok/s | 1731.0 | 2121.4 | +22.6% |
| TTFT, s | 145.933 | 119.135 | -18.4% |

This saved **26.8 seconds** before the first response in this one pair. It is
one simple retrieval task, not broad long-context validation. The 23-token
answers decoded at 56.6 and 51.0 tok/s respectively; such a short answer is
insufficient to establish sustained decode performance at this length.

Two-second samples observed at most 22,156 MiB GPU use, at least 1,866 MiB GPU
free, at least 11.96 GiB host available and at most 77°C. Swap allocation did
not exceed its initial approximately 209 MiB during the recorded samples.
[Responses](near-256k-results.json), [configuration](near-256k-metadata.json),
[memory samples](near-256k-memory.jsonl).

## Code generation after 252K tokens

A second long-input pair replaced the final retrieval question with the code
task. Each arm read **252,100 tokens** and emitted the **same complete
466-token program**. This checks longer decoding at the occupied context,
separately from the short retrieval answer above.

| Work | Previous fast | Pipeline profile | Change |
| --- | ---: | ---: | ---: |
| Decode, tok/s | 76.4 | 84.4 | +10.5% |
| Prefill, tok/s | 1738.9 | 2122.9 | +22.1% |
| TTFT, s | 145.250 | 119.018 | -18.1% |
| Total response time, s | 151.329 | 124.516 | -17.7% |

The full program was reviewed before execution and guarded by its SHA-256.
Both copies passed their two embedded cases, 100 seeded list pairs and input
preservation. This remains **one pair on one code task after synthetic filler**;
it is not a general long-document coding evaluation.
The sampler observed at most 22,176 MiB GPU use, at least 1,846 MiB free,
at least 12.02 GiB host available and at most 76°C.
[Responses](near-code-results.json), [configuration](near-code-metadata.json),
[code checks](near-code-checks.json), [memory samples](near-code-memory.jsonl).

## Production caches and two queued clients

Nine short requests completed with the production prompt-cache setting and
7,168 MiB conversation-cache budget restored. Authentication was removed only
from the loopback test copy. RPATH resolved the same runtime libraries as the
benchmark's explicit library directories; see [resolved paths](runtime-libraries.json).
One engine slot served clients 1 and 2, three rounds each, with 256-token caps.

| Clients | Median total tok/s | TTFT p50 / p95, s | Completion p50 / p95, s | Errors |
| --- | ---: | ---: | ---: | ---: |
| 1 | 65.9 | 0.70 / 0.73 | 3.88 / 4.31 | 0 |
| 2 | 66.1 | 2.66 / 4.64 | 5.82 / 7.87 | 0 |

This total rate includes prompt and queue time. Two callers share one queue;
they do not decode simultaneously. There is no paired concurrency baseline
here and no test of two full 252K sessions. The samples observed at most
22,102 MiB GPU use and at least 11.10 GiB host available.
[Results](concurrency-results.json), [config](concurrency-config.json),
[load log](concurrency-load.txt), [memory samples](concurrency-memory.jsonl).

## Candidate selection and profiling

The isolated pilots are candidate-selection evidence, not substitutes for the
three-pair combined result above:

- Five CPU/CUDA switches together: one pair, code 70.0 → 73.2 tok/s, Korean
  94.2 → 95.6. Their individual contributions were not isolated.
- `auto:32768` alone: one pair, 31,610-token prefill 1945.1 → 2262.6 tok/s;
  4K was essentially unchanged.
- Adaptive overlap alone: one pair, code 77.0 → 82.7 tok/s, Korean
  94.1 → 97.1. Short summary outputs differed and had mixed decode rates.
- With the CPU/CUDA group and 32K chunk, overlap was preferred to the
  two-window-lag/nowait alternative: code 86.9 versus 86.4, Korean 98.2 versus
  97.5, long-code 107.6 versus 103.2 tok/s. This is only one pair.
- Cost-aware MTP on top of that stack gave 87.8 code / 101.6 Korean / 107.5
  long-code tok/s in one further run. Its separate gain is not established;
  the complete selected stack was then tested against the previous fast profile.

Each pilot has `*-results.json`, `*-metadata.json` and per-engine logs beside
this report. [Pilot prompts](pilot-prompts.json) include the 4K-background
coding request. The final comparison puts that request last; the scheduling
and cost pilots put it third, so their rates should not be compared as if cache
history were identical.

`tools/qfn_mtp_sweep.py` explored windows 4, 6 and 8, confidence 0.5/0.7,
three prompts, two rounds: **36 requests**. The direct calibrator token template
and adaptive cache history differ from the final HTTP benchmark. In its second
round at confidence 0.5, code/Korean rates were 113.8/95.3 at window 4,
108.4/83.1 at 6, and 95.9/77.9 at 8. At confidence 0.7, window 4 also led both
short tasks. The repeated code response sped up after cache adaptation even
without changing confidence; that is not a configuration gain. The four-token
cap was retained. [Full sweep](mtp-sweep-results.jsonl), [metadata](mtp-sweep-metadata.json).

The timing-instrumented baseline run located CPU experts and synchronization
as substantial decode costs. For its code request, one window averaged
54.79 ms, with 48.33 ms in verification, including 27.08 ms attributed to CPU
expert work; draft generation took 2.95 ms. The timers overlap and profiling
changes execution, so these are diagnostic figures, not an additive budget or
normal throughput. [Profile log](profile-r0-sm75-1-engine.txt).

The scheduling options are documented in [upstream DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
and implemented in this fork's `src/program/generate.cpp`, `src/core/verify.cpp`
and `src/kernels/cpu/native_expert.cpp`. Upstream speedups were not substituted
for measurements on this node. An EXPDEC benchmark was built but not run:
source inspection shows the selected fused mode-8 path bypasses much of it.
No EXPDEC speedup is claimed.

## Reproduce

Run commands from the checkout through `tools/turing-run.sh`. Configurations
must refer to the measured binary/model/server and contain no API key for the
loopback benchmark. Start from the two configs in `final-ab-metadata.json`.
For example, with their paths substituted:

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_bench.py \
  --config .turing/bench/base.json --candidate-config .turing/bench/candidate.json \
  --server /path/to/0.1.57/serve/server.py --out .turing/bench/repeat \
  --cases bench/results/2026-11-11-titan-rtx-pipeline/final-ab-prompts.json \
  --rounds 3 --max-new 512
```

Use `quality-prompts.json`, `--rounds 2 --max-new 384 --dump-first-logits` for
the answer checks. Use `--only-long --retrieval --long-words 240000` for the
near-limit input; restore the 7,168 MiB conversation-cache budget in both
configs for that run. `monitor.py` records GPU and host-memory samples at
two-second intervals around a command. Sampled extrema are not allocator peaks.

To reproduce the long code input, generate its cases file below, then pass it
to the same harness with `--cases`, `--rounds 1 --max-new 512`, and the restored
conversation-cache budget in both arms. The generator checks the exact JSON
hash against [the recorded manifest](near-code-prompt-manifest.json).

```sh
bash tools/turing-run.sh /path/to/python \
  bench/results/2026-11-11-titan-rtx-pipeline/make-near-code-prompt.py \
  --out .turing/bench/near-code-cases.json
```

```sh
python3 -B bench/results/2026-11-11-titan-rtx-pipeline/check-code.py
python3 -B bench/results/2026-11-11-titan-rtx-pipeline/check-quality.py
python3 -B bench/results/2026-11-11-titan-rtx-pipeline/check-near-code.py
python3 -B bench/results/2026-11-11-titan-rtx-pipeline/summarize.py
```

`compare-logits.py .turing/bench/maxspeed/quality` recomputes the distribution
comparison from the local FP32 dumps. It requires NumPy.

All downloads, builds, generated configurations, logs, caches and reports
stayed inside `/home/eesdf/Strata-Turing`. Ordinary files outside it were
mounted read-only. No driver, clock, power limit, system package or original
Strata installation was modified. The local production launch config preserves
authentication and stays ignored with mode 0600.

On the measured node, `bash .turing/launch/start-qfn.sh turing-tuned` selects
this profile, as does invoking that helper without a mode. `turing-fast` retains
the previous fast profile. No production server was left running by these tests.

This does not establish the hardware's absolute maximum speed, general model
quality, full-corpus perplexity, vision performance or two-slot 256K capacity.
