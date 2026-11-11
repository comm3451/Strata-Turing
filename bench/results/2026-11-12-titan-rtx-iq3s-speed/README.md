# IQ3_S/BF16 serving speed on TITAN RTX

Measured on 2026-11-12 on one TITAN RTX 24 GiB, Xeon E5-2680 v4
(14 cores / 28 threads, AVX2), and 64 GB DDR4. The model stays
ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S with BF16 N-grams,
INT8 KV and 262,144 total context tokens. Single-user latency comes first.

## What this comparison holds fixed

All arms use the same sm_75 executable, SHA-256
`1b50bc01e1f3b6957b6fe8a58b577096bb5546278899691e3ab6f2578cfc4c49`,
built with CUDA 13.0.88/GCC 14.4, and the installed Python server 0.1.57.
Work started from `238e4da7fb2df0635cfc3a00f3ff953f1f407b8d`. The final
confirmation follows policy commit `be27ef3c8cd0fc22fc94bb96366cce8bb26cabf2`.
Driver: 595.99.02. Model identities and the BF16 table SHA-256 are in the
[BF16 report](../2026-11-11-titan-rtx-bf16-ngram/README.md).
The owner requires these five explicit settings in every final comparison:

```text
STRATA_ROUTE_PRIOR=0
STRATA_ROUTE_TAIL_SKIP=0
STRATA_KVA=0
STRATA_PREFILL_CPU_SHARE=0
STRATA_HC_Q8=0
```

They were also read from the actual running engine's process environment.
The source resolves each zero to off. The serving config and both node
profile generators enforce them. In particular, the CUDA serving default
of tail-skip rank 7 must not return when a profile is regenerated.
Model files and KV precision remain unchanged between the final arms.

The new reference keeps the serving configuration, with all five owner flags off: stock DFlash2 window
6/confidence 0.5, 13 workers, automatic PCIe sharing and expert cache,
32,768 resident KV cells, 32K prompt chunks, 40 GiB hot-expert pinning
without forced THP, 2,048 MiB additional VRAM reserve, and prompt cache 0.
The parked-conversation budget was 1,024 MiB with a 4,096 MiB RAM floor;
cache 0 prevented it from retaining conversations.

The main performance cases use identical prompts, greedy sampling, thinking
disabled and a 768-token output cap. Each fresh engine has three 256-token
warmups. The Korean answer reaches the cap: it measures throughput, not a
complete-answer quality score. Generated answer lengths can differ when
more experts run on the GPU instead of the CPU; total latency differences
are not pure execution-speed ratios. The 4K/32K inputs are synthetic word
lists, not a representative coding repository or reasoning benchmark.

Initial screening used the previous feature defaults, including tail skip 7.
It was interrupted when the owner disabled these features. Its measurements
are historical diagnostics only; they do not establish a gain for the new
configuration. The final screening and confirmation use fresh engines with
all five flags at zero.

## Screening with all five owner flags off

The new single-run screening produced the following decode rates. The
performance inputs are the same in every row. These are exploratory values,
not the confirmation medians below.

| Candidate | Short code | Korean | 4K summary | 32K summary | Code after 4K |
|---|---:|---:|---:|---:|---:|
| Reference, 2,048 MiB reserve | 74.9 | 62.6 | 42.1 | 63.5 | 56.5 |
| 1,024 MiB reserve | 79.4 | 64.5 | 39.9 | 62.4 | 60.9 |
| 512 MiB reserve | 81.3 | 65.9 | 43.7 | 60.8 | 65.0 |
| Fetch admission, overlap off, 2,048 MiB | 83.0 | 60.1 | 50.0 | 66.0 | 66.9 |
| Fetch admission, overlap off, 512 MiB | 90.5 | 65.4 | 42.4 | 62.2 | 74.5 |

The combined 512 MiB candidate is selected for confirmation. It uses
`STRATA_ADAPT_FETCH=2` and `STRATA_ADAPT_OVERLAP=0`; the latter is required
for admission to activate. This changes weight placement and transfer
scheduling. It does not enable routing bias or omit tail experts. More
experts executing on the GPU instead of the CPU can still change rounding
and output text. Both pilot merge functions pass 100 seeded input cases.

The final serving candidate additionally enables two prompt checkpoints,
a 1,024 MiB parked-conversation budget and a 3,072 MiB free-RAM floor.
Only chats of at least 4,096 tokens are parked. A parked prefix must advance
4,096 tokens before replacing its saved copy. Thirteen workers, automatic
PCIe sharing, stock DFlash2 window 6/confidence 0.5, model files, BF16
N-grams, INT8 KV and 262,144 context stay unchanged.

## Confirmed decode and cold prefill

Three alternating pairs (A/B, B/A, A/B), 30 requests, all successful and
all reporting zero reused tokens. Both arms keep the five owner flags at
zero. The candidate changes only the additional VRAM reserve from 2,048
to 512 MiB, enables fetch admission at 2, and disables adaptive overlap.
Prompt caching remains off in both arms for this comparison.

| Median of three | Reference | Candidate | Change |
|---|---:|---:|---:|
| Short code decode | 75.0 tok/s | 86.9 tok/s | +15.9% |
| Korean decode, capped | 62.7 tok/s | 62.8 tok/s | +0.2% |
| 4,349-token summary decode | 42.8 tok/s | 49.1 tok/s | +14.7% |
| 31,610-token summary decode | 63.8 tok/s | 67.9 tok/s | +6.4% |
| Code decode after 4,393 input tokens | 56.7 tok/s | 71.0 tok/s | +25.2% |
| 4,349-token cold prefill | 1001.4 tok/s | 1018.0 tok/s | +1.7% |
| 31,610-token cold prefill | 2156.4 tok/s | 2174.9 tok/s | +0.9% |

Candidate short-code decode ranges from 86.9 to 90.3 tok/s, compared with
74.9 to 75.1 for the reference. Code after 4K input ranges from 70.9 to
74.5, compared with 56.7 to 57.0. All three pairs favor the candidate for
both code cases. Korean is effectively tied in the medians. The summary
rates vary with answer text and length; the first pair's summary decode
was slightly slower, so this is not a gain on every individual request.

All twelve code answers in this comparison pass the reviewed merge AST
check and 100 seeded input cases each. Other performance responses are
retained for inspection; the throughput suite is not a broad accuracy test.

## Repeated conversations

The cache comparison completed three alternating pairs and 54/54 content
checks. Both arms use the selected 512 MiB / fetch-admission decode settings.
The disabled arm sets prompt cache 0; the enabled arm sets 2. All other
cache limits, model files, sampling and required zero flags match.

| Client time to first token, median of three | Cache off | Cache on |
|---|---:|---:|
| First 31,670-token input | 14.979 s | 15.117 s |
| Repeat the same input | 14.469 s | 0.515 s |
| Continue the conversation | 14.538 s | 0.444 s |
| Change the last question | 14.586 s | 0.794 s |
| Small unrelated question | 0.308 s | 0.659 s |
| Return after the unrelated question | 14.594 s | 0.678 s |
| Change a value near the start of the document | 14.666 s | 14.935 s |

Two clients were also submitted together in each arm and round. They share
one actual serving slot. Reported by first/second response order, their
median TTFTs are 14.618/29.314 seconds without cache and 0.190/0.918 seconds with cache.
Both clients return the correct different values. Request order is not
fixed, so the sorted response-order figures avoid assigning queue delay to
one arbitrary client name.

These are prefix-reuse latency gains, not equivalent increases in raw
prefill throughput. A changed early prefix correctly incurs another cold
read and returns the new value. Small side questions cost some cache-state
copy time; preserving the long conversation makes returning much faster.

## Near the configured context limit

The serving candidate first processed a separate 31,675-token conversation,
then retained that smaller conversation while reading 252,146 tokens.
All seven retrieval checks returned the expected values. The cold long
input took 123.388 seconds to its first token, with 2,048.0 tok/s prefill.
Repeating it took 0.195 seconds, and a follow-up took 0.417 seconds while
reusing 252,168 tokens. These short retrieval answers are not sustained
decode-speed measurements.

Changing the final question at this length reused only the 32,768-token
checkpoint and read 219,423 tokens again. Its TTFT was 110.109 seconds.
The log reports a 3,757 MiB snapshot, larger than the 1 GiB parked budget.
The two differently worded queued questions also needed substantial
recomputation. Cache reuse at 32K does not imply the same latency after
edits at 252K; ordinary continuation and rewinding are different cases.

A separate code request read 252,100 tokens with zero reuse, at 2,036.9
tok/s prefill and 124.077 seconds TTFT. It generated 580 tokens at 86.4
tok/s and finished in 130.753 seconds. Its merge function passed the
reviewed AST check and 100 seeded inputs. This is one candidate run, not
a paired long-context speed comparison or a general coding evaluation.

## Answer checks and memory

The final serving profile passed 18 small checks: three each for Python
merge code, Korean explanation, arithmetic, logic, JSON and tool-shaped
JSON. The Korean paraphrases were read manually and recorded in the
checker. Arithmetic checks the resulting value, not a strict number-only
format; tool-shaped JSON is not an end-to-end tool-call protocol test.
Together with the performance and near-limit code cases, 16 generated
merge functions passed 100 seeded inputs each without changing the inputs.
The 9 pilot, 54 paired-cache and 7 near-limit retrieval responses also
matched their expected values.

Across the final validation runs, the maximum sampled GPU use was
23,676 MiB (23.12 GiB); the lowest host MemAvailable was 3.51 GiB.
Near-limit continuity with the smaller chat parked reached 23,660 MiB
and 3.65 GiB available RAM. The samples recorded no new swap-out pages
in these validation runs. Small host-wide page-in deltas were present
(5–735 pages per run), so these counters are not a claim of zero paging.
Two-second sampling can miss short peaks; the memory budget remains tight.

The selected profile was applied to the checkout-local configuration read
by `strata.service`. After the owner supplied the required interactive
administrator authentication to start it, the service was active/running.
The actual engine process had all five required flags at zero, fetch
admission at 2 and overlap at 0. The authenticated API reported IQ3_S,
262,144 context and one serving slot, and answered a synthetic check
correctly. `production-verification.json` records the safe fields only;
credentials and private conversations are not included.

## Dedicated-node GPU checks

The GPU reports display activity disabled. This node runs Linux; NVIDIA's
[TCC driver-model setting is Windows-only](https://docs.nvidia.com/deploy/nvidia-smi/index.html#driver-model).
No TCC switch, driver replacement, power-limit change or overclock was made.
During CUDA work one observation was P2, 1,935 MHz SM, 6,500 MHz memory,
172.4 W and 56 C, with no active clock-event reason. The reported maxima
were 2,100/7,001 MHz and the existing power limit was 280 W.
A request for the supported maximum memory frequency with `-lmc 7001,7001`
returned **not supported**; its exit code was 0 despite making no change.
Application-clock queries were deprecated. The requested frequency is not
reported as an applied optimization.

The explicit zero-additional-reserve pilot filled VRAM until the engine
reported 3 MiB free. It repeatedly evicted and recaptured verify graphs.
The 31,610-token prefill fell from 2,153.3 to 972.8 tok/s; client TTFT rose
from about 14.7 to 32.560 seconds. Filling all VRAM with expert weights
starves the working allocations needed to run quickly. These are historical pilot
observations with tail skip 7, not final-profile confirmation medians.

## Historical screening before the owner disabled the five features

The historical data in `historical-screening.json` are kept separately from the final all-zero-policy
measurements. They are not counted in the final comparison. The new
`pilot-results.json`, `pilot-configs.json` and `pilot-engine-evidence.json`
contain only the later all-zero-policy screening. Each pilot has one fresh
engine and five cases and is exploratory.

The first `fetch2` probe was inactive: the log explicitly says admission
was disabled because overlap was on. `fetch2-serial` disables overlap and
actually activates admission. `rotate` combines resident-complement mode,
overlap off and exchange rotation; it is not a rotation-only comparison.
It reduced host memory use but was slower on this workload.

The alternate tuned DFlash2 pack was downloaded and hash-verified before
measurements: 339,694,592 bytes, SHA-256
`4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c`.
The installed server derives its draft path from the model pack directory.
A checkout-local alias directory points at the unchanged model pack and
exposes the verified tuned draft under the loader's plain filename. No
stock-hash marker was forged, and the existing installation was unchanged.
The resolved draft path was checked before running both tuned candidates.
Neither tuned setting consistently won; the stock draft remains selected.

## Limits

Two concurrent clients in the cache harness queue through one serving
slot. This is not two simultaneously resident 256K contexts. A 1 GiB
parked cache cannot hold an entire second 256K conversation; long history
reuse depends on the live session and an unchanged token prefix. Editing
the prefix or switching away from an oversized conversation can require
another cold prefill. Cold throughput and cached TTFT are reported separately.

All measurements here use text. The existing vision configuration is kept,
but image inference at the reduced reserve is not benchmarked. Synthetic
retrieval and the small code/JSON/tool smoke suite do not establish broad
model quality. The scripts, logs, downloads and temporary state stay within
this checkout; no files in the existing Strata installation were changed.

## Reproduction and files

`pilot-configs.json` contains credential-free local configurations. Copy
`baseline` and `reserve512-fetch2` into new JSON files under `.turing/` and
compare them with the same installed server and executable:

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_bench.py \
  --config .turing/repro/baseline.json \
  --candidate-config .turing/repro/candidate.json \
  --server /path/to/serve/server.py --out .turing/repro/confirm \
  --cases bench/results/2026-11-11-titan-rtx-dflash2/performance-prompts.json \
  --rounds 3 --max-new 768 --warmup-rounds 3 --warmup-tokens 256
```

The final serving config is produced by `tools/qfn_iq3s_speed_profile.py`.
For the cache comparison, both arms use this final decode configuration;
only the baseline's `--prompt-cache` is set to `0`. The enabled arm keeps
`2`. Use `tools/qfn_cache_bench.py` with `--words 30000 --rounds 3`.
Both arms have the same 1 GiB parked budget, 3 GiB floor and 512 MiB GPU
reserve. The tiny cache warmup differs from the decode suite's three
256-token warmups; only equal-warmup arms are compared to each other.

A separate cache run uses `--words 240000 --continuity-only --prime-parked`.
It first seeds an independent 30K-word chat so the near-limit prompt runs
while a smaller conversation can remain parked. The near-limit code prompt
comes from the previous IQ3_S report's `make-near-prompts.py`. The 18 smoke
inputs are its `iq3s-quality-prompts.json`. `prompt-manifest.json` identifies
the exact strings. These inputs are synthetic; no private client prompts
or answers are included.

`check-answers.py` validates recorded retrieval values, structured responses,
and a whitelist of manually reviewed merge-function ASTs against 100 seeded
inputs each. It does not execute surrounding generated test code.
`summarize.py` recomputes medians and ranges. `monitor.py` records memory and
GPU counters while a command runs, through the isolation wrapper. RAM
availability and swap counters are host-wide; small page-in changes cannot
be attributed to the model from these counters alone. Two-second samples
can miss short peaks. Warmup files also record the engine's resident memory.
