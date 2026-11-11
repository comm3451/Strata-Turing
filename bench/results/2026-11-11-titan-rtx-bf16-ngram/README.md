# BF16 N-grams with QFN IQ3_S on TITAN RTX

Measured on 2026-11-11 on one TITAN RTX 24 GiB, Xeon E5-2680 v4
(14 cores / 28 threads, AVX2) and 64 GB DDR4. The model stays
ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S, with stock DFlash2
window 6/confidence 0.5 and 262,144 total context tokens. Only the
N-gram table changes from IQ4_NL to BF16.

## File and compatibility

The BF16 table is
[`shared/ngrams-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/apetersson/Qwen3.8-Flash-Next-GGUF/blob/9af845f821fec6bf2a9ddbacd3778930ba80157f/shared/ngrams-Qwen3.8-Flash-Next-BF16.gguf),
from revision `9af845f821fec6bf2a9ddbacd3778930ba80157f` of
`apetersson/Qwen3.8-Flash-Next-GGUF`. It is 102,400,491,712 bytes;
the downloader verified SHA-256
`11828e522266cf7aa9579dea8a0ab375be7afcadede4bd9b9fcfb0722d952ca1`
before exposing the final file. Downloading and hashing were finished
before these measurements. The original 28,800,138,432-byte IQ4_NL table
and main IQ3_S model are identified in the
[IQ3_S report](../2026-11-11-titan-rtx-iq3s/README.md).

The actual file header has one tensor, `per_layer_token_embd.weight`,
dimensions `[160, 320001536]`, GGML type 30 (BF16). Data starts at byte 192;
each row is 320 bytes, and the tensor exactly fills the remaining file.
The retained split metadata says shard 1 of 2 with 1,224 tensors, but there
are no projection tensors in this file. The fork's existing PLE-only GGUF
recognition handles this without adding it to the native projection shards.
`gguf-header.json` records the header read from the downloaded file.

The engine's direct reader fetches needed rows from SSD. It does not load
the entire 102 GB table into RAM or VRAM. Its existing BF16 row decoder
widens the stored values to FP32; the main model remains IQ3_S. The default
row cache holds 1,048,576 rows in both arms: its data payload increases
from 90 MiB to 320 MiB, a 230 MiB increase. The equal row count is preserved;
this is not a comparison with an equal byte-sized N-gram cache.

## Comparison method

Both arms use the same sm_75 CUDA executable, SHA-256
`1b50bc01e1f3b6957b6fe8a58b577096bb5546278899691e3ab6f2578cfc4c49`,
built with CUDA 13.0.88/GCC 14.4, and installed Python server 0.1.57.
Source/report base is `a350f987caec6be392bea40a610f82110c1d81c4`.
No engine code or precision switches change in this comparison. The
previously added PLE-only loader handling is exercised with the full BF16
file here. Each metadata file records executable/server/harness hashes.

The confirmation runs three alternating pairs, A/B, B/A, A/B. A uses
IQ4_NL N-grams; B uses BF16. Each arm starts a fresh engine and performs
three 256-token warmups. The five measured cases use greedy sampling,
thinking disabled and a 768-token output cap. All inputs match the
[DFlash2 comparison prompts](../2026-11-11-titan-rtx-dflash2/performance-prompts.json).

Both retain INT8 KV with 32,768 resident cells, 13 workers, 2,048 MiB VRAM
reserve, automatic expert-cache/PCIe sharing, 32K prompt chunks and the
IQ3_S pin-by-need profile capped at 40 GiB, without forced THP. DFlash2
alone runs at window 6/confidence 0.5 in both arms; no MTP layer is loaded.
The production 1,024 MiB conversation-cache setting and 4,096 MiB free-RAM
floor are preserved. `--prompt-cache 0` disables conversation caching, as
the logs state; these tests do not exercise retained conversation states.

The long performance inputs are synthetic word lists. Their token
distribution does not represent every natural-language document or code
repository. Different N-gram precision can change outputs and their lengths;
completion-time changes are not pure engine-speed ratios. This is not a
reproduction of a broad C++ accuracy score or a reasoning-enabled benchmark.

## Repeated results

All 30 confirmation requests succeeded and report zero reused tokens.
There is no uniform speed improvement from changing the N-gram format.

| Metric, median of three | IQ4_NL | BF16 | Change |
|---|---:|---:|---:|
| Short code decode | 96.5 tok/s | 92.6 tok/s | -4.0% |
| Korean decode, capped | 74.2 tok/s | 76.2 tok/s | +2.7% |
| 4,349-token word-list summary decode | 46.5 tok/s | 45.4 tok/s | -2.4% |
| 31,610-token word-list summary decode | 63.7 tok/s | 67.0 tok/s | +5.2% |
| Code decode after 4,393 input tokens | 74.2 tok/s | 77.1 tok/s | +3.9% |
| 4,349-token prefill | 992.3 tok/s | 991.0 tok/s | -0.1% |
| 31,610-token prefill | 2,169.7 tok/s | 2,156.6 tok/s | -0.6% |

Short-code decode ranges are 96.2–96.5 versus 92.4–92.8 tok/s; all three
pairs show the small BF16 regression. Korean and 4K-input code favor BF16
in all three pairs. The 32K-summary decode differences include different
answers and one pair with only a 0.4 tok/s difference.

Short-code completion time increases from 4.779 to 5.702 seconds, while
the median answer grows from 432 to 501 tokens. The 4K-input code is
faster per token with BF16 but its median answer grows from 321 to 357
tokens, so total time increases from 8.581 to 8.866 seconds. These examples
do not reproduce a claim that BF16 always produces shorter or faster answers.
Client TTFT is 0.320 versus 0.313 seconds for short code and 14.631 versus
14.720 seconds for the 31,610-token prompt.

The initial single BF16 probe is retained separately: short code 92.9 tok/s,
Korean 76.4 tok/s and 31,610-token prefill 2,158.8 tok/s. It is not counted
as a fourth confirmation round.

During confirmation, 2-second samples show peak VRAM 22,118 MiB and
minimum available host RAM 4.65 GiB. Swap occupancy stays between 32,044
and 32,548 KiB. These are occupancy samples, not page-in/out counters.
The same configured model/VRAM limits are preserved; the larger N-gram
row-cache payload is part of the BF16 format's cost.
Warmed engine RSS is 52.698–52.700 GiB with IQ4_NL and 52.923–52.924 GiB
with BF16. Every arm has 7,911 GPU expert-cache slots. Apart from output-log
paths, the recorded configs differ at only the `--ple-gguf` value.

## Actual near-256K inputs

One separate BF16 run tests the production-budget configuration with two
large inputs. This is a BF16 capacity/content check, not another paired
IQ4_NL/BF16 comparison. Both requests report zero reused tokens.

| Observation | BF16 result |
|---|---:|
| Retrieval input | 252,146 tokens |
| Retrieval prefill | 2,043.0 tok/s |
| Retrieval client TTFT | 123.718 s |
| Three recovered values | all correct |
| Code input | 252,100 tokens |
| Code prefill | 2,028.5 tok/s |
| Code client TTFT | 124.539 s |
| Code decode | 94.7 tok/s |
| Complete code answer | 594 tokens |
| Code total time including prefill | 130.785 s |

The recorded merge function passes 100 seeded input cases. The 23-token
retrieval answer is too short to measure steady decode throughput. Peak
sampled VRAM is 22,118 MiB and minimum available host RAM is 4.54 GiB.
These two synthetic prompts establish neither broad long-context accuracy
nor capacity for two full 256K sessions.

Recreate these inputs with
[`make-near-prompts.py`](../2026-11-11-titan-rtx-iq3s/make-near-prompts.py)
and use `--cases <generated JSON> --retrieval --rounds 1 --sm75 1`
with the BF16 config. The prompt hashes are in `near-prompts-manifest.json`.

## Answer checks

The separate BF16 smoke run completes all 18 requests: code, Korean,
arithmetic, JSON, logic and tool-shaped JSON at roughly 0.6K, 5K and 20K
input tokens. It uses the same
[quality prompts](../2026-11-11-titan-rtx-iq3s/iq3s-quality-prompts.json),
greedy sampling, thinking disabled and a 384-token output cap. All answers
finish below the cap and all requests report zero reused tokens.

`check-answers.py` checks 33 recorded answers across the pilot, confirmation,
near-context and quality runs. It isolates 17 merge functions, accepts only
the reviewed reference AST or its reviewed tail-copy variant, and tests each
with 100 seeded inputs while checking that inputs are unchanged. Retrieval
and JSON answers match their expected values. Two Korean paraphrases were
manually reviewed and then added as exact accepted strings. Arithmetic
checks the correct final value, 313; these answers include working despite
the prompt requesting only the number, so they do not pass strict formatting.
The tool-shaped JSON checks do not exercise native API tool calling.

This is a small content smoke check, not evidence of improved overall
reasoning or coding accuracy over IQ4_NL. Quality-run samples show peak
VRAM 22,132 MiB and minimum available RAM 4.56 GiB.

## Serving after activation

The checkout-local server was restored on port 8080 with IQ3_S, BF16
N-grams, DFlash2 and 262,144 context tokens. The API model name, key,
address and single serving slot stay unchanged. The previous private
IQ4_NL configuration is retained for rollback inside the checkout. No
system service or files in the existing installation were changed.

The startup excerpt confirms that the BF16 table and DFlash2 alone loaded.
After loading, an unauthenticated completion request returned HTTP 401.
Two authenticated clients submitted tiny arithmetic requests together;
both answers were correct, through the one serving slot. `restored-health.json`
records this check. It is neither a throughput benchmark nor evidence that
two full 256K sessions fit in memory. This checkout-local launch does not
change the system service's boot-time model configuration.

## Reproduce and inspect

Start with the measured IQ3_S/DFlash2 config and change only the value of
`--ple-gguf` to the BF16 file. Keep the main model, native pack, tokenizer,
model API name, drafter and context unchanged. The published configs are
sanitized; replace `<strata>` and `<home>` with local paths before use.
Check the startup log's `PLE on, table ... BF16.gguf` and `DFlash2 drafter on`
lines to confirm which table and drafter actually loaded.

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_bench.py \
  --config .turing/bench/iq4nl.json --candidate-config .turing/bench/bf16.json \
  --server /path/to/control/0.1.57/serve/server.py \
  --out .turing/bench/ngram-confirm --rounds 3 \
  --cases bench/results/2026-11-11-titan-rtx-bf16-ngram/performance-prompts.json \
  --warmup-rounds 3 --warmup-tokens 256 --max-new 768
```

Per-request JSON retains the responses and engine/client timings. The
summary retains individual values and medians; logs and 2-second memory
samples retain the actual runtime observations. Prompt manifests hash
each exact input. Original model files outside this checkout are unchanged.
