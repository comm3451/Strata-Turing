# IQ3_XXS/BF16 conversation-cache validation

Measured on 2026-11-12 on TITAN RTX 24 GiB, Xeon E5-2680 v4 and 64 GB DDR4.
The existing IQ3_XXS model, BF16 N-gram table, stock DFlash2 window 6 /
confidence 0.5, INT8 KV and 262,144 context remain unchanged. All five owner
flags remain zero: ROUTE_PRIOR, ROUTE_TAIL_SKIP, KVA, PREFILL_CPU_SHARE and
HC_Q8 (each with the STRATA_ prefix). GPU power limit was 220 W throughout
the sampled validation. No model or power-limit comparison was run.

## Configuration

`tools/qfn_iq3xxs_cache_profile.py` changes only these engine options:

| Option | Previous | Applied |
|---|---:|---:|
| `--conversation-cache-mib` | 1024 | 8192 |
| `--conversation-cache-slots` | 4, implicit default | 8 |
| `--prompt-cache` | 2 | 8 |
| `--conversation-cache-min-free-mib` | 3072 | 3072 |

The node had about 11.3 GiB available before the change. Eight GiB is the
selected upper budget for this node, not the engine's maximum accepted
number or an allocation at startup. Parking checks available RAM, keeps
the 3 GiB floor, and evicts older conversations if necessary. The number of
full-context conversations retained depends on their snapshot sizes and
other memory use; eight slots do not guarantee eight 256K conversations.
Minimum parked length and replacement gain remain 4,096 tokens each.

The private configuration was backed up before replacement. Following the
owner's service restart, the actual process arguments, environment and API
reported the requested configuration. No system unit, model, driver or
other checkout was changed.

## What the large prompt contains

The document is `retrieval_prompt(240000)` from `tools/qfn_turing_bench.py`:
240,000 words selected deterministically with seed 7 from a twenty-word
vocabulary, with three invented project codes interspersed. It asks for
only the codes in ALPHA, BETA, GAMMA order. The correct answer is:

```text
729183, 416927, 853204
```

The resulting chat input contains 252,146 tokens. Its answer contains only
23 generated tokens by the engine's count. This deliberately tests retaining
and retrieving a large context with a small, checkable answer. It is not a
reasoning evaluation or a sustained decode-speed benchmark.

Four requests were made through the existing authenticated service, using
greedy sampling, thinking disabled and a 64-token output cap:

1. Submit the document as one user message.
2. Switch to the unrelated question `Return only the number: 19 + 23 = ?`.
3. Submit the original document again, without an assistant message.
4. Submit the document, its first assistant answer, then
   `Return only the access code for BETA.`.

The service was also receiving user requests. This is a live functional
validation, not an isolated latency comparison. User conversation contents
are excluded. Cache counters and memory readings describe the shared service.

## Results

| Request | Input tokens | Reused | Newly read | Output tokens | Engine input phase | Client TTFT |
|---|---:|---:|---:|---:|---:|---:|
| First document | 252146 | 0 | 252146 | 23 | 127.655 s | 162.331 s |
| Unrelated question | 26 | 0 | 26 | 3 | 0.681 s | 38.523 s |
| Return to document | 252146 | 252139 | 7 | 23 | 1.021 s | 33.839 s |
| Follow-up question | 252191 | 252169 | 22 | 7 | 1.251 s | 8.939 s |

All four answers matched the expected numbers. Client TTFT includes waiting
behind other requests; it must not be represented by the shorter engine
input-phase duration. The engine reported a 4,651,126,144-byte (4.33 GiB)
snapshot for the first long conversation and parked it in 2,747.2 ms. A
1 GiB budget cannot hold that snapshot. The later return and follow-up
confirmed reuse of more than 252,000 tokens.

Two-second memory samples reached a minimum of 4,386.61 MiB (4.28 GiB)
available RAM. Host-wide swap counters increased by 24 page-ins and zero
page-outs. These samples can miss brief peaks. They do not prove that every
future mix of conversations fits. Raw synthetic responses and safe numeric
results are in `checks.json`, with memory samples in `memory.jsonl` and the
identity/settings summary in `validation.json`.

The four-request verification process exited successfully. No further test
requests are scheduled. Its parked snapshot remains subject to ordinary
cache eviction; this validation did not flush other users' conversation
state. Decode rates from these 3–23-token answers are not evidence for a
speed gain. Broad model quality, eight full-context conversations and a
controlled old/new cache latency comparison were not tested.
