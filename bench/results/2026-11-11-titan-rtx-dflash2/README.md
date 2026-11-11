# Stock DFlash2 versus MTP on TITAN RTX

Measured on 2026-11-11 on one TITAN RTX 24 GiB, Xeon E5-2680 v4
(14 cores / 28 threads, AVX2) and 64 GB DDR4. With QFN IQ3_S, stock
DFlash2 improves the measured code-generation workloads. It does not speed
up every task: Korean decode is effectively tied, and the 4K word-list
summary is slower. Prefill is effectively unchanged.

## Model and drafter

Both arms use the same ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S
weights and **IQ4_NL N-gram table**, whose pinned revisions and hashes are
in the [IQ3_S report](../2026-11-11-titan-rtx-iq3s/README.md).
These are not BF16 N-gram results. The BF16 download and its SHA-256
verification finished before inference started; no download or file hash
scan ran during these measurements.

The stock drafter is derived from
[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, Apache-2.0.
Strata's converted Q4_0 pack with Q4_0 selector codebooks is 339,694,592
bytes, SHA-256
`4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd`.
The existing file was hashed and reused read-only. Both drafters use the
same 106,299-token draft vocabulary: SHA-256
`b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`.
This is **not Strata's separately tuned drafter pack**.

All runs use the same CUDA 13.0.88/GCC 14.4 sm_75 binary, SHA-256
`1b50bc01e1f3b6957b6fe8a58b577096bb5546278899691e3ab6f2578cfc4c49`,
and installed Python server 0.1.57. No engine code was changed for this
comparison. Source/report base is `861fca99e7c502eba3f4af7e7be46bd1f7e6280d`.
Each metadata file records the server, executable and harness hashes.

## Repeated comparison

Three pairs run A/B, B/A, A/B, with a fresh engine and three 256-token
warmups per arm. A is MTP, `--spec 4 --spec-min-p 0.5`. B is DFlash2,
`--spec 6 --draft-conf 0.5 --draft-fallback none`, with no MTP layer loaded.
The server config uses `drafting: dflash2` and explicit `user_args` for the
window/confidence so the server does not replace them with its automatic
choice. The engine logs confirm the active drafter in every arm.

Both retain 262,144 total context, INT8 KV with 32,768 resident cells,
13 workers, 2,048 MiB VRAM reserve, automatic expert-cache/PCIe sharing,
32K prompt chunks and the measured IQ3_S pin-by-need profile capped at
40 GiB, without forced THP. Prompt and parked conversation caches are zero
for this comparison. All 30 measured requests report zero reused tokens.
Sampling is greedy, thinking disabled, with a 768-token output cap.

| Metric, median of three | MTP | DFlash2 | Change |
|---|---:|---:|---:|
| Short code decode | 79.9 tok/s | 96.4 tok/s | +20.7% |
| Korean decode, capped | 74.2 tok/s | 74.3 tok/s | +0.1% |
| 4,349-token word-list summary decode | 48.3 tok/s | 45.1 tok/s | -6.6% |
| 31,610-token word-list summary decode | 64.4 tok/s | 65.2 tok/s | +1.2% |
| Code decode after 4,393 input tokens | 64.2 tok/s | 74.0 tok/s | +15.3% |
| 4,349-token prefill | 993.0 tok/s | 989.8 tok/s | -0.3% |
| 31,610-token prefill | 2,154.1 tok/s | 2,163.8 tok/s | +0.5% |

All three paired code comparisons favor DFlash2, at both input lengths.
Short-code decode ranges are 78.7–80.2 versus 94.7–96.5 tok/s. Short-code
completion time falls from 6.646 to 4.782 seconds, but the median answer
also shrinks from 501 to 432 tokens. That latency gain includes a shorter
answer. After 4,393 input tokens, both median answers are 321 tokens;
completion time is 9.217 versus 8.606 seconds, including prefill.

Short-code client TTFT is 0.310 versus 0.321 seconds; Korean TTFT is
0.373 versus 0.419 seconds. The 31,610-token TTFT is 14.745 versus
14.672 seconds. DFlash2 is not an across-the-board first-token improvement.
Outputs are not byte-identical between arms. Different draft windows and
expert placement can change numerical paths; these measurements do not
establish universal quality equivalence.

The initial single-start probes are exploratory, not extra confirmation
rounds. Server 0.1.57 chose DFlash2 window 6/confidence 0.8 automatically.
That probe measured short code/Korean at 82.9/72.0 tok/s. Confidence 0.5
measured 96.8/74.2 tok/s and was selected for the independent comparison.
Windows 4/8, confidence 0.35, and the separately tuned pack were not swept.

## Near the 256K limit

A separate **single pair** restores the production parked-cache budget of
1,024 MiB with a 4,096 MiB free-RAM floor in both arms. It uses the same
warmup, greedy sampling and zero prompt reuse. Each arm reads a
252,146-token retrieval document and then a 252,100-token code prompt.
This tests actual long input, not just a configured capacity.
Both retain `--prompt-cache 0`, so the engine reports that conversation
caching is disabled. The 1,024 MiB setting matches the production config,
but this run does not exercise retained conversation states or cache reuse.

| Metric, one observation per arm | MTP | DFlash2 |
|---|---:|---:|
| 252,146-token retrieval prefill | 2,043.6 tok/s | 2,038.8 tok/s |
| Retrieval client TTFT | 123.682 s | 123.972 s |
| Three recovered values | all correct | all correct |
| 252,100-token code prefill | 2,025.4 tok/s | 2,026.6 tok/s |
| Code client TTFT | 124.734 s | 124.658 s |
| Code decode | 81.9 tok/s | 94.5 tok/s |
| Code answer length | 587 tokens | 589 tokens |
| Code total time including prefill | 131.874 s | 130.869 s |

Long-input code decode is 15.4% faster in this pair, with almost the same
answer length. Both recorded merge functions pass 100 seeded input cases.
Prefill dominates total time, so the complete request is only about one
second faster. The 23-token retrieval answers are too short to establish
steady decode throughput. More pairs and real long documents are needed
before treating this as a general long-context speed/quality result.

Recreate these inputs using
[`make-near-prompts.py`](../2026-11-11-titan-rtx-iq3s/make-near-prompts.py)
and compare with `--cases <generated JSON> --retrieval --rounds 1`.
The per-prompt hashes are in `near-prompts-manifest.json`.

## Memory

The same VRAM budget yields 7,780 expert-cache slots with MTP and 7,911
with DFlash2. This comparison includes that practical allocation difference;
it does not hold the slot count fixed to isolate draft arithmetic. The logs
report an 837 MiB MTP layer plus a 212.9 MiB head, versus a 625 MiB DFlash2
drafter including its head. Actual total GPU usage is measured separately.

During confirmation, 2-second samples show peak VRAM 22,198 MiB and
minimum available host RAM 4.75 GiB. First-arm warmed engine RSS is about
52.94 GiB for MTP and 52.70 GiB for DFlash2. Forced THP is absent in both.
Swap use during confirmation stays between 282,444 and 283,844 KiB; these
are occupancy samples, not page-in/out counters. The first exploratory
startup reached higher swap occupancy (about 1.75 GiB), retained in its
raw samples and excluded from the confirmation figures.
The near-limit pair peaks at 22,178 MiB VRAM and has at least 4.65 GiB
available host RAM in the 2-second samples.

## Answer checks and limits

The DFlash2 production-budget configuration passes 18 basic content checks:
code, Korean explanation, arithmetic, JSON, logic and a tool-shaped JSON
answer, each at roughly 0.6K, 5K and 20K input tokens. Inputs match
[`iq3s-quality-prompts.json`](../2026-11-11-titan-rtx-iq3s/iq3s-quality-prompts.json)
from the earlier MTP report; this is not a new paired accuracy benchmark.
The math check validates the final numeric answer, not number-only format.
One correct Korean answer initially failed a marker-word check because it
used “modifiable/unmodifiable”; its exact wording was inspected and added
to the accepted answers. The failed marker check was not a model-content
failure. This limited check does not establish broad coding accuracy.

`check-answers.py` validates 36 recorded answers across the probes,
confirmation, near-limit pair and quality run. It executes only the
reviewed merge-function AST, with 100 seeded inputs per function; surrounding
generated programs/tests are not executed. All 19 recorded merge functions
pass. Capped Korean/word-summary performance responses are not counted as
quality checks. Run the checker through `tools/turing-run.sh`.

After all benchmark runs completed, a production restart did not leave a
ready-server log, and the host was observed to have rebooted at 19:33:40 KST.
The old systemd IQ3_XXS service restarted on boot. Available previous-boot
kernel logs contain no matching OOM, kernel-panic or NVIDIA Xid entry; this
does not establish the reboot's cause. The measurements above were already
complete, but they do not establish unattended or reboot-time reliability.
No system service files were changed, and production restoration is a
separate operation from these measurements.

After the owner stopped the old service, IQ3_S/DFlash2 was restored on the
original address and port with the existing API key and 262,144 capacity.
The startup log confirms DFlash2 alone at confidence 0.5. Two tiny
simultaneous requests returned the correct answers through one actual slot;
an unauthenticated request returned 401. `restored-health.json` records this
smoke check. It is neither a concurrency throughput comparison nor evidence
that two full 256K sessions fit. The running process is checkout-local;
the systemd service configuration remains unchanged.

## Reproduce and inspect

Use the same installed server version and explicit drafter choice. The
sanitized JSON configs/metadata require replacing `<strata>` and `<home>`
with local paths; do not substitute another model, drafter or server and
call the result the same comparison. The stock drafter belongs under
`<data>/dflash2`, next to `<data>/packs`; both draft vocabularies must match.
Use the log to confirm `DFlash2 drafter on` and `no MTP layer is loaded`.

```sh
bash tools/turing-run.sh /path/to/python tools/qfn_turing_bench.py \
  --config .turing/bench/mtp.json --candidate-config .turing/bench/df2.json \
  --server /path/to/control/0.1.57/serve/server.py \
  --out .turing/bench/drafter-confirm --rounds 3 \
  --cases bench/results/2026-11-11-titan-rtx-dflash2/performance-prompts.json \
  --warmup-rounds 3 --warmup-tokens 256 --max-new 768
```

`summary.json` preserves every per-arm value and median. The result JSONs
retain responses and engine/client timings; logs retain drafter selection
and memory allocation. Prompt manifests hash each exact input. These are
focused greedy workload measurements, not the upstream standard benchmark,
a broad coding score, or the hardware's maximum possible speed.
