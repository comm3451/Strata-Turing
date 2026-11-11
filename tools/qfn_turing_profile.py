#!/usr/bin/env python3
"""Copy a QFN launch config into this checkout with a measured node profile.

The source and model files are read-only inputs. Use tools/turing-run.sh to serve
the resulting config. Native IQ requires this fork's compiled CUDA engine.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def set_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    out = list(args)
    while flag in out:
        i = out.index(flag)
        if i + 1 == len(out) or out[i + 1].startswith("--"):
            raise ValueError(f"missing value for {flag}")
        del out[i:i + 2]
    return out if value is None else out + [flag, value]


def make_config(source: dict, exe: Path, log: Path, fp16_prefill: bool = False,
                native_iq_prefill: bool = False, turing_attention: bool = False,
                titan_throughput: bool = False, titan_fused: bool = False,
                titan_iq3s: bool = False) -> dict:
    if titan_iq3s and titan_fused:
        raise ValueError("choose either the IQ3_XXS fused profile or the IQ3_S profile")
    if titan_fused or titan_iq3s:
        titan_throughput = True
    if titan_throughput:
        fp16_prefill = native_iq_prefill = turing_attention = True
    cfg = copy.deepcopy(source)
    args = cfg["args"]
    native = args[args.index("--native") + 1] if "--native" in args else ""
    quant = "iq3_s" if titan_iq3s else "iq3_xxs"
    if "flash-next" not in native.lower() or quant not in native.lower() or "--mtp" not in args:
        raise ValueError(f"this profile requires Qwen Flash-Next {quant.upper()} with an existing MTP pack")
    settings = {
        "--max-context": "262144", "--kv": "int8", "--kv-resident": "32768",
        "--spec": "4", "--spec-min-p": "0.5", "--pcie-frac": None,
        "--pool-workers": "13", "--vram-reserve-mib": "2048",
        "--expert-cache": "auto", "--prefill": "auto:32768" if titan_throughput else "auto",
        "--adapt-every": None, "--adapt-swaps": None, "--adapt-decay": None,
    }
    if titan_iq3s:
        settings.update({"--conversation-cache-mib": "1024", "--conversation-cache-min-free-mib": "4096"})
    for flag, value in settings.items():
        args = set_arg(args, flag, value)
    cfg.update(args=args, exe=str(exe), cwd=str(ROOT), log=str(log),
               drafting="mtp", parallel=1, open_browser=False)
    cfg["env"] = {**cfg.get("env", {}), "STRATA_ROUTE_TAIL_SKIP": "0",
                  "STRATA_ROUTE_PRIOR": "0", "STRATA_KVA": "0",
                  "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_HC_Q8": "0",
                  "STRATA_SM75_PERF": "1", "STRATA_EXP_MODE": "8",
                  "STRATA_NO_PRESET": "1", "STRATA_BF16_TC": "1" if fp16_prefill else "0",
                  "STRATA_PF_FUSED_NATIVE_SM75": "1" if native_iq_prefill else "0",
                  "STRATA_SELECT_SIMT": "2" if turing_attention else "0",
                  "STRATA_PA_V2T": "1" if turing_attention else "0"}
    if native_iq_prefill:
        cfg["env"].update(STRATA_PF_FUSED="1", STRATA_PF_FUSED_SM75="1")
    if titan_throughput:
        cfg["env"].update(
            STRATA_CPU_IQ_ALL_EXACT1="1", STRATA_SIDE_MIXER="2",
            STRATA_B6_MMVQ_ROWS="1", STRATA_ATTN_DAILY="1", STRATA_QSA_ROWS="1",
            STRATA_ADAPT_OVERLAP="1", STRATA_ADAPT_NOWAIT="0", STRATA_ADAPT_LAG="1",
            STRATA_SPEC_COST="1",
        )
    if titan_fused:
        cfg["env"].update(STRATA_SM75_IQ2XS_FUSED="1", STRATA_Q2_BITPLANE="1",
                          STRATA_PF_HC_REGION="1")
    if titan_iq3s:
        # Full-arena starts stalled on this 64 GB node; capping CUDA registration allowed startup.
        # Keep the complete model in RAM but register at most 40 GiB, hot misses first.
        cfg["env"].pop("STRATA_ARENA_THP", None)
        cfg["env"].update(STRATA_SM75_IQ2XS_FUSED="0", STRATA_Q2_BITPLANE="1",
                          STRATA_PF_HC_REGION="1", STRATA_NO_ARENA_THP="1",
                          STRATA_ARENA_PIN_GIB="40", STRATA_PIN_BY_NEED="1")
    cfg.setdefault("host", "127.0.0.1")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="existing launch JSON; read only")
    parser.add_argument("--exe", required=True, help="absolute path to the measured engine executable")
    parser.add_argument("--out", required=True, help="new JSON file inside this checkout")
    parser.add_argument("--fp16-prefill", action="store_true",
                        help="opt in to FP16 tensor-core projections; rounding and replies can differ")
    parser.add_argument("--native-iq-prefill", action="store_true",
                        help="opt in to the fork's native IQ Turing kernels; requires the new build and can change rounding")
    parser.add_argument("--turing-attention", action="store_true",
                        help="opt in to the FP32 tiled QSA scorer and Turing v2 prompt attention; rounding can differ")
    parser.add_argument("--titan-throughput", action="store_true",
                        help="use the measured TITAN RTX/64 GB pipeline and 32K prefill profile; includes all three fast switches")
    parser.add_argument("--titan-fused", action="store_true",
                        help="include titan-throughput plus IQ2_XS fused decode, CPU Q2 bit-planes and shared prefill scratch")
    parser.add_argument("--titan-iq3s", action="store_true",
                        help="QFN IQ3_S on this 64 GB node: 40 GiB hot-expert pinning, no forced THP, 1 GiB parked cache")
    args = parser.parse_args()
    if os.environ.get("STRATA_TURING_ISOLATED") != "1":
        parser.error("run through bash tools/turing-run.sh")
    if args.titan_iq3s and args.titan_fused:
        parser.error("--titan-iq3s and --titan-fused select different model quantizations")
    out = Path(args.out).resolve()
    exe = Path(args.exe)
    if not out.is_relative_to(ROOT) or not exe.is_absolute() or not exe.is_file():
        parser.error("out must be inside the checkout and exe must be an existing absolute path")
    cfg = make_config(json.loads(Path(args.source).read_text()), exe, out.with_suffix(".engine.log"),
                      args.fp16_prefill, args.native_iq_prefill, args.turing_attention, args.titan_throughput,
                      args.titan_fused, args.titan_iq3s)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Preserve an existing API key without printing it or creating a world-readable copy.
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(out)


if __name__ == "__main__":
    main()
