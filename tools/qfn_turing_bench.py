#!/usr/bin/env python3
"""Interleave the existing SM75 series off/on, retaining full responses and timings.

Run through tools/turing-run.sh with a local config whose exe is an absolute path.
This is a text-only microbenchmark, not proof of quality at full 256K context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from ab_engine import PROMPTS, long_prompt
from strata_bench import chat, digest

ROOT = Path(__file__).resolve().parents[1]


def retrieval_prompt(words: int) -> str:
    rng = random.Random(7)
    vocabulary = "river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz meadow anchor violet thunder".split()
    lengths = [words // 3, words // 3, words - 2 * (words // 3)]
    chunks = [" ".join(rng.choice(vocabulary) for _ in range(n)) for n in lengths]
    return (
        "Read the document and remember the project access codes. Ignore the filler words.\n"
        "Project ALPHA has access code 729183.\n" + chunks[0] +
        "\nProject BETA has access code 416927.\n" + chunks[1] +
        "\nProject GAMMA has access code 853204.\n" + chunks[2] +
        "\nReturn only the three access codes, in ALPHA, BETA, GAMMA order, separated by commas."
    )


def local_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"output/config path must be inside {ROOT}")
    return path


def wait_ready(url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}; see server log")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError("server did not become ready")


def stop(proc: subprocess.Popen) -> None:
    # Each server has its own process group, including its engine child.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-config", help="compare two local configs instead of toggling SM75_PERF")
    parser.add_argument("--compare-builds", action="store_true", help="explicitly allow different executables; records both hashes")
    parser.add_argument("--server", default=str(ROOT / "serve/server.py"))
    parser.add_argument("--out", required=True, help="new result directory inside the checkout")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sm75", type=int, choices=(0, 1), nargs="+", default=[0, 1],
                        help="series settings to run; one value isolates a separate config comparison")
    parser.add_argument("--long-words", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--warmup-rounds", type=int, default=1,
                        help="warmup requests before recording each arm (default: 1)")
    parser.add_argument("--warmup-tokens", type=int, default=16,
                        help="warmup cap; values above 16 use a binary-search prompt instead of Say hello")
    parser.add_argument("--only-long", action="store_true")
    parser.add_argument("--retrieval", action="store_true", help="use a three-marker retrieval smoke test for the long case")
    parser.add_argument("--cases", help="local JSON list of [name, prompt] pairs, replacing built-in cases")
    parser.add_argument("--dump-first-logits", action="store_true", help="save first-token FP32 logits separately for each engine run")
    parser.add_argument("--ready-timeout", type=float, default=600)
    args = parser.parse_args()
    if os.environ.get("STRATA_TURING_ISOLATED") != "1":
        parser.error("run this benchmark through bash tools/turing-run.sh")
    if min(args.rounds, args.long_words, args.max_new, args.ready_timeout, args.warmup_rounds, args.warmup_tokens) <= 0:
        parser.error("rounds, long-words, max-new, ready-timeout and warmup settings must be positive")
    config_path = local_path(args.config)
    base = json.loads(config_path.read_text())
    candidate = json.loads(local_path(args.candidate_config).read_text()) if args.candidate_config else None
    exe = Path(base["exe"])
    if not exe.is_absolute() or not exe.is_file():
        parser.error("config exe must be an existing absolute path, not a component selector")
    if base.get("api_key"):
        parser.error("use an isolated local benchmark config without an API key")
    if candidate is not None:
        candidate_exe = Path(candidate.get("exe", ""))
        if candidate.get("api_key") or not candidate_exe.is_absolute() or not candidate_exe.is_file():
            parser.error("candidate must have an existing absolute executable and no API key")
        if candidate.get("exe") != base.get("exe") and not args.compare_builds:
            parser.error("different executables require --compare-builds")
        if args.sm75 != [0, 1]:
            parser.error("--candidate-config cannot be combined with a custom --sm75 list")
    elif args.compare_builds:
        parser.error("--compare-builds requires --candidate-config")
    # Never attach to an existing service or kill it.
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", args.port))
    out = local_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    with exe.open("rb") as binary:
        engine_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    metadata = {
        "engine": str(exe), "server": str(Path(args.server).resolve()),
        "engine_sha256": engine_hash,
        "candidate_engine_sha256": hashlib.sha256(Path(candidate["exe"]).read_bytes()).hexdigest() if candidate else None,
        "server_sha256": hashlib.sha256(Path(args.server).read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": base, "candidate_config": candidate, "arguments": vars(args),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
        "note": "Local raw data; review paths/config/answers before publishing. Prompts are shorter than context capacity.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    cases = [
        ("code", PROMPTS[0]),
        ("korean", "파이썬에서 리스트와 튜플의 차이를 한국어로 설명하고, 각각 적합한 사용 예를 보여 주세요."),
        ("long", retrieval_prompt(args.long_words) if args.retrieval else long_prompt(args.long_words)),
    ]
    if args.only_long:
        cases = cases[-1:]
    if args.cases:
        cases = json.loads(local_path(args.cases).read_text())
        if not cases or any(not isinstance(c, list) or len(c) != 2 or not all(isinstance(v, str) and v for v in c) for c in cases):
            parser.error("cases must be a nonempty list of [nonempty name, nonempty prompt] pairs")
        if len({c[0] for c in cases}) != len(cases):
            parser.error("case names must be unique")
    (out / "prompts.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    url = f"http://127.0.0.1:{args.port}"
    with (out / "results.jsonl").open("w") as results:
        for rnd in range(args.rounds):
            order = args.sm75 if rnd % 2 == 0 else args.sm75[::-1]
            for variant in order:
                arm = ("candidate" if variant else "baseline") if candidate is not None else f"sm75-{variant}"
                label = f"r{rnd}-{arm}"
                cfg = dict(candidate if candidate is not None and variant else base)
                enabled = int(cfg.get("env", {}).get("STRATA_SM75_PERF", "0")) if candidate is not None else variant
                cfg.update(cwd=str(ROOT), port=args.port, host="127.0.0.1", open_browser=False,
                           log=str(out / f"{label}-engine.log"))
                cfg["env"] = dict(cfg.get("env", {}), STRATA_SM75_PERF=str(enabled), STRATA_NO_PRESET="1")
                if args.dump_first_logits:
                    cfg["env"]["STRATA_DUMP_FIRST_LOGITS"] = str(out / f"{label}-first")
                path = out / f"{label}.json"
                path.write_text(json.dumps(cfg, indent=2))
                with (out / f"{label}-server.log").open("w") as log:
                    proc = subprocess.Popen(
                        [sys.executable, str(Path(args.server).resolve()), "--engine", "strata",
                         "--config", str(path), "--port", str(args.port)],
                        cwd=ROOT, env=dict(os.environ, STRATA_NO_PRESET="1"),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    try:
                        wait_ready(url, proc, args.ready_timeout)
                        warmups = []
                        for _ in range(args.warmup_rounds):
                            prompt = ("Implement binary search in Python, explain its invariants and time complexity, "
                                      "and include examples for empty input and a missing value.") if args.warmup_tokens > 16 else "Say hello."
                            warm = chat(url, "", [{"role": "user", "content": prompt}], args.warmup_tokens, timeout=600)
                            warmups.append(digest(warm))
                            if not warmups[-1]["ok"]:
                                raise RuntimeError(f"{label}: warmup failed: {warm['err']}")
                        memory = []
                        # Under turing-run's PID namespace these are this benchmark's engine children only.
                        for entry in Path("/proc").iterdir():
                            if not entry.name.isdecimal():
                                continue
                            try:
                                if (entry / "comm").read_text().strip() != "strata-host":
                                    continue
                                values = {}
                                for line in (entry / "smaps_rollup").read_text().splitlines():
                                    key, _, value = line.partition(":")
                                    if key in ("Rss", "Pss", "Anonymous", "AnonHugePages", "Locked"):
                                        values[key + "_kib"] = int(value.split()[0])
                                memory.append(values)
                            except (OSError, ValueError):
                                pass
                        (out / f"{label}-warmup.json").write_text(json.dumps({"requests": warmups, "engine_memory": memory}, indent=2))
                        print(f"{label}: ready", flush=True)
                        for name, prompt in cases:
                            reply = chat(url, "", [{"role": "user", "content": prompt}], args.max_new, timeout=1200)
                            metrics = digest(reply)
                            row = {"round": rnd, "arm": arm, "sm75_perf": enabled, "case": name,
                                   "metrics": metrics, "response": reply}
                            if args.retrieval and name == "long":
                                row["retrieval_exact"] = "".join(reply["text"].split()) == "729183,416927,853204"
                            results.write(json.dumps(row, ensure_ascii=False) + "\n")
                            results.flush()
                            print(json.dumps({"run": label, "case": name, **metrics}), flush=True)
                            if not metrics["ok"]:
                                raise RuntimeError(f"{label}/{name}: request failed")
                    finally:
                        stop(proc)
                time.sleep(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
