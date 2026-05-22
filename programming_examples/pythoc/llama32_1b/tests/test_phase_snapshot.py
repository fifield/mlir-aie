# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Per-phase metrics harness for the llama32_1b PythoC port.
#
# Each "phase" of the AIR -> PythoC migration swaps one or more kernels.
# This harness captures a deterministic JSON snapshot of correctness and
# performance numbers, then lets you diff snapshots phase-over-phase.
#
# Usage:
#   pytest tests/test_phase_snapshot.py -v -s   # capture + assert no regression
#   python tests/test_phase_snapshot.py         # capture only, print diff
#   python tests/test_phase_snapshot.py --compare A.json B.json
#
# The harness shells out to llama32_1b_inference.py with synthetic weights
# (no HF download required) and parses its stdout. It does NOT import the
# orchestration code -- intentional, to keep the snapshot stable across
# refactors of the inference module.

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
BUILD_DIR = PROJECT_DIR / "build_peano"
SNAPSHOT_DIR = HERE / "snapshots"
INFERENCE_SCRIPT = PROJECT_DIR / "llama32_1b_inference.py"

# Defaults tuned for "fast snapshot" -- ~30s end-to-end with cached kernels.
DEFAULT_N_TOKENS = 10
DEFAULT_PROMPT = "What is the capital of France?"

# Regression thresholds for compare_snapshots().
DECODE_P95_REGRESSION_PCT = 15.0
PER_KERNEL_REGRESSION_PCT = 15.0


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Examples of lines we parse:
#   "  Compiled rms_gemv_rope: 2.0s -> rms_gemv_rope.elf"
#   "  Token 1: id=14498, time=328ms"
#   "  Layer 15 K_cache: [WARN] corr=0.877268, max_err=2.6236, mean_err=0.3708"
#   "  Layer  0 V_cache: [OK]   corr=0.998812, max_err=0.4101, mean_err=0.0290"
#   "  Logits (pos 2047): corr=0.845401, max_err=2.1670, mean_err=0.4013"
#   "  NPU top-1: 71506 (<synth:[71506]>)"
#   "  CPU top-1: 9226  (<synth:[9226]>)"
#   "  Match: NO"
RE_COMPILED   = re.compile(r"^\s*Compiled\s+(\S+):\s+([\d.]+)s\s*->\s*\S+")
RE_TOKEN      = re.compile(r"^\s*Token\s+(\d+):\s*id=(\d+),\s*time=(\d+)ms")
RE_KCACHE     = re.compile(r"^\s*Layer\s+(\d+)\s+K_cache:\s*\[(\w+)\]\s*corr=([-\d.eE]+),\s*max_err=([-\d.eE]+),\s*mean_err=([-\d.eE]+)")
RE_VCACHE     = re.compile(r"^\s*Layer\s+(\d+)\s+V_cache:\s*\[(\w+)\]\s*corr=([-\d.eE]+),\s*max_err=([-\d.eE]+),\s*mean_err=([-\d.eE]+)")
RE_LOGITS     = re.compile(r"^\s*Logits\s*\(pos\s+(\d+)\):\s*corr=([-\d.eE]+),\s*max_err=([-\d.eE]+),\s*mean_err=([-\d.eE]+)")
RE_NPU_TOP1   = re.compile(r"^\s*NPU top-1:\s*(\d+)")
RE_CPU_TOP1   = re.compile(r"^\s*CPU top-1:\s*(\d+)")
RE_MATCH      = re.compile(r"^\s*Match:\s*(YES|NO)")


@dataclass
class ParsedRun:
    compile_times: dict[str, float] = field(default_factory=dict)
    token_log: list[dict[str, Any]] = field(default_factory=list)
    k_corr_per_layer: dict[int, float] = field(default_factory=dict)
    v_corr_per_layer: dict[int, float] = field(default_factory=dict)
    logits_corr: float | None = None
    logits_max_err: float | None = None
    logits_mean_err: float | None = None
    npu_top1: int | None = None
    cpu_top1: int | None = None
    match: bool | None = None
    raw_stdout: str = ""
    raw_stderr: str = ""
    returncode: int = 0


def parse_inference_output(stdout: str) -> ParsedRun:
    """Best-effort line-by-line parser for the inference script's stdout."""
    p = ParsedRun()
    for line in stdout.splitlines():
        m = RE_COMPILED.match(line)
        if m:
            p.compile_times[m.group(1)] = float(m.group(2))
            continue
        m = RE_TOKEN.match(line)
        if m:
            p.token_log.append({
                "idx": int(m.group(1)),
                "id": int(m.group(2)),
                "time_ms": int(m.group(3)),
            })
            continue
        m = RE_KCACHE.match(line)
        if m:
            p.k_corr_per_layer[int(m.group(1))] = float(m.group(3))
            continue
        m = RE_VCACHE.match(line)
        if m:
            p.v_corr_per_layer[int(m.group(1))] = float(m.group(3))
            continue
        m = RE_LOGITS.match(line)
        if m:
            p.logits_corr = float(m.group(2))
            p.logits_max_err = float(m.group(3))
            p.logits_mean_err = float(m.group(4))
            continue
        m = RE_NPU_TOP1.match(line)
        if m:
            p.npu_top1 = int(m.group(1))
            continue
        m = RE_CPU_TOP1.match(line)
        if m:
            p.cpu_top1 = int(m.group(1))
            continue
        m = RE_MATCH.match(line)
        if m:
            p.match = (m.group(1) == "YES")
            continue
    return p


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def _git_rev() -> tuple[str, str]:
    """Return (branch, short_rev). Falls back to ('unknown', 'unknown')."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return branch, rev
    except Exception:
        return "unknown", "unknown"


def _peano_commit() -> str | None:
    """Parse the llvm-aie commit hash out of `clang++ --version` for the PEANO
    install that the Makefile resolves to. Returns the full 40-char hash or
    None if it can't be determined."""
    peano = os.environ.get("PEANO_INSTALL_DIR") or \
        "/home/jfifield/npu-dev-air/venv/lib/python3.12/site-packages/llvm-aie"
    clangxx = Path(peano) / "bin" / "clang++"
    if not clangxx.exists():
        return None
    try:
        out = subprocess.check_output(
            [str(clangxx), "--version"], text=True, stderr=subprocess.STDOUT,
        )
    except Exception:
        return None
    m = re.search(r"llvm-aie\s+([0-9a-f]{7,40})", out)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------


def run_inference(
    n_tokens: int = DEFAULT_N_TOKENS,
    prompt: str = DEFAULT_PROMPT,
    verify: bool = True,
    profile: bool = True,
    extra_args: list[str] | None = None,
    timeout: int = 600,
) -> ParsedRun:
    """Invoke llama32_1b_inference.py via subprocess with synthetic weights.

    Returns a ParsedRun with everything we could pull from stdout. We do NOT
    raise on non-zero exit -- caller decides; the snapshot writer still
    captures whatever it could parse.
    """
    if not BUILD_DIR.exists():
        raise RuntimeError(
            f"build_peano/ does not exist at {BUILD_DIR}. "
            f"Run `make compile` first."
        )

    cmd = [
        sys.executable, str(INFERENCE_SCRIPT),
        "--run-only",
        "--n-tokens", str(n_tokens),
        "--prompt", prompt,
        "--synthetic-weights",
        "--quant", "bf16",
    ]
    if profile:
        cmd.append("--profile")
    if verify:
        cmd.append("--verify")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    # Make sure PEANO points at the AIR-tree pip llvm-aie if available -- the
    # Makefile does this, we mirror the same default here.
    pip_peano = "/home/jfifield/npu-dev-air/venv/lib/python3.12/site-packages/llvm-aie"
    if Path(pip_peano, "bin", "clang++").exists():
        env.setdefault("PEANO_INSTALL_DIR", pip_peano)

    proc = subprocess.run(
        cmd, cwd=BUILD_DIR, capture_output=True, text=True,
        env=env, timeout=timeout,
    )
    parsed = parse_inference_output(proc.stdout)
    parsed.raw_stdout = proc.stdout
    parsed.raw_stderr = proc.stderr
    parsed.returncode = proc.returncode
    return parsed


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


def _summarize_token_times(token_log: list[dict[str, Any]]) -> dict[str, Any]:
    times = [t["time_ms"] for t in token_log]
    if not times:
        return {"count": 0, "median_ms": None, "p95_ms": None, "max_ms": None}
    times_sorted = sorted(times)
    p95_idx = max(0, int(round(0.95 * (len(times_sorted) - 1))))
    return {
        "count": len(times),
        "median_ms": float(statistics.median(times)),
        "p95_ms": float(times_sorted[p95_idx]),
        "max_ms": float(max(times)),
        "mean_ms": float(statistics.mean(times)),
    }


def _kv_summary(per_layer: dict[int, float]) -> dict[str, Any]:
    if not per_layer:
        return {"min": None, "max": None, "n_layers": 0}
    vals = list(per_layer.values())
    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n_layers": len(vals),
    }


def build_snapshot(
    parsed: ParsedRun,
    *,
    prompt: str,
    n_tokens: int,
    prompt_len: int | None = None,
) -> dict[str, Any]:
    branch, rev = _git_rev()
    snap = {
        "schema_version": 1,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": branch,
        "git_rev_short": rev,
        "peano_commit": _peano_commit(),
        "params": {
            "n_tokens": n_tokens,
            "prompt": prompt,
            "prompt_len_chars": len(prompt),
            "weights": "synthetic",
            "quant": "bf16",
            "model": "instruct",
        },
        "inference_returncode": parsed.returncode,
        "compile_times_sec": parsed.compile_times,
        "decode": {
            "tokens": parsed.token_log,
            "token_ids": [t["id"] for t in parsed.token_log],
            "latency": _summarize_token_times(parsed.token_log),
        },
        "verify": {
            "k_cache_corr": _kv_summary(parsed.k_corr_per_layer),
            "v_cache_corr": _kv_summary(parsed.v_corr_per_layer),
            "k_cache_per_layer": parsed.k_corr_per_layer,
            "v_cache_per_layer": parsed.v_corr_per_layer,
            "logits_corr": parsed.logits_corr,
            "logits_max_err": parsed.logits_max_err,
            "logits_mean_err": parsed.logits_mean_err,
            "npu_top1": parsed.npu_top1,
            "cpu_top1": parsed.cpu_top1,
            "match": parsed.match,
        },
    }
    if prompt_len is not None:
        snap["params"]["prompt_len_tokens"] = prompt_len
    return snap


def write_snapshot(snap: dict[str, Any], out_dir: Path = SNAPSHOT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = snap["timestamp_utc"].replace(":", "").replace("-", "")
    fname = f"{snap['branch']}_{snap['git_rev_short']}_{ts}.json"
    # Sanitize -- branch names can contain slashes.
    fname = fname.replace("/", "_")
    path = out_dir / fname
    path.write_text(json.dumps(snap, indent=2, sort_keys=True))
    return path


def latest_snapshot(out_dir: Path = SNAPSHOT_DIR) -> Path | None:
    if not out_dir.exists():
        return None
    snaps = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return snaps[-1] if snaps else None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _pct_change(old: float, new: float) -> float:
    if old == 0:
        return float("inf") if new != 0 else 0.0
    return (new - old) / old * 100.0


def compare_snapshots(
    a: dict[str, Any] | Path,
    b: dict[str, Any] | Path,
    *,
    per_kernel_threshold_pct: float = PER_KERNEL_REGRESSION_PCT,
    decode_p95_threshold_pct: float = DECODE_P95_REGRESSION_PCT,
    out=sys.stdout,
) -> dict[str, Any]:
    """Diff two snapshots. Returns a dict with `regressions` (list of strings)
    and `summary` (human-readable text). Prints a human-readable report.

    `failed` is True iff any per-kernel compile time grew by more than
    per_kernel_threshold_pct OR decode_p95 grew by more than
    decode_p95_threshold_pct. Token-sequence drift is reported but does NOT
    fail under synthetic weights (synthetic amplifies bf16 drift).
    """
    def _load(x):
        if isinstance(x, Path):
            return json.loads(Path(x).read_text())
        return x

    a, b = _load(a), _load(b)
    regressions: list[str] = []
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append(f"Snapshot comparison")
    lines.append(f"  A: {a.get('branch')}@{a.get('git_rev_short')} {a.get('timestamp_utc')}")
    lines.append(f"  B: {b.get('branch')}@{b.get('git_rev_short')} {b.get('timestamp_utc')}")
    lines.append("=" * 72)

    # Per-kernel compile times
    a_ct = a.get("compile_times_sec", {}) or {}
    b_ct = b.get("compile_times_sec", {}) or {}
    all_kernels = sorted(set(a_ct) | set(b_ct))
    if all_kernels:
        lines.append("\n-- Per-kernel compile time (s) --")
        lines.append(f"  {'kernel':40s}  {'A':>8s}  {'B':>8s}  {'delta%':>8s}")
        for k in all_kernels:
            av = a_ct.get(k)
            bv = b_ct.get(k)
            if av is None or bv is None:
                lines.append(f"  {k:40s}  {av if av is not None else '--':>8}  {bv if bv is not None else '--':>8}     n/a")
                continue
            delta = _pct_change(av, bv)
            flag = ""
            if delta > per_kernel_threshold_pct:
                flag = "  <-- REGRESSION"
                regressions.append(f"compile[{k}] +{delta:.1f}%")
            lines.append(f"  {k:40s}  {av:8.2f}  {bv:8.2f}  {delta:+7.1f}%{flag}")

    # Decode latency
    a_dec = (a.get("decode") or {}).get("latency", {}) or {}
    b_dec = (b.get("decode") or {}).get("latency", {}) or {}
    lines.append("\n-- Per-token decode latency (ms) --")
    for key in ("median_ms", "p95_ms", "max_ms", "mean_ms"):
        av = a_dec.get(key)
        bv = b_dec.get(key)
        if av is None or bv is None:
            lines.append(f"  {key:12s}  A={av}  B={bv}")
            continue
        delta = _pct_change(av, bv)
        flag = ""
        if key == "p95_ms" and delta > decode_p95_threshold_pct:
            flag = "  <-- REGRESSION"
            regressions.append(f"decode_p95 +{delta:.1f}%")
        lines.append(f"  {key:12s}  A={av:7.1f}  B={bv:7.1f}  delta={delta:+6.1f}%{flag}")

    # Token sequence
    a_tokens = (a.get("decode") or {}).get("token_ids") or []
    b_tokens = (b.get("decode") or {}).get("token_ids") or []
    lines.append("\n-- Decode token id sequence --")
    lines.append(f"  A: {a_tokens}")
    lines.append(f"  B: {b_tokens}")
    common = min(len(a_tokens), len(b_tokens))
    diff_count = sum(1 for i in range(common) if a_tokens[i] != b_tokens[i])
    diff_count += abs(len(a_tokens) - len(b_tokens))
    lines.append(f"  diff: {diff_count} token(s) differ (informational; synthetic weights amplify drift)")

    # Verify deltas (informational only)
    av_v = a.get("verify") or {}
    bv_v = b.get("verify") or {}
    lines.append("\n-- Correctness summary --")
    for label, key in [
        ("K_cache corr min", ("k_cache_corr", "min")),
        ("V_cache corr min", ("v_cache_corr", "min")),
        ("logits_corr",       None),
        ("npu_top1",          None),
        ("cpu_top1",          None),
        ("match",             None),
    ]:
        if isinstance(key, tuple):
            ax = ((av_v.get(key[0]) or {}).get(key[1]))
            bx = ((bv_v.get(key[0]) or {}).get(key[1]))
        else:
            label_key = label.replace(" ", "_") if not key else key
            ax = av_v.get(label_key if not key else key)
            bx = bv_v.get(label_key if not key else key)
        lines.append(f"  {label:20s}  A={ax}  B={bx}")

    lines.append("")
    if regressions:
        lines.append(f"FAIL: {len(regressions)} regression(s):")
        for r in regressions:
            lines.append(f"  - {r}")
    else:
        lines.append("OK: no regressions detected")

    summary = "\n".join(lines)
    print(summary, file=out)
    return {
        "failed": bool(regressions),
        "regressions": regressions,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------


def _capture_and_write(n_tokens: int = DEFAULT_N_TOKENS,
                       prompt: str = DEFAULT_PROMPT) -> tuple[Path, dict[str, Any]]:
    parsed = run_inference(n_tokens=n_tokens, prompt=prompt,
                           verify=True, profile=True)
    snap = build_snapshot(parsed, prompt=prompt, n_tokens=n_tokens)
    path = write_snapshot(snap)
    return path, snap


def test_phase_snapshot_capture():
    """Capture a new per-phase snapshot. Always passes if the inference run
    completes successfully. Regression detection is handled by
    test_phase_snapshot_no_regression() below."""
    path, snap = _capture_and_write()
    print(f"\n[snapshot] wrote {path}")
    assert snap["inference_returncode"] == 0, (
        f"inference exited with code {snap['inference_returncode']}; "
        f"see stderr in raw output"
    )
    # Spot-checks: we should have at least some compile times (or run on a
    # warm cache => empty is also fine) and at least one decode token.
    assert snap["decode"]["latency"]["count"] >= 1, "no decode tokens parsed"


def test_phase_snapshot_no_regression():
    """If a prior snapshot exists, compare against it and fail on regressions.
    Otherwise (first run) skip with a clear message."""
    import pytest
    snaps = sorted(SNAPSHOT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if len(snaps) < 2:
        pytest.skip(
            "need at least 2 snapshots to compare; "
            "run test_phase_snapshot_capture twice to bootstrap."
        )
    result = compare_snapshots(snaps[-2], snaps[-1])
    assert not result["failed"], result["summary"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-phase metrics snapshot harness for llama32_1b PythoC."
    )
    sub = p.add_subparsers(dest="cmd")

    cap = sub.add_parser("capture", help="Run inference, write a snapshot, "
                                          "diff vs latest prior snapshot.")
    cap.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS)
    cap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)

    cmp = sub.add_parser("compare", help="Diff two snapshot JSON files.")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)

    args = p.parse_args(argv)
    if args.cmd is None or args.cmd == "capture":
        # Find the prior latest BEFORE writing the new one.
        prior = latest_snapshot()
        n_tokens = getattr(args, "n_tokens", DEFAULT_N_TOKENS)
        prompt = getattr(args, "prompt", DEFAULT_PROMPT)
        path, snap = _capture_and_write(n_tokens=n_tokens, prompt=prompt)
        print(f"\n[snapshot] wrote {path}")
        if snap["inference_returncode"] != 0:
            print(f"[snapshot] WARNING: inference exited with "
                  f"code {snap['inference_returncode']}")
            # Dump tail of stderr to help debug.
            tail = "\n".join(
                json.loads(path.read_text()).get("raw_stderr", "").splitlines()[-20:]
            )
            if tail.strip():
                print(f"[snapshot] stderr tail:\n{tail}")
        if prior is not None and prior.exists():
            print()
            compare_snapshots(prior, path)
        else:
            print("\n[snapshot] no prior snapshot to diff against -- "
                  "this is the baseline.")
        return 0 if snap["inference_returncode"] == 0 else 1
    elif args.cmd == "compare":
        result = compare_snapshots(args.a, args.b)
        return 1 if result["failed"] else 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
