#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate direct packed-AWQ GEMV NPU primitive on representative model tensors.

The current correctness-first AWQ GEMV primitive stages the whole qweight/params
argument into L1, so full model M dimensions are not expected to fit yet. This
script validates real model K/group-size/data semantics by slicing representative
row blocks from each full projection shape and comparing NPU output to the direct
CPU AWQ reference. It also optionally probes full-M compile feasibility.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
from safetensors import safe_open

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLE_DIR))

from kernel_builder.aie_ir_gen import build_awq_gemv_ir  # noqa: E402
from kernel_builder.cache import KernelCache  # noqa: E402
from llama32_1b_awq_runtime import awq_gemv_npu, awq_gemv_npu_tiled  # noqa: E402
from llama32_1b_weights import AwqLinear, awq_gemv_cpu_reference  # noqa: E402


@dataclass
class ValidationResult:
    prefix: str
    full_m: int
    k: int
    group_size: int
    row_start: int
    rows_validated: int
    max_abs: float
    mean_abs: float
    allclose_atol: float
    passed: bool
    compile_s: float
    run_s: float
    note: str = ""


# One representative per distinct projection shape in Llama 3.2 1B AWQ.
REPRESENTATIVES = [
    "model.layers.0.self_attn.k_proj",      # M=512, K=2048
    "model.layers.0.self_attn.q_proj",      # M=2048, K=2048
    "model.layers.0.mlp.gate_proj",         # M=8192, K=2048
    "model.layers.0.mlp.down_proj",         # M=2048, K=8192
    "lm_head",                              # M=128256, K=2048
]


def _to_numpy(t):
    if str(t.dtype) == "torch.bfloat16":
        return t.float().cpu().numpy().astype(bfloat16)
    return t.cpu().numpy()


def _load_awq_slice(model_path: Path, prefix: str, row_start: int, rows: int):
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as f:
        q = f.get_tensor(prefix + ".qweight_repacked")
        p = f.get_tensor(prefix + ".params_interleaved")
        full_m = int(q.shape[0])
        k = int(q.shape[1]) * 2
        groups = int(p.shape[1]) // 2
        group_size = k // groups
        row_end = min(full_m, row_start + rows)
        row_start = max(0, row_end - rows)
        q_np = _to_numpy(q[row_start:row_end]).astype(np.uint8, copy=False)
        p_np = _to_numpy(p[row_start:row_end]).astype(bfloat16, copy=False)
    return full_m, AwqLinear(
        qweight=np.ascontiguousarray(q_np),
        params=np.ascontiguousarray(p_np),
        k=k,
        m=int(q_np.shape[0]),
        group_size=group_size,
    )


def _load_awq_full(model_path: Path, prefix: str):
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as f:
        q = f.get_tensor(prefix + ".qweight_repacked")
        p = f.get_tensor(prefix + ".params_interleaved")
        q_np = _to_numpy(q).astype(np.uint8, copy=False)
        p_np = _to_numpy(p).astype(bfloat16, copy=False)
        k = int(q_np.shape[1]) * 2
        groups = int(p_np.shape[1]) // 2
        group_size = k // groups
    return AwqLinear(
        qweight=np.ascontiguousarray(q_np),
        params=np.ascontiguousarray(p_np),
        k=k,
        m=int(q_np.shape[0]),
        group_size=group_size,
    )


def _deterministic_x(k: int) -> np.ndarray:
    # Mixed signs and fractional BF16 values; deterministic without touching RNG.
    i = np.arange(k, dtype=np.float32)
    x = ((i % 17) - 8.0) / 4.0
    x += ((i % 5) - 2.0) / 16.0
    return x.astype(bfloat16)


def validate_slice(cache: KernelCache, model_path: Path, prefix: str, rows: int, row_start: int, atol: float, *, variant: str = "scalar"):
    full_m, awq = _load_awq_slice(model_path, prefix, row_start=row_start, rows=rows)
    x = _deterministic_x(awq.k)

    t0 = time.perf_counter()
    out = awq_gemv_npu(cache, x, awq, variant=variant).astype(np.float32)
    t1 = time.perf_counter()
    ref = awq_gemv_cpu_reference(x, awq, dtype=np.float32).astype(np.float32)
    abs_err = np.abs(out - ref)
    passed = bool(np.all(abs_err <= atol))
    return ValidationResult(
        prefix=prefix,
        full_m=full_m,
        k=awq.k,
        group_size=awq.group_size,
        row_start=row_start,
        rows_validated=awq.m,
        max_abs=float(abs_err.max(initial=0.0)),
        mean_abs=float(abs_err.mean() if abs_err.size else 0.0),
        allclose_atol=atol,
        passed=passed,
        compile_s=0.0,  # KernelCache currently combines compile/run in wrapper timing.
        run_s=float(t1 - t0),
    )


def validate_full_tiled(cache: KernelCache, model_path: Path, prefix: str, tile_m: int, ref_tile_m: int, atol: float, *, variant: str = "scalar"):
    awq = _load_awq_full(model_path, prefix)
    x = _deterministic_x(awq.k)

    t0 = time.perf_counter()
    out = awq_gemv_npu_tiled(cache, x, awq, tile_m=tile_m, variant=variant).astype(np.float32)
    t1 = time.perf_counter()

    ref = np.empty((awq.m,), dtype=np.float32)
    for row_start in range(0, awq.m, ref_tile_m):
        row_end = min(awq.m, row_start + ref_tile_m)
        tile = AwqLinear(
            qweight=np.ascontiguousarray(awq.qweight[row_start:row_end], dtype=np.uint8),
            params=np.ascontiguousarray(awq.params[row_start:row_end], dtype=bfloat16),
            k=awq.k,
            m=row_end - row_start,
            group_size=awq.group_size,
        )
        ref[row_start:row_end] = awq_gemv_cpu_reference(x, tile, dtype=np.float32).astype(np.float32)

    abs_err = np.abs(out - ref)
    passed = bool(np.all(abs_err <= atol))
    return ValidationResult(
        prefix=prefix,
        full_m=awq.m,
        k=awq.k,
        group_size=awq.group_size,
        row_start=0,
        rows_validated=awq.m,
        max_abs=float(abs_err.max(initial=0.0)),
        mean_abs=float(abs_err.mean() if abs_err.size else 0.0),
        allclose_atol=atol,
        passed=passed,
        compile_s=0.0,
        run_s=float(t1 - t0),
        note=f"full projection via NPU tile_m={tile_m}, CPU ref_tile_m={ref_tile_m}",
    )


def probe_full_compile(model_path: Path, prefix: str, *, variant: str = "scalar"):
    full_m, awq = _load_awq_slice(model_path, prefix, row_start=0, rows=1)
    try:
        t0 = time.perf_counter()
        build_awq_gemv_ir(awq.k, full_m, awq.group_size, variant=variant, verbose=False)
        return {"prefix": prefix, "full_m": full_m, "k": awq.k, "ok": True, "seconds": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001 - diagnostic probe
        return {
            "prefix": prefix,
            "full_m": full_m,
            "k": awq.k,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[-1] if str(exc) else repr(exc),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=_EXAMPLE_DIR / "awq_repacked")
    parser.add_argument("--cache-dir", type=Path, default=Path("awq_model_shape_validation_cache"))
    parser.add_argument("--rows-k2048", type=int, default=8)
    parser.add_argument("--rows-k8192", type=int, default=2)
    parser.add_argument("--full-projection", action="store_true")
    parser.add_argument("--full-tile-m-k2048", type=int, default=32)
    parser.add_argument("--full-tile-m-k8192", type=int, default=4)
    parser.add_argument("--ref-tile-m", type=int, default=32)
    parser.add_argument(
        "--max-full-m",
        type=int,
        default=8192,
        help="Skip --full-projection validation for shapes with M above this limit (default skips full lm_head).",
    )
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.75)
    parser.add_argument("--probe-full-compile", action="store_true")
    parser.add_argument("--variant", choices=["vecdeq"], default="vecdeq")
    args = parser.parse_args()

    cache = KernelCache(cache_dir=args.cache_dir, verbose=False)
    results = []
    for prefix in REPRESENTATIVES:
        # K is known only after reading shape, so load one row first.
        full_m, one = _load_awq_slice(args.model, prefix, row_start=0, rows=1)
        if args.full_projection and full_m <= args.max_full_m:
            tile_m = args.full_tile_m_k8192 if one.k == 8192 else args.full_tile_m_k2048
            print(
                f"VALIDATE_FULL {prefix}: full M={full_m}, K={one.k}, group={one.group_size}, tile_m={tile_m}",
                flush=True,
            )
            result = validate_full_tiled(
                cache,
                args.model,
                prefix,
                tile_m=tile_m,
                ref_tile_m=args.ref_tile_m,
                atol=args.atol,
                variant=args.variant,
            )
        else:
            rows = args.rows_k8192 if one.k == 8192 else args.rows_k2048
            if args.full_projection:
                print(
                    f"VALIDATE_SLICE {prefix}: full M={full_m} exceeds max_full_m={args.max_full_m}; "
                    f"validating rows={rows}",
                    flush=True,
                )
            else:
                print(
                    f"VALIDATE {prefix}: full M={full_m}, K={one.k}, group={one.group_size}, rows={rows}",
                    flush=True,
                )
            result = validate_slice(
                cache,
                args.model,
                prefix,
                rows=rows,
                row_start=args.row_start,
                atol=args.atol,
                variant=args.variant,
            )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    full_compile = []
    if args.probe_full_compile:
        for prefix in REPRESENTATIVES:
            print(f"PROBE_FULL_COMPILE {prefix}", flush=True)
            item = probe_full_compile(args.model, prefix, variant=args.variant)
            full_compile.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)

    summary = {
        "passed": all(r.passed for r in results),
        "results": [asdict(r) for r in results],
        "full_compile_probe": full_compile,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
