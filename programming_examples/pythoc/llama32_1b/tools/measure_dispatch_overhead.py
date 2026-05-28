#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Measure inner aiex.configure/aiex.run dispatch overhead.

The default benchmark builds tiny no-op sub-devices and a dispatcher whose
runtime sequence fires 0, 1, 2, 4, 8, and 16 inner runs. The count=0 case is
an outer XRT-launch baseline; the slope across higher counts estimates the
per-inner-run PDI/configure cost under the same aiecc ``--expand-load-pdis``
path used by the real kernels.

Optionally, the script can also run the cached production ``o_gemv_ffn`` once
per iteration to record the current full decode-FFN launch cost.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from ml_dtypes import bfloat16

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from aie.dialects.aie import AIEDevice, device  # noqa: E402
from aie.dialects.aiex import runtime_sequence  # noqa: E402
from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp  # noqa: E402
from aie.extras.context import mlir_mod_ctx  # noqa: E402
from aie.ir import InsertionPoint  # noqa: E402

from builders._emit import bf16_np  # noqa: E402
from builders.o_gemv_ffn import (  # noqa: E402
    EMB_DIM,
    HIDDEN_DIM,
    build_o_gemv_ffn_module,
)
from kernel_builder.cache import KernelCache  # noqa: E402


def _parse_counts(text: str) -> list[int]:
    counts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if any(c < 0 for c in counts):
        raise argparse.ArgumentTypeError("counts must be non-negative")
    return counts


def _build_noop_dispatch_module(run_count: int, distinct_devices: int) -> str:
    if distinct_devices < 1:
        raise ValueError("distinct_devices must be >= 1")

    with mlir_mod_ctx() as ctx:
        def emit_noop_device(sym: str) -> None:
            @device(AIEDevice.npu2, sym_name=sym)
            def _noop_dev():
                @runtime_sequence(bf16_np(1), sym_name=f"{sym}_sequence")
                def _seq(arg0):
                    pass

        for idx in range(distinct_devices):
            emit_noop_device(f"noop{idx}")

        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(bf16_np(1), sym_name="dispatch_overhead")
            def _outer(arg0):
                for i in range(run_count):
                    sym = f"noop{i % distinct_devices}"
                    cfg = ConfigureOp(symbol=sym)
                    blk = cfg.body.blocks.append()
                    with InsertionPoint(blk):
                        RunOp(
                            runtime_sequence_symbol=f"{sym}_sequence",
                            args=[arg0],
                        )

        return str(ctx.module)


def _variant_name(run_count: int, distinct_devices: int) -> str:
    return f"dispatch_noop_r{run_count}_d{distinct_devices}"


def _compile_noop_variants(
    cache: KernelCache,
    counts: Iterable[int],
    distinct_devices: int,
) -> None:
    for count in counts:
        name = _variant_name(count, distinct_devices)
        ir = _build_noop_dispatch_module(count, distinct_devices)
        cache.compile_and_cache(name, ir, instance_name="dispatch_overhead")
    cache._save_manifest()


def _time_kernel(
    cache: KernelCache,
    name: str,
    inputs: list[np.ndarray],
    *,
    output_indices: list[int],
    static_input_indices: set[int] | None = None,
    bo_key: str | None = None,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        cache.load_and_run(
            name,
            {},
            *inputs,
            output_indices=output_indices,
            static_input_indices=static_input_indices,
            bo_key=bo_key,
        )

    samples_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cache.load_and_run(
            name,
            {},
            *inputs,
            output_indices=output_indices,
            static_input_indices=static_input_indices,
            bo_key=bo_key,
        )
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def _summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples_ms),
        "median": statistics.median(samples_ms),
        "min": min(samples_ms),
        "max": max(samples_ms),
    }


def _fit_slope(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Return least-squares intercept and slope in ms/run."""
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _o_gemv_ffn_inputs() -> list[np.ndarray]:
    # Small random values avoid denormal/extreme behavior while preserving the
    # production ABI and BO sizes. Static weights are written only once by the
    # timing path.
    rng = np.random.default_rng(0)

    def rand(shape, scale=0.01):
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(bfloat16)

    return [
        rand((EMB_DIM, EMB_DIM)),
        rand((EMB_DIM,)),
        np.zeros((EMB_DIM,), dtype=bfloat16),
        rand((EMB_DIM,)),
        np.zeros((EMB_DIM,), dtype=bfloat16),
        rand((EMB_DIM,), scale=1.0),
        np.zeros((EMB_DIM,), dtype=bfloat16),
        rand((HIDDEN_DIM, EMB_DIM)),
        np.zeros((HIDDEN_DIM,), dtype=bfloat16),
        rand((HIDDEN_DIM, EMB_DIM)),
        np.zeros((HIDDEN_DIM,), dtype=bfloat16),
        np.zeros((HIDDEN_DIM,), dtype=bfloat16),
        rand((EMB_DIM, HIDDEN_DIM)),
        np.zeros((EMB_DIM,), dtype=bfloat16),
        np.zeros((EMB_DIM,), dtype=bfloat16),
    ]


def _compile_o_gemv_add_variants(
    cache: KernelCache,
    counts: Iterable[int],
) -> None:
    add_syms = ("a1_eltwise_add_seg", "a2_eltwise_add_seg")
    for count in counts:
        seq = tuple(add_syms[i % len(add_syms)] for i in range(count))
        name = f"o_gemv_ffn_add_dispatch_r{count}"
        ir = build_o_gemv_ffn_module(dispatch_sequence=seq)
        cache.compile_and_cache(name, ir, instance_name="o_gemv_ffn")
    cache._save_manifest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        type=_parse_counts,
        default=_parse_counts("0,1,2,4,8,16"),
        help="Comma-separated inner run counts for slope variants.",
    )
    parser.add_argument(
        "--distinct-devices",
        type=int,
        default=2,
        help="Number of no-op sub-devices to cycle through; 2 forces PDI swaps.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=PROJECT_DIR / "build_peano" / "dispatch_overhead",
        help="Cache/work directory for generated no-op benchmark ELFs.",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--include-o-gemv-add",
        action="store_true",
        help="Also compile/run production-shaped add-only o_gemv_ffn dispatcher variants.",
    )
    parser.add_argument(
        "--include-current-o-gemv-ffn",
        action="store_true",
        help="Also time the cached production o_gemv_ffn from --decode-cache-dir.",
    )
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=PROJECT_DIR / "build_peano" / "decode_kernel_cache",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.workdir = args.workdir.resolve()
    args.decode_cache_dir = args.decode_cache_dir.resolve()

    # KernelCache discovers external object files from Path.cwd(). The normal
    # build leaves those objects in build_peano/, so use that cwd for
    # production-shaped variants that link external kernels.
    if args.include_o_gemv_add:
        obj_dir = PROJECT_DIR / "build_peano"
        if obj_dir.exists():
            os.chdir(obj_dir)

    args.workdir.mkdir(parents=True, exist_ok=True)
    cache = KernelCache(cache_dir=args.workdir, verbose=args.verbose)

    if args.skip_compile:
        if not cache.load_manifest():
            raise RuntimeError(f"no manifest found in {args.workdir}")
    else:
        _compile_noop_variants(cache, args.counts, args.distinct_devices)
        if args.include_o_gemv_add:
            _compile_o_gemv_add_variants(cache, args.counts)

    if args.compile_only:
        print(f"Compiled benchmark variants in {args.workdir}")
        return 0

    dummy = [np.zeros((1,), dtype=bfloat16)]
    rows = []
    print("\nNo-op dispatch variants")
    print("runs  mean_ms  median_ms  min_ms  max_ms")
    for count in args.counts:
        name = _variant_name(count, args.distinct_devices)
        samples = _time_kernel(
            cache,
            name,
            dummy,
            output_indices=[],
            static_input_indices={0},
            bo_key=name,
            warmup=args.warmup,
            iters=args.iters,
        )
        stats = _summarize(samples)
        rows.append((count, stats["median"]))
        print(
            f"{count:4d}  {stats['mean']:7.3f}  {stats['median']:9.3f}"
            f"  {stats['min']:6.3f}  {stats['max']:6.3f}"
        )

    fit_x = [x for x, _ in rows]
    fit_y = [y for _, y in rows]
    intercept, slope = _fit_slope(fit_x, fit_y)
    print(
        f"\nFit on medians: intercept={intercept:.3f} ms, "
        f"slope={slope * 1000.0:.2f} us/inner-run"
    )
    if 0 in fit_x:
        base = dict(rows)[0]
        print("Delta from count=0 baseline:")
        for count, median in rows:
            if count:
                print(f"  {count:4d}: {(median - base) * 1000.0 / count:8.2f} us/run")

    if args.include_o_gemv_add:
        print("\nProduction-shaped add-only o_gemv_ffn dispatcher variants")
        add_inputs = _o_gemv_ffn_inputs()
        add_rows = []
        print("runs  median_ms")
        for count in args.counts:
            name = f"o_gemv_ffn_add_dispatch_r{count}"
            samples = _time_kernel(
                cache,
                name,
                add_inputs,
                output_indices=[],
                static_input_indices={0, 7, 9, 12},
                bo_key=name,
                warmup=args.warmup,
                iters=args.iters,
            )
            median = _summarize(samples)["median"]
            add_rows.append((count, median))
            print(f"{count:4d}  {median:9.3f}")
        if add_rows:
            add_x = [x for x, _ in add_rows]
            add_y = [y for _, y in add_rows]
            add_intercept, add_slope = _fit_slope(add_x, add_y)
            print(
                f"Fit on medians: intercept={add_intercept:.3f} ms, "
                f"slope={add_slope * 1000.0:.2f} us/inner-run"
            )
            if 0 in add_x:
                base = dict(add_rows)[0]
                print("Delta from count=0 baseline:")
                for count, median in add_rows:
                    if count:
                        print(f"  {count:4d}: {(median - base) * 1000.0 / count:8.2f} us/run")

    if args.include_current_o_gemv_ffn:
        print("\nCurrent cached o_gemv_ffn")
        prod_cache = KernelCache(cache_dir=args.decode_cache_dir, verbose=args.verbose)
        if not prod_cache.load_manifest() or "o_gemv_ffn" not in prod_cache.artifacts:
            print(f"  missing o_gemv_ffn in {args.decode_cache_dir}; run make compile first")
        else:
            prod_inputs = _o_gemv_ffn_inputs()
            samples = _time_kernel(
                prod_cache,
                "o_gemv_ffn",
                prod_inputs,
                output_indices=[14],
                static_input_indices={0, 7, 9, 12},
                bo_key="dispatch_bench_o_gemv_ffn",
                warmup=args.warmup,
                iters=args.iters,
            )
            stats = _summarize(samples)
            print(
                f"  mean={stats['mean']:.3f} ms median={stats['median']:.3f} ms "
                f"min={stats['min']:.3f} ms max={stats['max']:.3f} ms"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
