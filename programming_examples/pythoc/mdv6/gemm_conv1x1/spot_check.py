#!/usr/bin/env python3
"""Spot-check two GEMM conv1x1 xclbins on NPU vs numpy reference.

Usage:
  flock /tmp/npu-dev.lock python3 spot_check.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel


HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"


def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(x.astype(np.float32)).reshape(-1)
    return (flat.view(np.uint32) >> 16).astype(np.uint16)


def bf16_u16_to_f32(x: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(x.astype(np.uint32)).reshape(-1)
    return (flat << 16).view(np.float32)


def pack_weights_fused(wt_f32, bn_w_f32, bn_b_f32, ic, oc):
    """Non-K-blocked packing: [IC/8, OC/8, 8ic, 8oc] + bn_w(oc) + bn_b(oc)."""
    assert ic % 8 == 0 and oc % 8 == 0
    ib, ob = ic // 8, oc // 8
    w = wt_f32.reshape(ib, 8, ob, 8).transpose(0, 2, 1, 3)
    w_u16 = f32_to_bf16_u16(w.reshape(-1).astype(np.float32))
    bn_w_u = f32_to_bf16_u16(bn_w_f32)
    bn_b_u = f32_to_bf16_u16(bn_b_f32)
    return np.concatenate([w_u16, bn_w_u, bn_b_u])


def pack_weight_chunk_kblocked(wt_kb_f32, bn_w_f32, bn_b_f32, k_block, oc):
    """K-blocked packing per chunk: [kb/8, oc/8, 8ic, 8oc] + bn_w(oc) + bn_b(oc)."""
    assert k_block % 8 == 0 and oc % 8 == 0
    kb, ob = k_block // 8, oc // 8
    w = wt_kb_f32.reshape(kb, 8, ob, 8).transpose(0, 2, 1, 3)
    w_u16 = f32_to_bf16_u16(w.reshape(-1).astype(np.float32))
    bn_w_u = f32_to_bf16_u16(bn_w_f32)
    bn_b_u = f32_to_bf16_u16(bn_b_f32)
    return np.concatenate([w_u16, bn_w_u, bn_b_u])


def gemm_bn_silu_ref(In_f32, W_f32, bn_w_f32, bn_b_f32):
    """Reference: matmul + BN + fast_sigmoid SiLU (matches C++ kernel tail)."""
    mac = In_f32.astype(np.float32) @ W_f32.astype(np.float32)
    bned = bn_w_f32[None, :] * mac + bn_b_f32[None, :]
    ax = np.abs(bned)
    return bned * (0.5 + bned / (2.0 + 2.0 * ax))


def run_config(name, *, tile_m, ic, oc, k_block, ppc, n_cores=32, tol=0.5):
    xclbin = BUILD_DIR / f"{name}.xclbin"
    insts = BUILD_DIR / f"{name}.bin"
    assert xclbin.exists(), f"missing {xclbin}"
    assert insts.exists(), f"missing {insts}"

    rng = np.random.default_rng(42)

    # Host shapes:
    #   In : [n_cores, ppc, tile_m, ic]  bf16-as-u16
    #   W  : either oc*ic + 2*oc, OR n_k_blocks * (k_block*oc + 2*oc)
    #   Out: [n_cores, ppc, tile_m, oc]  bf16-as-u16
    In_f32 = rng.standard_normal(
        (n_cores, ppc, tile_m, ic), dtype=np.float32
    ) * 0.1
    W_f32 = rng.standard_normal((ic, oc), dtype=np.float32) * 0.1
    # Identity BN (gamma=1, beta=0) so reference == matmul + SiLU.
    bn_w_f32 = np.ones(oc, dtype=np.float32)
    bn_b_f32 = np.zeros(oc, dtype=np.float32)

    # Round inputs/weights to bf16 for the reference.
    In_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(In_f32)).reshape(
        n_cores, ppc, tile_m, ic
    )
    W_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(W_f32)).reshape(ic, oc)
    bn_w_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_w_f32))
    bn_b_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_b_f32))

    # Reference per (core, patch).
    ref = np.zeros((n_cores, ppc, tile_m, oc), dtype=np.float32)
    for c in range(n_cores):
        for p in range(ppc):
            ref[c, p] = gemm_bn_silu_ref(
                In_bf16[c, p], W_bf16, bn_w_bf16, bn_b_bf16
            )

    # Pack inputs (row-major over [c,p,m,ic]).
    in_u16 = f32_to_bf16_u16(In_bf16.reshape(-1))

    # Pack weights.
    if k_block > 0:
        n_k_blocks = ic // k_block
        chunks = []
        for kbi in range(n_k_blocks):
            s = kbi * k_block
            chunks.append(
                pack_weight_chunk_kblocked(
                    W_bf16[s:s + k_block, :], bn_w_bf16, bn_b_bf16,
                    k_block, oc,
                )
            )
        wt_u16 = np.concatenate(chunks)
    else:
        wt_u16 = pack_weights_fused(W_bf16, bn_w_bf16, bn_b_bf16, ic, oc)

    print(f"  [{name}] in={len(in_u16)} wt={len(wt_u16)} "
          f"out={n_cores*ppc*tile_m*oc} (k_block={k_block})")

    # Load and run.
    kh = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts)))
    in_buf = iron.tensor(in_u16, dtype=np.uint16)
    wt_buf = iron.tensor(wt_u16, dtype=np.uint16)
    out_buf = iron.zeros(n_cores * ppc * tile_m * oc, dtype=np.uint16)

    print(f"  [{name}] running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f} ms)")

    out_u16 = np.array(out_buf.numpy()).copy()
    out_f32 = bf16_u16_to_f32(out_u16).reshape(n_cores, ppc, tile_m, oc)

    diff = np.abs(out_f32 - ref)
    max_diff = float(diff.max())
    rel = max_diff / (float(np.max(np.abs(ref))) + 1e-10)
    print(f"  [{name}] max abs diff = {max_diff:.6f}  (rel = {rel:.4f})")
    ok = max_diff < tol
    print(f"  [{name}] {'PASS' if ok else 'FAIL'} (tol={tol})")
    return ok, max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=("nonkb", "kb", "both"), default="both",
    )
    args = parser.parse_args()

    results = {}
    if args.config in ("nonkb", "both"):
        ok, md = run_config(
            "gemm_t100_ic96_oc96_p1",
            tile_m=100, ic=96, oc=96, k_block=0, ppc=1, n_cores=32,
        )
        results["nonkb"] = (ok, md)
    if args.config in ("kb", "both"):
        ok, md = run_config(
            "gemm_t20_ic256_oc256_kb64_p1",
            tile_m=20, ic=256, oc=256, k_block=64, ppc=1, n_cores=32,
        )
        results["kb"] = (ok, md)

    print("\nSummary:")
    for k, (ok, md) in results.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'}  max abs diff = {md:.6f}")
    return 0 if all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
