#!/usr/bin/env python3
# verify_one_xclbin.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2

"""End-to-end NPU validation for rep_elan_bf16_pythoc kernels.

Builds an IRON multicore xclbin around one of the .o files from
`build_kernels.py`, runs it on the NPU, and compares against a numpy/torch
reference. Two configurations are exercised:

  * `--kernel conv3x3`  — small mc_re8_rn3-ish config:
        (n_cores=4, tile_h=4, tile_w=4, ic=64, oc=16, ks=3, stride=1, ppc=1)

  * `--kernel gemm_kblocked` — gemm_t20_ic256_oc256_kb64_p1-ish:
        (n_cores=4, tile_m=20, ic=256, oc=256, k_block=64, ppc=1)

Both expect to print `PASS!` on hardware.

Usage
-----
  source /home/jfifield/npu-dev-pythoc/env.sh
  python build_kernels.py
  flock /tmp/npu-dev.lock python verify_one_xclbin.py --kernel conv3x3
  flock /tmp/npu-dev.lock python verify_one_xclbin.py --kernel gemm_kblocked
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure PythoC + IRON imports come from the local install.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aie.iron as iron
from aie.iron import (
    Buffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.pythoc import PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module

from rep_elan_bf16_pythoc import (
    DEFAULT_BUILD_DIR,
    make_conv3x3_fused_packed_bf16,
    make_gemm_conv1x1_kblocked_bf16,
    make_gemm_conv1x1_fused_packed_bf16,
)


# ─────────────────────────────────────────────────────────────────────────────
# bf16 ↔ uint16 helpers (matches existing pythoc examples)
# ─────────────────────────────────────────────────────────────────────────────


def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    """f32 → bf16 (truncated, no rounding), packed as uint16."""
    flat = np.ascontiguousarray(x.astype(np.float32)).reshape(-1)
    return (flat.view(np.uint32) >> 16).astype(np.uint16)


def bf16_u16_to_f32(x: np.ndarray) -> np.ndarray:
    """bf16-as-uint16 → f32."""
    flat = np.ascontiguousarray(x.astype(np.uint32)).reshape(-1)
    return (flat << 16).view(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Conv3x3 multicore IRON wrapper (lifted from ml/mdv6/conv/aie2_multicore.py)
# ─────────────────────────────────────────────────────────────────────────────


def build_conv3x3_module(
    dev,
    *,
    tile_h: int,
    tile_w: int,
    ic: int,
    oc: int,
    kernel_size: int = 3,
    stride_val: int = 1,
    padding_val: int = 1,
    n_cores: int = 4,
    patches_per_core: int = 1,
    input_depth: int = 1,
    obj_dir: Optional[Path] = None,
):
    """Lifted from ml/mdv6/conv/aie2_multicore.py with `Kernel(...)` swapped
    for `PythocKernel(name, ".o", [...])`."""
    if kernel_size == 1:
        padding_val = 0

    patch_h = (tile_h - 1) * stride_val + kernel_size
    patch_w = (tile_w - 1) * stride_val + kernel_size
    patch_size_raw = patch_h * patch_w * ic
    patch_size = patch_size_raw + (patch_size_raw % 2)  # DMA align
    conv_weight_size = oc * ic * kernel_size * kernel_size
    weight_block_size = conv_weight_size + 2 * oc
    output_tile_size = tile_h * tile_w * oc

    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col

    patch_ty = np.ndarray[(patch_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]

    core_input_size = patches_per_core * patch_size
    core_output_size = patches_per_core * output_tile_size

    host_input_size = n_cores * core_input_size
    host_output_size = n_cores * core_output_size
    host_input_ty = np.ndarray[(host_input_size,), np.dtype[np.uint16]]
    host_output_ty = np.ndarray[(host_output_size,), np.dtype[np.uint16]]

    # The IRON Kernel resolves `.o` filenames relative to aiecc's --tmpdir
    # (work_dir). Pass the bare filename and stage the .o into work_dir
    # before calling compile_mlir_module. See _stage_objects below.
    kernel = make_conv3x3_fused_packed_bf16(
        patch_ty, weight_ty, output_tile_ty, build_dir=obj_dir
    )

    RTP_LEN = 6  # (tile_h, tile_w, ic, oc, stride, padding)
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    init_rtp = np.array(
        [tile_h, tile_w, ic, oc, stride_val, padding_val], dtype=np.int32
    )
    rtps = [
        Buffer(rtp_ty, name=f"rtp_{i}", initial_value=init_rtp, use_write_rtp=True)
        for i in range(n_cores)
    ]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    def core_fn(of_in, of_wt, of_out, kern, my_rtp, barrier):
        barrier.wait_for_value(1)
        t_h = my_rtp[0]
        t_w = my_rtp[1]
        ic_v = my_rtp[2]
        oc_v = my_rtp[3]
        str_v = my_rtp[4]
        pad_v = my_rtp[5]
        elem_wt = of_wt.acquire(1)
        for _ in range_(patches_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kern(elem_in, elem_wt, elem_out, t_h, t_w, ic_v, oc_v, str_v, pad_v)
            of_in.release(1)
            of_out.release(1)
        of_wt.release(1)
        barrier.release_with_value(1)

    col_in_fifos = []
    col_out_fifos = []
    wt_fifos = []
    workers = []

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
        col_in_size = cores_this_col * core_input_size
        col_out_size = cores_this_col * core_output_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[core_input_size * i for i in range(cores_this_col)],
            obj_types=[patch_ty] * cores_this_col,
            depths=[input_depth] * cores_this_col,
            names=[f"input_{col}_{i}" for i in range(cores_this_col)],
        )

        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[core_output_size * i for i in range(cores_this_col)],
            obj_types=[output_tile_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"output_{col}_{i}" for i in range(cores_this_col)],
        )

        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"weights_{col}")
        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        for i in range(cores_this_col):
            global_core_idx = col * cores_per_col + i
            w = Worker(
                core_fn,
                [
                    in_splits[i].cons(),
                    wt_fifo.cons(),
                    out_joins[i].prod(),
                    kernel,
                    rtps[global_core_idx],
                    barriers[global_core_idx],
                ],
                stack_size=4096,
            )
            workers.append(w)

    rt = Runtime()
    with rt.sequence(host_input_ty, weight_ty, host_output_ty) as (I, W, O):
        rt.start(*workers)

        t_h, t_w = tile_h, tile_w
        ic_c, oc_c, s_c, p_c = ic, oc, stride_val, padding_val

        def set_rtps(*rtp_bufs):
            for rb in rtp_bufs:
                rb[0] = t_h
                rb[1] = t_w
                rb[2] = ic_c
                rb[3] = oc_c
                rb[4] = s_c
                rb[5] = p_c

        rt.inline_ops(set_rtps, rtps)
        for b in barriers:
            rt.set_barrier(b, 1)

        for wf in wt_fifos:
            rt.fill(wf.prod(), W)

        for col in range(n_cols):
            cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
            col_in_size = cores_this_col * core_input_size
            col_out_size = cores_this_col * core_output_size

            tap_in = TensorAccessPattern(
                (host_input_size,),
                offset=col * cores_per_col * core_input_size,
                sizes=[1, col_in_size],
                strides=[0, 1],
            )
            tap_out = TensorAccessPattern(
                (host_output_size,),
                offset=col * cores_per_col * core_output_size,
                sizes=[1, col_out_size],
                strides=[0, 1],
            )
            rt.fill(col_in_fifos[col].prod(), I, tap_in)
            rt.drain(col_out_fifos[col].cons(), O, tap_out, wait=True)

    return Program(dev, rt).resolve_program()


# ─────────────────────────────────────────────────────────────────────────────
# Conv3x3 reference (matches the C++ fast_sigmoid SiLU + BN-after-mac semantics)
# ─────────────────────────────────────────────────────────────────────────────


def conv3x3_bn_silu_ref(
    patch_hwc: np.ndarray,
    conv_w: np.ndarray,  # [oc, ic, 3, 3]
    bn_w: np.ndarray,
    bn_b: np.ndarray,
    tile_h: int,
    tile_w: int,
    ic: int,
    oc: int,
    stride: int,
    padding: int,
) -> np.ndarray:
    """Reference Conv3x3+BN+SiLU in float32 using the same fast_sigmoid form."""
    out = np.zeros((tile_h, tile_w, oc), dtype=np.float32)
    for oh in range(tile_h):
        for ow in range(tile_w):
            for j in range(oc):
                val = 0.0
                for i in range(ic):
                    for kh in range(3):
                        for kw in range(3):
                            ih = oh * stride + kh
                            iw = ow * stride + kw
                            val += float(patch_hwc[ih, iw, i]) * float(conv_w[j, i, kh, kw])
                val = float(bn_w[j]) * val + float(bn_b[j])
                ax = abs(val)
                out[oh, ow, j] = val * (0.5 + val / (2.0 + 2.0 * ax))
    return out


def _pack_conv3x3_weights(
    conv_w_f32: np.ndarray,  # [oc, ic, 3, 3]
    bn_w_f32: np.ndarray,
    bn_b_f32: np.ndarray,
) -> np.ndarray:
    """Pack weights into [OC/8, IC/8, 9, 8ic, 8oc] + bn_w(oc) + bn_b(oc).

    The C++ kernel loads B as a contiguous 64-element vector per (oc_blk,
    ic_blk, k) triple; rearranging mirrors what aie::mmul<4,8,8> wants.
    """
    oc, ic, kH, kW = conv_w_f32.shape
    assert oc % 8 == 0 and ic % 8 == 0 and kH == 3 and kW == 3

    # Convert via bf16 round-trip so the host reference and device see the
    # same rounded weight bits.
    conv_w_bf16_u16 = f32_to_bf16_u16(conv_w_f32).reshape(oc, ic, kH, kW)

    # Build [OC/8, IC/8, 9, 8ic, 8oc] from [oc, ic, 3, 3]
    oc_b = oc // 8
    ic_b = ic // 8
    # Step 1: re-index to (oc_b, 8oc, ic_b, 8ic, 3, 3)
    w = conv_w_bf16_u16.reshape(oc_b, 8, ic_b, 8, kH, kW)
    # Move 8ic, 8oc to inner, flatten kh,kw to k:
    # target axes order: (oc_b, ic_b, kh, kw, 8ic, 8oc)
    w = np.transpose(w, (0, 2, 4, 5, 3, 1))
    # Flatten (kh, kw) → k=9
    w = w.reshape(oc_b, ic_b, kH * kW, 8, 8)
    packed = w.reshape(-1).astype(np.uint16)

    bn_w_u = f32_to_bf16_u16(bn_w_f32)
    bn_b_u = f32_to_bf16_u16(bn_b_f32)
    return np.concatenate([packed, bn_w_u, bn_b_u])


def verify_conv3x3(
    work_dir: Path,
    *,
    n_cores: int = 4,
    tile_h: int = 4,
    tile_w: int = 4,
    ic: int = 64,
    oc: int = 16,
    stride: int = 1,
    padding: int = 1,
    patches_per_core: int = 1,
    verbose: bool = False,
) -> int:
    dev = NPU2()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    obj_dir = DEFAULT_BUILD_DIR
    if not (obj_dir / "conv3x3_fused_packed_bf16.o").exists():
        print(
            f"ERROR: missing {obj_dir/'conv3x3_fused_packed_bf16.o'}. "
            "Run build_kernels.py first."
        )
        return 1

    # Stage the kernel .o into work_dir so aiecc's relative link_with resolves.
    import shutil
    shutil.copyfile(
        obj_dir / "conv3x3_fused_packed_bf16.o",
        work_dir / "conv3x3_fused_packed_bf16.o",
    )

    print(
        f"[1/3] Building IRON module: conv3x3 n_cores={n_cores} "
        f"tile={tile_h}x{tile_w} ic={ic} oc={oc} stride={stride} pad={padding}"
    )
    module = build_conv3x3_module(
        dev,
        tile_h=tile_h,
        tile_w=tile_w,
        ic=ic,
        oc=oc,
        kernel_size=3,
        stride_val=stride,
        padding_val=padding,
        n_cores=n_cores,
        patches_per_core=patches_per_core,
        obj_dir=obj_dir,
    )
    mlir_path = work_dir / "conv3x3.mlir"
    with open(mlir_path, "w") as fh:
        print(module, file=fh)
    print(f"      -> {mlir_path}")

    print("[2/3] Compiling design with aiecc")
    insts_path = work_dir / "conv3x3_insts.bin"
    xclbin_path = work_dir / "conv3x3_final.xclbin"
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(insts_path),
        xclbin_path=str(xclbin_path),
        work_dir=str(work_dir),
        verbose=verbose,
    )
    print(f"      -> {xclbin_path}")

    print("[3/3] Running on NPU and validating results")

    patch_h = (tile_h - 1) * stride + 3
    patch_w = (tile_w - 1) * stride + 3
    patch_size_raw = patch_h * patch_w * ic
    patch_size = patch_size_raw + (patch_size_raw % 2)
    output_tile_size = tile_h * tile_w * oc

    rng = np.random.default_rng(42)
    patches_f32 = rng.standard_normal((n_cores, patch_h, patch_w, ic), dtype=np.float32)
    conv_w_f32 = rng.standard_normal((oc, ic, 3, 3), dtype=np.float32) * 0.1
    bn_w_f32 = np.ones(oc, dtype=np.float32)
    bn_b_f32 = np.zeros(oc, dtype=np.float32)

    # Round to bf16 for the reference so it matches the device precision.
    patches_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(patches_f32)).reshape(
        n_cores, patch_h, patch_w, ic
    )
    conv_w_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(conv_w_f32)).reshape(oc, ic, 3, 3)
    bn_w_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_w_f32))
    bn_b_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_b_f32))

    # Reference (per core)
    refs = [
        conv3x3_bn_silu_ref(
            patches_bf16[c],
            conv_w_bf16,
            bn_w_bf16,
            bn_b_bf16,
            tile_h, tile_w, ic, oc, stride, padding,
        )
        for c in range(n_cores)
    ]

    # Pack input per core (DMA-pad up to patch_size)
    input_parts = []
    for c in range(n_cores):
        p_u16 = f32_to_bf16_u16(patches_bf16[c].reshape(-1))
        if len(p_u16) < patch_size:
            p_u16 = np.pad(p_u16, (0, patch_size - len(p_u16)))
        input_parts.append(p_u16)
    input_concat = np.concatenate(input_parts)

    weights_u16 = _pack_conv3x3_weights(conv_w_bf16, bn_w_bf16, bn_b_bf16)

    kh = DefaultNPURuntime.load(NPUKernel(str(xclbin_path), str(insts_path)))
    in_buf = iron.tensor(input_concat, dtype=np.uint16)
    wt_buf = iron.tensor(weights_u16, dtype=np.uint16)
    out_buf = iron.zeros(n_cores * output_tile_size, dtype=np.uint16)

    print("      Running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f} ms)")

    out_u16 = np.array(out_buf.numpy()).copy()
    max_diff_global = 0.0
    for c in range(n_cores):
        start = c * output_tile_size
        tile_out_u16 = out_u16[start:start + output_tile_size]
        tile_out = bf16_u16_to_f32(tile_out_u16).reshape(tile_h, tile_w, oc)
        diff = np.abs(refs[c] - tile_out)
        md = float(diff.max())
        max_diff_global = max(max_diff_global, md)
        print(f"      Core {c}: max abs diff = {md:.6f}")

    # The reference is fp32 against the C++ fast_sigmoid SiLU + bf16-quantised
    # mmul accumulation. With ic=64 and the chained mul_elem_32 path, a
    # tolerance of ~0.5 (bf16 ULPs scaled by the accumulation depth) is the
    # documented bar from the original test_multicore_conv3x3.py.
    tol = 0.5
    if max_diff_global < tol:
        print(f"      Max abs diff = {max_diff_global:.6f} (tol={tol})")
        print("PASS!")
        return 0
    print(f"FAILED: max abs diff {max_diff_global:.6f} >= tol {tol}")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# gemm_kblocked verification (single-core, simpler IRON wrapper)
# ─────────────────────────────────────────────────────────────────────────────
#
# The kblocked kernel is a stateful chain: the host calls it n_k_blocks
# times, each time streaming a k_block-wide slice of weights. We exercise
# the single-core path here (which is enough to certify the kernel object);
# scaling to 4 cores follows the same conv3x3 pattern but with one extra
# loop over k_start. This is intentionally compact — the goal is to prove
# the .o is correct on hardware, not to ship the final layer wrapper.


def build_gemm_kblocked_module(
    dev,
    *,
    tile_m: int,
    full_ic: int,
    oc: int,
    k_block: int,
    obj_dir: Optional[Path] = None,
):
    """Single-core IRON wrapper exercising gemm_conv1x1_kblocked_bf16 once."""
    assert full_ic % k_block == 0, "full_ic must be a multiple of k_block"
    n_k_blocks = full_ic // k_block

    in_size = tile_m * full_ic              # input persists across k-blocks
    wt_chunk_size = k_block * oc + 2 * oc   # weight + bn(w,b)
    out_size = tile_m * oc

    in_ty = np.ndarray[(in_size,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(wt_chunk_size,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint16]]

    # Host buffers (concatenate all k chunks for the weight stream)
    host_in_ty = in_ty
    host_wt_ty = np.ndarray[(n_k_blocks * wt_chunk_size,), np.dtype[np.uint16]]
    host_out_ty = out_ty

    kernel = make_gemm_conv1x1_kblocked_bf16(in_ty, wt_ty, out_ty, build_dir=obj_dir)

    # Depth=1 across the board to fit per-core L1 (kb_wt is large: k_block*oc
    # bf16 + 2*oc BN params per chunk → ~34 KB for k_block=64, oc=256).
    of_in = ObjectFifo(in_ty, depth=1, name="kb_in")
    of_wt = ObjectFifo(wt_ty, depth=1, name="kb_wt")
    of_out = ObjectFifo(out_ty, depth=1, name="kb_out")

    # Unroll the k-loop in Python (n_k_blocks is small, e.g. 4) so the
    # per-call k_start scalar is a compile-time constant rather than an
    # `index`-typed range_ induction variable (the kernel signature wants i32).
    def core_fn(of_in, of_wt, of_out, kern):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for kbi in range(n_k_blocks):
            elem_wt = of_wt.acquire(1)
            kern(
                elem_in, elem_wt, elem_out,
                tile_m, full_ic, oc,
                kbi * k_block, k_block, n_k_blocks,
            )
            of_wt.release(1)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_wt.cons(), of_out.prod(), kernel],
                    stack_size=4096)

    rt = Runtime()
    with rt.sequence(host_in_ty, host_wt_ty, host_out_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_in.prod(), I)
        # The weight TAP streams n_k_blocks consecutive chunks of size wt_chunk_size.
        tap_w = TensorAccessPattern(
            (n_k_blocks * wt_chunk_size,),
            offset=0,
            sizes=[n_k_blocks, wt_chunk_size],
            strides=[wt_chunk_size, 1],
        )
        rt.fill(of_wt.prod(), W, tap_w)
        rt.drain(of_out.cons(), O, wait=True)

    return Program(dev, rt).resolve_program()


def _pack_gemm_kb_weight_chunk(
    wt_kb_f32: np.ndarray,  # [k_block, oc] slice for this chunk
    bn_w_f32: np.ndarray,
    bn_b_f32: np.ndarray,
    k_block: int,
    oc: int,
) -> np.ndarray:
    """Pack a single k-block weight chunk in [kb/8, oc/8, 8ic, 8oc] layout
    then append bn_w(oc) + bn_b(oc).

    Mirrors the C++ kernel's expected layout:
      wt_chunk + (kb * oc_blocks + oc_blk) * 64  →  one 8x8 bf16 block.
    """
    assert k_block % 8 == 0 and oc % 8 == 0
    kb = k_block // 8
    ob = oc // 8

    # [k_block, oc] → [kb, 8ic, ob, 8oc] → [kb, ob, 8ic, 8oc]
    w = wt_kb_f32.reshape(kb, 8, ob, 8).transpose(0, 2, 1, 3)
    w_u16 = f32_to_bf16_u16(w.reshape(-1).astype(np.float32))

    bn_w_u = f32_to_bf16_u16(bn_w_f32)
    bn_b_u = f32_to_bf16_u16(bn_b_f32)
    return np.concatenate([w_u16, bn_w_u, bn_b_u])


def gemm_kblocked_ref(
    in_f32: np.ndarray,        # [tile_m, full_ic]
    wt_f32: np.ndarray,        # [full_ic, oc]
    bn_w_f32: np.ndarray,
    bn_b_f32: np.ndarray,
) -> np.ndarray:
    """Reference: matmul + BN + fast_sigmoid SiLU."""
    mac = in_f32.astype(np.float32) @ wt_f32.astype(np.float32)
    bned = bn_w_f32[None, :] * mac + bn_b_f32[None, :]
    ax = np.abs(bned)
    return bned * (0.5 + bned / (2.0 + 2.0 * ax))


def verify_gemm_kblocked(
    work_dir: Path,
    *,
    tile_m: int = 20,
    full_ic: int = 256,
    oc: int = 256,
    k_block: int = 64,
    verbose: bool = False,
) -> int:
    dev = NPU2()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    obj_dir = DEFAULT_BUILD_DIR
    obj_name = "gemm_conv1x1_kblocked_bf16.o"
    if not (obj_dir / obj_name).exists():
        print(f"ERROR: missing {obj_dir/obj_name}. Run build_kernels.py first.")
        return 1

    import shutil
    shutil.copyfile(obj_dir / obj_name, work_dir / obj_name)

    print(
        f"[1/3] Building IRON module: gemm_kblocked tile_m={tile_m} "
        f"ic={full_ic} oc={oc} k_block={k_block}"
    )
    module = build_gemm_kblocked_module(
        dev, tile_m=tile_m, full_ic=full_ic, oc=oc, k_block=k_block, obj_dir=obj_dir,
    )
    mlir_path = work_dir / "gemm_kblocked.mlir"
    with open(mlir_path, "w") as fh:
        print(module, file=fh)
    print(f"      -> {mlir_path}")

    print("[2/3] Compiling design with aiecc")
    insts_path = work_dir / "gemm_kblocked_insts.bin"
    xclbin_path = work_dir / "gemm_kblocked_final.xclbin"
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(insts_path),
        xclbin_path=str(xclbin_path),
        work_dir=str(work_dir),
        verbose=verbose,
    )
    print(f"      -> {xclbin_path}")

    print("[3/3] Running on NPU and validating results")

    rng = np.random.default_rng(42)
    in_f32 = rng.standard_normal((tile_m, full_ic), dtype=np.float32) * 0.1
    wt_f32 = rng.standard_normal((full_ic, oc), dtype=np.float32) * 0.1
    bn_w_f32 = np.ones(oc, dtype=np.float32)
    bn_b_f32 = np.zeros(oc, dtype=np.float32)

    # Round to bf16 for the reference
    in_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(in_f32)).reshape(tile_m, full_ic)
    wt_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(wt_f32)).reshape(full_ic, oc)
    bn_w_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_w_f32))
    bn_b_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_b_f32))

    ref = gemm_kblocked_ref(in_bf16, wt_bf16, bn_w_bf16, bn_b_bf16)

    # Build the n_k_blocks weight stream
    n_k_blocks = full_ic // k_block
    chunks = []
    for kbi in range(n_k_blocks):
        k_start = kbi * k_block
        chunks.append(
            _pack_gemm_kb_weight_chunk(
                wt_bf16[k_start:k_start + k_block, :],
                bn_w_bf16, bn_b_bf16,
                k_block, oc,
            )
        )
    wt_stream = np.concatenate(chunks)

    in_u16 = f32_to_bf16_u16(in_bf16.reshape(-1))

    kh = DefaultNPURuntime.load(NPUKernel(str(xclbin_path), str(insts_path)))
    in_buf = iron.tensor(in_u16, dtype=np.uint16)
    wt_buf = iron.tensor(wt_stream, dtype=np.uint16)
    out_buf = iron.zeros(tile_m * oc, dtype=np.uint16)

    print("      Running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f} ms)")

    out_u16 = np.array(out_buf.numpy()).copy()
    out_f32 = bf16_u16_to_f32(out_u16).reshape(tile_m, oc)

    diff = np.abs(ref - out_f32)
    max_diff = float(diff.max())
    rel_diff = max_diff / (float(np.max(np.abs(ref))) + 1e-10)
    print(f"      Max abs diff = {max_diff:.6f}")
    print(f"      Max rel diff = {rel_diff:.6f}")

    tol = 0.5  # bf16-accumulation tolerance for ic=256 with fast_sigmoid SiLU
    if max_diff < tol:
        print("PASS!")
        return 0
    print(f"FAILED: max abs diff {max_diff:.6f} >= tol {tol}")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# gemm_conv1x1_fused_packed_bf16 verification (single-K-block GEMM + BN + SiLU)
# ─────────────────────────────────────────────────────────────────────────────


def build_gemm_fused_module(
    dev,
    *,
    tile_m: int,
    ic: int,
    oc: int,
    obj_dir: Optional[Path] = None,
):
    """Single-core IRON wrapper for gemm_conv1x1_fused_packed_bf16."""
    in_size = tile_m * ic
    wt_size = ic * oc + 2 * oc
    out_size = tile_m * oc

    in_ty = np.ndarray[(in_size,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(wt_size,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint16]]

    kernel = make_gemm_conv1x1_fused_packed_bf16(in_ty, wt_ty, out_ty, build_dir=obj_dir)

    of_in = ObjectFifo(in_ty, depth=1, name="gf_in")
    of_wt = ObjectFifo(wt_ty, depth=1, name="gf_wt")
    of_out = ObjectFifo(out_ty, depth=1, name="gf_out")

    # The kernel takes (input, weights, output, tile_h, tile_w, ic, oc,
    # stride_unused, padding_unused). For a single-spatial-row GEMM treat
    # the M dimension as tile_h=tile_m and tile_w=1 (so tile_h*tile_w=tile_m).
    def core_fn(of_in, of_wt, of_out, kern):
        e_in = of_in.acquire(1)
        e_wt = of_wt.acquire(1)
        e_out = of_out.acquire(1)
        kern(e_in, e_wt, e_out, tile_m, 1, ic, oc, 1, 0)
        of_in.release(1)
        of_wt.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_wt.cons(), of_out.prod(), kernel],
        stack_size=4096,
    )

    rt = Runtime()
    with rt.sequence(in_ty, wt_ty, out_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_in.prod(), I)
        rt.fill(of_wt.prod(), W)
        rt.drain(of_out.cons(), O, wait=True)

    return Program(dev, rt).resolve_program()


def _pack_gemm_fused_weights(
    wt_f32: np.ndarray,   # [ic, oc]
    bn_w_f32: np.ndarray,
    bn_b_f32: np.ndarray,
    ic: int,
    oc: int,
) -> np.ndarray:
    """Pack weights into [IC/8, OC/8, 8ic, 8oc] + bn_w(oc) + bn_b(oc).

    Matches gemm_conv1x1_fused_packed_bf16 layout in rep_elan_bf16.cc.
    """
    assert ic % 8 == 0 and oc % 8 == 0
    ib = ic // 8
    ob = oc // 8
    w = wt_f32.reshape(ib, 8, ob, 8).transpose(0, 2, 1, 3)
    w_u16 = f32_to_bf16_u16(w.reshape(-1).astype(np.float32))
    bn_w_u = f32_to_bf16_u16(bn_w_f32)
    bn_b_u = f32_to_bf16_u16(bn_b_f32)
    return np.concatenate([w_u16, bn_w_u, bn_b_u])


def verify_gemm_fused(
    work_dir: Path,
    *,
    tile_m: int = 88,
    ic: int = 64,
    oc: int = 64,
    verbose: bool = False,
) -> int:
    dev = NPU2()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    obj_dir = DEFAULT_BUILD_DIR
    obj_name = "gemm_conv1x1_fused_packed_bf16.o"
    if not (obj_dir / obj_name).exists():
        print(f"ERROR: missing {obj_dir/obj_name}. Run build_kernels.py first.")
        return 1

    import shutil
    shutil.copyfile(obj_dir / obj_name, work_dir / obj_name)

    print(
        f"[1/3] Building IRON module: gemm_fused tile_m={tile_m} ic={ic} oc={oc}"
    )
    module = build_gemm_fused_module(
        dev, tile_m=tile_m, ic=ic, oc=oc, obj_dir=obj_dir,
    )
    mlir_path = work_dir / "gemm_fused.mlir"
    with open(mlir_path, "w") as fh:
        print(module, file=fh)
    print(f"      -> {mlir_path}")

    print("[2/3] Compiling design with aiecc")
    insts_path = work_dir / "gemm_fused_insts.bin"
    xclbin_path = work_dir / "gemm_fused_final.xclbin"
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(insts_path),
        xclbin_path=str(xclbin_path),
        work_dir=str(work_dir),
        verbose=verbose,
    )
    print(f"      -> {xclbin_path}")

    print("[3/3] Running on NPU and validating results")

    rng = np.random.default_rng(42)
    in_f32 = rng.standard_normal((tile_m, ic), dtype=np.float32) * 0.1
    wt_f32 = rng.standard_normal((ic, oc), dtype=np.float32) * 0.1
    bn_w_f32 = np.ones(oc, dtype=np.float32)
    bn_b_f32 = np.zeros(oc, dtype=np.float32)

    # Round to bf16
    in_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(in_f32)).reshape(tile_m, ic)
    wt_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(wt_f32)).reshape(ic, oc)
    bn_w_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_w_f32))
    bn_b_bf16 = bf16_u16_to_f32(f32_to_bf16_u16(bn_b_f32))

    # Reference: matmul + BN + fast_sigmoid SiLU
    mac = in_bf16.astype(np.float32) @ wt_bf16.astype(np.float32)
    bned = bn_w_bf16[None, :] * mac + bn_b_bf16[None, :]
    ax = np.abs(bned)
    ref = bned * (0.5 + bned / (2.0 + 2.0 * ax))

    in_u16 = f32_to_bf16_u16(in_bf16.reshape(-1))
    wt_packed = _pack_gemm_fused_weights(wt_bf16, bn_w_bf16, bn_b_bf16, ic, oc)

    kh = DefaultNPURuntime.load(NPUKernel(str(xclbin_path), str(insts_path)))
    in_buf = iron.tensor(in_u16, dtype=np.uint16)
    wt_buf = iron.tensor(wt_packed, dtype=np.uint16)
    out_buf = iron.zeros(tile_m * oc, dtype=np.uint16)

    print("      Running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f} ms)")

    out_u16 = np.array(out_buf.numpy()).copy()
    out_f32 = bf16_u16_to_f32(out_u16).reshape(tile_m, oc)

    diff = np.abs(ref - out_f32)
    max_diff = float(diff.max())
    rel_diff = max_diff / (float(np.max(np.abs(ref))) + 1e-10)
    print(f"      Max abs diff = {max_diff:.6f}")
    print(f"      Max rel diff = {rel_diff:.6f}")

    tol = 0.2  # bf16-accumulation tolerance for ic=64 with fast_sigmoid SiLU
    if max_diff < tol:
        print("PASS!")
        return 0
    print(f"FAILED: max abs diff {max_diff:.6f} >= tol {tol}")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Validate rep_elan_bf16_pythoc kernels on NPU."
    )
    parser.add_argument(
        "--kernel",
        choices=("conv3x3", "gemm_kblocked", "gemm_fused", "all"),
        default="all",
        help="Which xclbin to build & validate.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=HERE / "verify_build",
        help="Output directory for MLIR/xclbin/insts.",
    )
    parser.add_argument("--verbose", action="store_true")
    # conv3x3 knobs
    parser.add_argument("--n-cores", type=int, default=4)
    parser.add_argument("--tile-h", type=int, default=4)
    parser.add_argument("--tile-w", type=int, default=4)
    parser.add_argument("--ic", type=int, default=64)
    parser.add_argument("--oc", type=int, default=16)
    # gemm_kblocked knobs
    parser.add_argument("--gemm-tile-m", type=int, default=20)
    parser.add_argument("--gemm-ic", type=int, default=256)
    parser.add_argument("--gemm-oc", type=int, default=256)
    parser.add_argument("--gemm-k-block", type=int, default=64)
    # gemm_fused knobs (single-K-block GEMM with full BN+SiLU tail)
    parser.add_argument("--gemmf-tile-m", type=int, default=88)
    parser.add_argument("--gemmf-ic", type=int, default=64)
    parser.add_argument("--gemmf-oc", type=int, default=64)
    args = parser.parse_args()

    if args.kernel in ("conv3x3", "all"):
        rc = verify_conv3x3(
            args.work_dir / "conv3x3",
            n_cores=args.n_cores,
            tile_h=args.tile_h,
            tile_w=args.tile_w,
            ic=args.ic,
            oc=args.oc,
            verbose=args.verbose,
        )
        if rc != 0:
            return rc

    if args.kernel in ("gemm_fused", "all"):
        rc = verify_gemm_fused(
            args.work_dir / "gemm_fused",
            tile_m=args.gemmf_tile_m,
            ic=args.gemmf_ic,
            oc=args.gemmf_oc,
            verbose=args.verbose,
        )
        if rc != 0:
            return rc

    if args.kernel in ("gemm_kblocked", "all"):
        rc = verify_gemm_kblocked(
            args.work_dir / "gemm_kblocked",
            tile_m=args.gemm_tile_m,
            full_ic=args.gemm_ic,
            oc=args.gemm_oc,
            k_block=args.gemm_k_block,
            verbose=args.verbose,
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
