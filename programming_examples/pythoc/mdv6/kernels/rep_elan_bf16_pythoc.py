#!/usr/bin/env python3
# rep_elan_bf16_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

"""MDV6 unified rep_elan kernels — true PythoC ``@aie_kernel`` ports.

This file replaces the previous "Pattern A" thin wrapper (which compiled
the C++ ``rep_elan_bf16.cc`` into one .o per kernel) with three real
PythoC ``@aie_kernel`` functions plus the already-PythoC scalar residual
+SiLU.

Kernels
-------
1. ``conv3x3_fused_packed_bf16``        — 3x3 conv + BN + SiLU
2. ``gemm_conv1x1_fused_packed_bf16``   — 1x1 conv as GEMM + BN + SiLU
3. ``gemm_conv1x1_kblocked_bf16``       — K-blocked GEMM (partial-accum chain)
4. ``residual_add_silu_bf16``           — scalar residual add + SiLU

The three matmul kernels share a single inline bf16 ``mmul<4,8,8>``
emulation chain — 1 ``mul_elem_32`` + 7 ``mac_elem_32`` calls plus the
T16-shuffle / extract-broadcast plumbing that the AIE API
``mul_4x8_8x8_bf16``/``mac_4x8_8x8_bf16`` helpers use on the C++ side
(see ``include/aie_api/detail/aie2p/emulated_mmul_intrinsics.hpp``).

Auto-vectorizer escape
----------------------
The BN+SiLU per-pixel scalar bf16 tails get auto-vectorized by Peano
``opt -O2`` into ``<32 x bfloat> fadd`` which AIE2P llc cannot legalize.
We patch ``subprocess.run`` (the same fix used in
``mdv6/prototypes/repncsp/repncsp_pythoc.py``) to inject ``-vectorize-loops=false
-vectorize-slp=false`` into the opt invocation. The fix is process-
global; importing this module installs it once.

Caller surface
--------------
The IRON wrappers use the ``make_<kernel>_*`` factory helpers below.
Each factory inline-compiles the corresponding ``@aie_kernel`` into a
fresh ``.o``. The factories accept buffer numpy-dtype/shape descriptors
and an optional ``build_dir``; ``build_kernels.py`` calls them with a
canonical set of types to produce one ``.o`` per kernel under
``build/<kernel>.o`` so IRON wrappers that pass ``Kernel(name, "<name>.o",
[...])`` continue to find their object files.
"""

from __future__ import annotations

import shutil
import subprocess as _subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from aie.iron.pythoc import aie_kernel, PythocKernel

from pythoc import ptr, i32, bf16, f32, void
from pythoc.aie import (
    aie_vector,
    load_v,
    store_v,
    vector_cast,
    vshuffle,
    vector_extract,
    vector_insert,
    concat,
    extract_elem,
    extract_v4bfloat16_broadcast_to_v32bfloat16,
    extract_v8bfloat16_broadcast_to_v32bfloat16,
    T16_4x8,
    T16_8x4,
)
from pythoc.aie.mmul import mmul_bf16, mmul_bf16_mac, acc_to_bf16
from pythoc.aie.profiling import event0, event1

# Lazy-loaded intrinsic wrapper from the registry — converts a <32 x bf16>
# vector to a <32 x f32> accumulator (ACC1024). Counterpart of acc_to_bf16
# which goes the other way. Used by the K-blocked GEMM to reload partial
# sums (stored as bf16 in the output buffer between K-block calls).
from pythoc.aie import v32bf16_to_v32accfloat as _bf16_to_acc


# ──────────────────────────────────────────────────────────────────────
# subprocess.run patch (auto-vectorizer disable for opt invocations)
# ──────────────────────────────────────────────────────────────────────
#
# Same fix as repncsp_pythoc.py. Without it, the bf16 BN+SiLU per-pixel
# tail in each kernel here gets auto-vectorized into <32 x bfloat> fadd
# which AIE2P llc (GISel legalizer) cannot lower.

_orig_subprocess_run = _subprocess.run


def _patched_subprocess_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args")
    if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and "opt" in str(cmd[0]):
        new_cmd = list(cmd)
        new_cmd[1:1] = [
            "-vectorize-loops=false",
            "-vectorize-slp=false",
        ]
        if args:
            args = (new_cmd,) + args[1:]
        else:
            kwargs["args"] = new_cmd
    return _orig_subprocess_run(*args, **kwargs)


if getattr(_subprocess.run, "__name__", "") != "_patched_subprocess_run":
    _subprocess.run = _patched_subprocess_run


# ──────────────────────────────────────────────────────────────────────
# Default build directory + kernel name list
# ──────────────────────────────────────────────────────────────────────

_KERNEL_NAMES = (
    "conv3x3_fused_packed_bf16",
    "gemm_conv1x1_fused_packed_bf16",
    "gemm_conv1x1_kblocked_bf16",
    "residual_add_silu_bf16",
)

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "build"


def _obj_path(kernel_name: str, build_dir: Optional[Path] = None) -> Path:
    build_dir = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    return build_dir / f"{kernel_name}.o"


# ──────────────────────────────────────────────────────────────────────
# extra_globals passed to PythocKernel for the 3 mmul-based kernels.
# These names aren't in compiler.py's default global set; the
# auto-broadcast/shuffle/cast/concat ops, the mmul_bf16 family, and the
# new T16 / extract_v*_broadcast helpers all need to be injected here.
# ──────────────────────────────────────────────────────────────────────

KERNEL_EXTRA_GLOBALS = {
    "vector_cast": vector_cast,
    "vshuffle": vshuffle,
    "vector_extract": vector_extract,
    "vector_insert": vector_insert,
    # ``concat`` IS already in default globals but harmless to re-inject.
    "concat": concat,
    "mmul_bf16": mmul_bf16,
    "mmul_bf16_mac": mmul_bf16_mac,
    "acc_to_bf16": acc_to_bf16,
    "_bf16_to_acc": _bf16_to_acc,
    # extract_v*_broadcast_to_v32bfloat16 and the T16_* constants are
    # already in default globals (see compiler.py around line 359-366).
}


# ──────────────────────────────────────────────────────────────────────
# Scalar PythoC kernel — residual_add_silu_bf16 (already pure PythoC)
# ──────────────────────────────────────────────────────────────────────


@aie_kernel
def residual_add_silu_bf16(
    current: ptr[bf16, True],
    residual: ptr[bf16, True],
    out: ptr[bf16, True],
    tile_m: i32,
    oc: i32,
) -> void:
    """out[i] = silu(current[i] + residual[i]) for i in [0, tile_m*oc)."""
    event0()
    n: i32 = tile_m * oc
    i: i32 = 0
    while i < n:
        x: f32 = f32(current[i]) + f32(residual[i])
        ax: f32 = x
        if x < 0.0:
            ax = -x
        denom: f32 = 2.0 + 2.0 * ax
        sig: f32 = 0.5 + x / denom
        out[i] = bf16(x * sig)
        i = i + 1
    event1()


# ──────────────────────────────────────────────────────────────────────
# conv3x3_fused_packed_bf16 — 3x3 conv + BN + SiLU
# ──────────────────────────────────────────────────────────────────────
#
# Mirrors the C++ kernel in ``rep_elan_bf16.cc`` line-for-line. The
# ``aie::mmul<4,8,8,bfloat16,bfloat16>`` ::mac chain is expanded inline
# using ``mmul_bf16`` / ``mmul_bf16_mac`` plus the T16-shuffle /
# extract-broadcast plumbing the AIE API uses on the C++ side
# (see ``emulated_mmul_intrinsics.hpp::mac_4x8_8x8_bf16``).
#
# Weight layout (host-packed): [OC/8, IC/8, 9, 8ic, 8oc]
# Followed by bn_w(oc) and bn_b(oc) at the end of the buffer.


@aie_kernel
def conv3x3_fused_packed_bf16(
    input: ptr[bf16, True],
    packed_weights: ptr[bf16, True],
    output: ptr[bf16, True],
    tile_h: i32,
    tile_w: i32,
    ic: i32,
    oc: i32,
    stride: i32,
    padding: i32,
) -> void:
    event0()

    patch_w: i32 = (tile_w - 1) * stride + 3
    spatial_out: i32 = tile_h * tile_w
    wt_size: i32 = oc * ic * 9
    bn_w_ptr: ptr[bf16] = packed_weights + wt_size
    bn_b_ptr: ptr[bf16] = bn_w_ptr + oc

    ic_blocks: i32 = ic // 8
    oc_blocks: i32 = oc // 8

    oc_blk: i32 = 0
    while oc_blk < oc_blocks:
        wt_base_off: i32 = oc_blk * ic_blocks * 9 * 64

        sp: i32 = 0
        while sp < spatial_out:
            wt_off: i32 = wt_base_off

            # First iteration: full mmul<4,8,8> with implicit zero acc.
            A0: aie_vector[bf16, 32] = _build_a32_3x3(
                input, sp, tile_w, stride, patch_w, ic, 0, 0, 0
            )
            B0: aie_vector[bf16, 64] = load_v(packed_weights + wt_off, 64)
            acc: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, B0)
            wt_off = wt_off + 64

            # Remaining (ic_blocks * 9) - 1 mac chain.
            kk: i32 = 1
            total_k: i32 = ic_blocks * 9
            while kk < total_k:
                ic_blk: i32 = kk // 9
                kk_in_blk: i32 = kk - ic_blk * 9
                kh: i32 = kk_in_blk // 3
                kw: i32 = kk_in_blk - kh * 3

                A: aie_vector[bf16, 32] = _build_a32_3x3(
                    input, sp, tile_w, stride, patch_w, ic, ic_blk, kh, kw
                )
                B: aie_vector[bf16, 64] = load_v(packed_weights + wt_off, 64)
                acc = _mac_4x8x8_bf16(A, B, acc)
                wt_off = wt_off + 64
                kk = kk + 1

            # Convert acc to bf16 and apply BN + SiLU per row.
            result32: aie_vector[bf16, 32] = acc_to_bf16(acc)
            _store_bn_silu_4x8_rows(
                result32, output, bn_w_ptr, bn_b_ptr, sp, spatial_out, oc, oc_blk
            )

            sp = sp + 4

        oc_blk = oc_blk + 1

    event1()


# ──────────────────────────────────────────────────────────────────────
# gemm_conv1x1_fused_packed_bf16 — 1×1 conv as GEMM + BN + SiLU
# ──────────────────────────────────────────────────────────────────────
#
# Weight layout: [IC/8, OC/8, 8ic, 8oc] + bn_w(oc) + bn_b(oc).


@aie_kernel
def gemm_conv1x1_fused_packed_bf16(
    input: ptr[bf16, True],
    packed_weights: ptr[bf16, True],
    output: ptr[bf16, True],
    tile_h: i32,
    tile_w: i32,
    ic: i32,
    oc: i32,
    stride_unused: i32,
    padding_unused: i32,
) -> void:
    event0()

    tile_m: i32 = tile_h * tile_w
    wt_size: i32 = oc * ic
    bn_w_ptr: ptr[bf16] = packed_weights + wt_size
    bn_b_ptr: ptr[bf16] = bn_w_ptr + oc

    ic_blocks: i32 = ic // 8
    oc_blocks: i32 = oc // 8

    oc_blk: i32 = 0
    while oc_blk < oc_blocks:
        sp: i32 = 0
        while sp < tile_m:
            # First mmul (ic_blk=0): use full 4×8×8 mul (zero-acc seed).
            A0: aie_vector[bf16, 32] = _build_a32_1x1(input, sp, ic, 0)
            B0: aie_vector[bf16, 64] = load_v(
                packed_weights + (0 * oc_blocks + oc_blk) * 64, 64
            )
            acc: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, B0)

            ic_blk: i32 = 1
            while ic_blk < ic_blocks:
                A: aie_vector[bf16, 32] = _build_a32_1x1(input, sp, ic, ic_blk)
                B: aie_vector[bf16, 64] = load_v(
                    packed_weights + (ic_blk * oc_blocks + oc_blk) * 64, 64
                )
                acc = _mac_4x8x8_bf16(A, B, acc)
                ic_blk = ic_blk + 1

            result32: aie_vector[bf16, 32] = acc_to_bf16(acc)
            _store_bn_silu_4x8_rows(
                result32, output, bn_w_ptr, bn_b_ptr, sp, tile_m, oc, oc_blk
            )
            sp = sp + 4

        oc_blk = oc_blk + 1

    event1()


# ──────────────────────────────────────────────────────────────────────
# gemm_conv1x1_kblocked_bf16 — K-blocked GEMM (partial-accumulator chain)
# ──────────────────────────────────────────────────────────────────────
#
# Weight chunk layout: [k_block/8, oc/8, 8ic, 8oc] + bn_w(oc) + bn_b(oc)
# BN params are written into every chunk but only consumed on the last
# K-block (when ``k_start + k_block >= full_ic``).


@aie_kernel
def gemm_conv1x1_kblocked_bf16(
    input: ptr[bf16, True],
    wt_chunk: ptr[bf16, True],
    output: ptr[bf16, True],
    tile_m: i32,
    full_ic: i32,
    oc: i32,
    k_start: i32,
    k_block: i32,
    n_k_blocks: i32,
) -> void:
    event0()

    is_first: i32 = 0
    if k_start == 0:
        is_first = 1
    is_last: i32 = 0
    if k_start + k_block >= full_ic:
        is_last = 1

    kb_blocks: i32 = k_block // 8
    oc_blocks: i32 = oc // 8
    wt_size: i32 = k_block * oc

    bn_w_ptr: ptr[bf16] = wt_chunk + wt_size
    bn_b_ptr: ptr[bf16] = bn_w_ptr + oc

    oc_blk: i32 = 0
    while oc_blk < oc_blocks:
        sp: i32 = 0
        while sp < tile_m:
            # Seed accumulator: zero (first K-block) or partial sums from output.
            acc: aie_vector[f32, 32]
            kb_start: i32 = 0
            if is_first == 1:
                # First call: seed via the first kb iteration's full mmul.
                A0: aie_vector[bf16, 32] = _build_a32_kblocked(
                    input, sp, full_ic, k_start, 0
                )
                B0: aie_vector[bf16, 64] = load_v(
                    wt_chunk + (0 * oc_blocks + oc_blk) * 64, 64
                )
                acc = _mul_4x8x8_bf16(A0, B0)
                kb_start = 1
            else:
                # Non-first call: load the partial sums (bf16) from output
                # and lift to f32 accumulator via the AIE2P
                # v32bf16→v32accfloat conversion intrinsic (counterpart of
                # acc_to_bf16 which goes f32→bf16). This is the PythoC
                # equivalent of the C++ ``MMUL(partial)`` constructor.
                p0: aie_vector[bf16, 8] = load_v(output + (sp + 0) * oc + oc_blk * 8, 8)
                p1: aie_vector[bf16, 8] = load_v(output + (sp + 1) * oc + oc_blk * 8, 8)
                p2: aie_vector[bf16, 8] = load_v(output + (sp + 2) * oc + oc_blk * 8, 8)
                p3: aie_vector[bf16, 8] = load_v(output + (sp + 3) * oc + oc_blk * 8, 8)
                p01: aie_vector[bf16, 16] = concat(p0, p1)
                p23: aie_vector[bf16, 16] = concat(p2, p3)
                partial32: aie_vector[bf16, 32] = concat(p01, p23)
                acc = _bf16_to_acc(partial32)
                kb_start = 0

            kb: i32 = kb_start
            while kb < kb_blocks:
                A: aie_vector[bf16, 32] = _build_a32_kblocked(
                    input, sp, full_ic, k_start, kb
                )
                B: aie_vector[bf16, 64] = load_v(
                    wt_chunk + (kb * oc_blocks + oc_blk) * 64, 64
                )
                acc = _mac_4x8x8_bf16(A, B, acc)
                kb = kb + 1

            result32: aie_vector[bf16, 32] = acc_to_bf16(acc)

            if is_last == 1:
                _store_bn_silu_4x8_rows(
                    result32, output, bn_w_ptr, bn_b_ptr, sp, tile_m, oc, oc_blk
                )
            else:
                # Write partial sums (bf16) back to output (no BN, no SiLU).
                r0: aie_vector[bf16, 8] = vector_extract(result32, 0, 8)
                r1: aie_vector[bf16, 8] = vector_extract(result32, 8, 8)
                r2: aie_vector[bf16, 8] = vector_extract(result32, 16, 8)
                r3: aie_vector[bf16, 8] = vector_extract(result32, 24, 8)
                store_v(output + (sp + 0) * oc + oc_blk * 8, r0)
                store_v(output + (sp + 1) * oc + oc_blk * 8, r1)
                store_v(output + (sp + 2) * oc + oc_blk * 8, r2)
                store_v(output + (sp + 3) * oc + oc_blk * 8, r3)

            sp = sp + 4

        oc_blk = oc_blk + 1

    event1()


# ──────────────────────────────────────────────────────────────────────
# Helper @aie_kernels — these are compiled alongside the main kernels via
# the ``helpers=[...]`` parameter on ``PythocKernel``.
# ──────────────────────────────────────────────────────────────────────


@aie_kernel
def _build_a32_3x3(
    input: ptr[bf16, True],
    sp: i32,
    tile_w: i32,
    stride: i32,
    patch_w: i32,
    ic: i32,
    ic_blk: i32,
    kh: i32,
    kw: i32,
) -> aie_vector[bf16, 32]:
    """Load 4 contiguous (sp+0..sp+3) 8-wide patches and concat into v32 bf16.

    Mirrors the C++ ``A.insert(p, load_v<8>(...))`` loop for the 3x3 conv.
    """
    oh0: i32 = (sp + 0) // tile_w
    ow0: i32 = (sp + 0) - oh0 * tile_w
    ih0: i32 = oh0 * stride + kh
    iw0: i32 = ow0 * stride + kw
    a0: aie_vector[bf16, 8] = load_v(
        input + (ih0 * patch_w + iw0) * ic + ic_blk * 8, 8
    )

    oh1: i32 = (sp + 1) // tile_w
    ow1: i32 = (sp + 1) - oh1 * tile_w
    ih1: i32 = oh1 * stride + kh
    iw1: i32 = ow1 * stride + kw
    a1: aie_vector[bf16, 8] = load_v(
        input + (ih1 * patch_w + iw1) * ic + ic_blk * 8, 8
    )

    oh2: i32 = (sp + 2) // tile_w
    ow2: i32 = (sp + 2) - oh2 * tile_w
    ih2: i32 = oh2 * stride + kh
    iw2: i32 = ow2 * stride + kw
    a2: aie_vector[bf16, 8] = load_v(
        input + (ih2 * patch_w + iw2) * ic + ic_blk * 8, 8
    )

    oh3: i32 = (sp + 3) // tile_w
    ow3: i32 = (sp + 3) - oh3 * tile_w
    ih3: i32 = oh3 * stride + kh
    iw3: i32 = ow3 * stride + kw
    a3: aie_vector[bf16, 8] = load_v(
        input + (ih3 * patch_w + iw3) * ic + ic_blk * 8, 8
    )

    a01: aie_vector[bf16, 16] = concat(a0, a1)
    a23: aie_vector[bf16, 16] = concat(a2, a3)
    return concat(a01, a23)


@aie_kernel
def _build_a32_1x1(
    input: ptr[bf16, True],
    sp: i32,
    ic: i32,
    ic_blk: i32,
) -> aie_vector[bf16, 32]:
    """1x1 GEMM-style A loader (4 spatial rows × 8 IC, contiguous)."""
    a0: aie_vector[bf16, 8] = load_v(input + (sp + 0) * ic + ic_blk * 8, 8)
    a1: aie_vector[bf16, 8] = load_v(input + (sp + 1) * ic + ic_blk * 8, 8)
    a2: aie_vector[bf16, 8] = load_v(input + (sp + 2) * ic + ic_blk * 8, 8)
    a3: aie_vector[bf16, 8] = load_v(input + (sp + 3) * ic + ic_blk * 8, 8)
    a01: aie_vector[bf16, 16] = concat(a0, a1)
    a23: aie_vector[bf16, 16] = concat(a2, a3)
    return concat(a01, a23)


@aie_kernel
def _build_a32_kblocked(
    input: ptr[bf16, True],
    sp: i32,
    full_ic: i32,
    k_start: i32,
    kb: i32,
) -> aie_vector[bf16, 32]:
    """K-blocked GEMM A loader: 4 rows × 8 IC at offset (k_start + kb*8)."""
    off_in: i32 = k_start + kb * 8
    a0: aie_vector[bf16, 8] = load_v(input + (sp + 0) * full_ic + off_in, 8)
    a1: aie_vector[bf16, 8] = load_v(input + (sp + 1) * full_ic + off_in, 8)
    a2: aie_vector[bf16, 8] = load_v(input + (sp + 2) * full_ic + off_in, 8)
    a3: aie_vector[bf16, 8] = load_v(input + (sp + 3) * full_ic + off_in, 8)
    a01: aie_vector[bf16, 16] = concat(a0, a1)
    a23: aie_vector[bf16, 16] = concat(a2, a3)
    return concat(a01, a23)


@aie_kernel
def _mul_4x8x8_bf16(
    A: aie_vector[bf16, 32],
    B: aie_vector[bf16, 64],
) -> aie_vector[f32, 32]:
    """``mmul<4,8,8>`` zero-acc seed: 1 mul_elem_32 + 7 mac_elem_32 calls.

    Mirrors ``mul_4x8_8x8_bf16`` from
    ``include/aie_api/detail/aie2p/emulated_mmul_intrinsics.hpp``.
    """
    # A: <32 x bf16> ≡ 4 rows × 8 cols (row-major). Transpose to 8 cols × 4 rows.
    a_i32: aie_vector[i32, 16] = vector_cast(A, i32, 16)
    at_i32: aie_vector[i32, 16] = vshuffle(a_i32, a_i32, T16_4x8)
    at: aie_vector[bf16, 32] = vector_cast(at_i32, bf16, 32)

    x0_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 0)
    x0_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x0_raw, i32, 16), vector_cast(x0_raw, i32, 16), T16_8x4
    )
    x0: aie_vector[bf16, 32] = vector_cast(x0_si, bf16, 32)

    x1_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 1)
    x1_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x1_raw, i32, 16), vector_cast(x1_raw, i32, 16), T16_8x4
    )
    x1: aie_vector[bf16, 32] = vector_cast(x1_si, bf16, 32)

    x2_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 2)
    x2_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x2_raw, i32, 16), vector_cast(x2_raw, i32, 16), T16_8x4
    )
    x2: aie_vector[bf16, 32] = vector_cast(x2_si, bf16, 32)

    x3_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 3)
    x3_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x3_raw, i32, 16), vector_cast(x3_raw, i32, 16), T16_8x4
    )
    x3: aie_vector[bf16, 32] = vector_cast(x3_si, bf16, 32)

    x4_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 4)
    x4_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x4_raw, i32, 16), vector_cast(x4_raw, i32, 16), T16_8x4
    )
    x4: aie_vector[bf16, 32] = vector_cast(x4_si, bf16, 32)

    x5_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 5)
    x5_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x5_raw, i32, 16), vector_cast(x5_raw, i32, 16), T16_8x4
    )
    x5: aie_vector[bf16, 32] = vector_cast(x5_si, bf16, 32)

    x6_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 6)
    x6_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x6_raw, i32, 16), vector_cast(x6_raw, i32, 16), T16_8x4
    )
    x6: aie_vector[bf16, 32] = vector_cast(x6_si, bf16, 32)

    x7_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 7)
    x7_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x7_raw, i32, 16), vector_cast(x7_raw, i32, 16), T16_8x4
    )
    x7: aie_vector[bf16, 32] = vector_cast(x7_si, bf16, 32)

    b_lo: aie_vector[bf16, 32] = vector_extract(B, 0, 32)
    b_hi: aie_vector[bf16, 32] = vector_extract(B, 32, 32)

    y0: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 0)
    y1: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 1)
    y2: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 2)
    y3: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 3)
    y4: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 0)
    y5: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 1)
    y6: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 2)
    y7: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 3)

    acc: aie_vector[f32, 32] = mmul_bf16(x0, y0)
    acc = mmul_bf16_mac(x1, y1, acc)
    acc = mmul_bf16_mac(x2, y2, acc)
    acc = mmul_bf16_mac(x3, y3, acc)
    acc = mmul_bf16_mac(x4, y4, acc)
    acc = mmul_bf16_mac(x5, y5, acc)
    acc = mmul_bf16_mac(x6, y6, acc)
    acc = mmul_bf16_mac(x7, y7, acc)
    return acc


@aie_kernel
def _mac_4x8x8_bf16(
    A: aie_vector[bf16, 32],
    B: aie_vector[bf16, 64],
    acc: aie_vector[f32, 32],
) -> aie_vector[f32, 32]:
    """``mmul<4,8,8>::mac`` step (8 chained mac_elem_32 calls).

    Mirrors ``mac_4x8_8x8_bf16`` from
    ``include/aie_api/detail/aie2p/emulated_mmul_intrinsics.hpp``.
    """
    # A: <32 x bf16> ≡ 4 rows × 8 cols (row-major). Transpose to 8 cols × 4 rows.
    a_i32: aie_vector[i32, 16] = vector_cast(A, i32, 16)
    at_i32: aie_vector[i32, 16] = vshuffle(a_i32, a_i32, T16_4x8)
    at: aie_vector[bf16, 32] = vector_cast(at_i32, bf16, 32)

    # Build 8 row-broadcasts of A (each row replicated to all 4 cols).
    x0_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 0)
    x0_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x0_raw, i32, 16), vector_cast(x0_raw, i32, 16), T16_8x4
    )
    x0: aie_vector[bf16, 32] = vector_cast(x0_si, bf16, 32)

    x1_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 1)
    x1_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x1_raw, i32, 16), vector_cast(x1_raw, i32, 16), T16_8x4
    )
    x1: aie_vector[bf16, 32] = vector_cast(x1_si, bf16, 32)

    x2_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 2)
    x2_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x2_raw, i32, 16), vector_cast(x2_raw, i32, 16), T16_8x4
    )
    x2: aie_vector[bf16, 32] = vector_cast(x2_si, bf16, 32)

    x3_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 3)
    x3_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x3_raw, i32, 16), vector_cast(x3_raw, i32, 16), T16_8x4
    )
    x3: aie_vector[bf16, 32] = vector_cast(x3_si, bf16, 32)

    x4_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 4)
    x4_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x4_raw, i32, 16), vector_cast(x4_raw, i32, 16), T16_8x4
    )
    x4: aie_vector[bf16, 32] = vector_cast(x4_si, bf16, 32)

    x5_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 5)
    x5_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x5_raw, i32, 16), vector_cast(x5_raw, i32, 16), T16_8x4
    )
    x5: aie_vector[bf16, 32] = vector_cast(x5_si, bf16, 32)

    x6_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 6)
    x6_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x6_raw, i32, 16), vector_cast(x6_raw, i32, 16), T16_8x4
    )
    x6: aie_vector[bf16, 32] = vector_cast(x6_si, bf16, 32)

    x7_raw: aie_vector[bf16, 32] = extract_v4bfloat16_broadcast_to_v32bfloat16(at, 7)
    x7_si: aie_vector[i32, 16] = vshuffle(
        vector_cast(x7_raw, i32, 16), vector_cast(x7_raw, i32, 16), T16_8x4
    )
    x7: aie_vector[bf16, 32] = vector_cast(x7_si, bf16, 32)

    # Split B into the two halves the C++ helper uses.
    b_lo: aie_vector[bf16, 32] = vector_extract(B, 0, 32)
    b_hi: aie_vector[bf16, 32] = vector_extract(B, 32, 32)

    # B-side: 8 column-broadcasts (4 from lo, 4 from hi).
    y0: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 0)
    y1: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 1)
    y2: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 2)
    y3: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_lo, 3)
    y4: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 0)
    y5: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 1)
    y6: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 2)
    y7: aie_vector[bf16, 32] = extract_v8bfloat16_broadcast_to_v32bfloat16(b_hi, 3)

    # 8-step chained mac.
    acc = mmul_bf16_mac(x0, y0, acc)
    acc = mmul_bf16_mac(x1, y1, acc)
    acc = mmul_bf16_mac(x2, y2, acc)
    acc = mmul_bf16_mac(x3, y3, acc)
    acc = mmul_bf16_mac(x4, y4, acc)
    acc = mmul_bf16_mac(x5, y5, acc)
    acc = mmul_bf16_mac(x6, y6, acc)
    acc = mmul_bf16_mac(x7, y7, acc)
    return acc


@aie_kernel
def _store_bn_silu_4x8_rows(
    result32: aie_vector[bf16, 32],
    output: ptr[bf16, True],
    bn_w_ptr: ptr[bf16, True],
    bn_b_ptr: ptr[bf16, True],
    sp: i32,
    spatial_out: i32,
    oc: i32,
    oc_blk: i32,
) -> void:
    """Apply per-channel BN (bn_w*x + bn_b) + SiLU and store 4 rows × 8 cols.

    Matches the C++ tail:
      bn_acc = mul(row, bn_w_vec)
      bn_out = add(bn_acc.to_vector<bf16>(), bn_b_vec)
      for j in 0..7:
        x  = (float)bn_out[j]
        ax = |x|
        out[j] = bf16(x * (0.5 + x / (2 + 2*ax)))
    """
    # Extract the 4 rows upfront with compile-time constant indices.
    row0: aie_vector[bf16, 8] = vector_extract(result32, 0, 8)
    row1: aie_vector[bf16, 8] = vector_extract(result32, 8, 8)
    row2: aie_vector[bf16, 8] = vector_extract(result32, 16, 8)
    row3: aie_vector[bf16, 8] = vector_extract(result32, 24, 8)

    # Load BN params for this OC block once.
    bn_w8: aie_vector[bf16, 8] = load_v(bn_w_ptr + oc_blk * 8, 8)
    bn_b8: aie_vector[bf16, 8] = load_v(bn_b_ptr + oc_blk * 8, 8)

    if sp + 0 < spatial_out:
        _bn_silu_row(row0, bn_w8, bn_b8, output, (sp + 0) * oc + oc_blk * 8)
    if sp + 1 < spatial_out:
        _bn_silu_row(row1, bn_w8, bn_b8, output, (sp + 1) * oc + oc_blk * 8)
    if sp + 2 < spatial_out:
        _bn_silu_row(row2, bn_w8, bn_b8, output, (sp + 2) * oc + oc_blk * 8)
    if sp + 3 < spatial_out:
        _bn_silu_row(row3, bn_w8, bn_b8, output, (sp + 3) * oc + oc_blk * 8)


@aie_kernel
def _bn_silu_row(
    row: aie_vector[bf16, 8],
    bn_w8: aie_vector[bf16, 8],
    bn_b8: aie_vector[bf16, 8],
    output: ptr[bf16, True],
    out_off: i32,
) -> void:
    """BN + SiLU on a single 8-wide bf16 row, scalar (per-element).

    Uses ``extract_elem`` to index into the 8-wide vectors because PythoC
    doesn't support direct ``vec[j]`` subscript syntax for aie_vector.
    """
    j: i32 = 0
    while j < 8:
        bw: f32 = f32(extract_elem(bn_w8, j))
        bb: f32 = f32(extract_elem(bn_b8, j))
        rv: f32 = f32(extract_elem(row, j))
        # Mirror C++ rounding chain: f32 mul → bf16 trunc → f32 add.
        t1_bf: bf16 = bf16(rv * bw)
        t2: f32 = f32(t1_bf) + bb
        ax: f32 = t2
        if t2 < 0.0:
            ax = -t2
        denom: f32 = 2.0 + 2.0 * ax
        sig: f32 = 0.5 + t2 / denom
        output[out_off + j] = bf16(t2 * sig)
        j = j + 1


# ──────────────────────────────────────────────────────────────────────
# Factory helpers — return PythocKernel objects bound to inline-compiled
# @aie_kernel functions. ``build_dir`` is preserved for API compatibility
# with the old Pattern-A factories but is unused for inline kernels.
# ──────────────────────────────────────────────────────────────────────

# Helper list shared by the three mmul kernels.
_MMUL_HELPERS = [
    _build_a32_3x3,
    _build_a32_1x1,
    _build_a32_kblocked,
    _mul_4x8x8_bf16,
    _mac_4x8x8_bf16,
    _bn_silu_row,
    _store_bn_silu_4x8_rows,
]


def make_conv3x3_fused_packed_bf16(
    patch_ty,
    weight_ty,
    out_tile_ty,
    build_dir: Optional[Path] = None,
) -> PythocKernel:
    """conv3x3 + BN + SiLU. Six int32 scalar args after the three buffers."""
    return PythocKernel(
        conv3x3_fused_packed_bf16,
        [
            patch_ty, weight_ty, out_tile_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        extra_globals=KERNEL_EXTRA_GLOBALS,
        helpers=_MMUL_HELPERS,
    )


def make_gemm_conv1x1_fused_packed_bf16(
    in_ty,
    weight_ty,
    out_ty,
    build_dir: Optional[Path] = None,
) -> PythocKernel:
    """1x1 conv as GEMM + BN + SiLU. Same int32 scalars as conv3x3."""
    return PythocKernel(
        gemm_conv1x1_fused_packed_bf16,
        [
            in_ty, weight_ty, out_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        extra_globals=KERNEL_EXTRA_GLOBALS,
        helpers=_MMUL_HELPERS,
    )


def make_gemm_conv1x1_kblocked_bf16(
    in_ty,
    wt_chunk_ty,
    out_ty,
    build_dir: Optional[Path] = None,
) -> PythocKernel:
    """K-blocked GEMM. Scalar args: tile_m, full_ic, oc, k_start, k_block, n_k_blocks."""
    return PythocKernel(
        gemm_conv1x1_kblocked_bf16,
        [
            in_ty, wt_chunk_ty, out_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        extra_globals=KERNEL_EXTRA_GLOBALS,
        helpers=_MMUL_HELPERS,
    )


def make_residual_add_silu_bf16(
    cur_ty,
    res_ty,
    out_ty,
    build_dir: Optional[Path] = None,
    *,
    inline: bool = True,
) -> PythocKernel:
    """Residual add + SiLU (scalar PythoC, no mmul helpers needed)."""
    return PythocKernel(
        residual_add_silu_bf16,
        [cur_ty, res_ty, out_ty, np.int32, np.int32],
    )


# ──────────────────────────────────────────────────────────────────────
# Build-to-.o helper used by build_kernels.py
# ──────────────────────────────────────────────────────────────────────

def _materialise_kernel_obj(kernel: PythocKernel, dest: Path) -> None:
    """Copy the kernel's inline-compiled .o under the canonical filename."""
    src = Path(kernel.object_file_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def build_all_objs(build_dir: Optional[Path] = None) -> list[Path]:
    """Inline-compile all 4 kernels and stage their .o under build/<name>.o.

    Returns the list of staged .o paths.
    """
    build_dir = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)

    # Canonical buffer types (large enough to encompass all callers; the
    # IRON wrapper passes the actual shape at link time — the inline .o
    # just provides the named symbol).
    patch_ty = np.ndarray[(2048,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(8192,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(2048,), np.dtype[np.uint16]]

    outputs: list[Path] = []

    kc = make_conv3x3_fused_packed_bf16(patch_ty, weight_ty, out_ty, build_dir=build_dir)
    dst = build_dir / "conv3x3_fused_packed_bf16.o"
    _materialise_kernel_obj(kc, dst)
    outputs.append(dst)

    kg = make_gemm_conv1x1_fused_packed_bf16(patch_ty, weight_ty, out_ty, build_dir=build_dir)
    dst = build_dir / "gemm_conv1x1_fused_packed_bf16.o"
    _materialise_kernel_obj(kg, dst)
    outputs.append(dst)

    kk = make_gemm_conv1x1_kblocked_bf16(patch_ty, weight_ty, out_ty, build_dir=build_dir)
    dst = build_dir / "gemm_conv1x1_kblocked_bf16.o"
    _materialise_kernel_obj(kk, dst)
    outputs.append(dst)

    kr = make_residual_add_silu_bf16(patch_ty, patch_ty, out_ty, build_dir=build_dir)
    dst = build_dir / "residual_add_silu_bf16.o"
    _materialise_kernel_obj(kr, dst)
    outputs.append(dst)

    return outputs


__all__ = [
    "DEFAULT_BUILD_DIR",
    "KERNEL_EXTRA_GLOBALS",
    "residual_add_silu_bf16",
    "conv3x3_fused_packed_bf16",
    "gemm_conv1x1_fused_packed_bf16",
    "gemm_conv1x1_kblocked_bf16",
    "make_conv3x3_fused_packed_bf16",
    "make_gemm_conv1x1_fused_packed_bf16",
    "make_gemm_conv1x1_kblocked_bf16",
    "make_residual_add_silu_bf16",
    "build_all_objs",
]
