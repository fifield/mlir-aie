#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Focused harness for the c2_attn build (attention as wave 0 in c2_merged).

LADDER (ATTN_DECODE_GQA_SCOPE.md "c2_attn build"):
  stepA   STEP A correctness net: NPU batched attention -> repack to flat
          head-major attn_out -> drive the UNMODIFIED c2_merged device.
          Validate the c2_merged output(NPU-attn) ~= c2_merged(CPU-attn) within
          tol (NPU attn differs from CPU by ~3e-4; compare ~1e-2).
          This uses TWO host dispatches (attention, then c2) -- the safety net
          (per the scope: "OK to use an EXTRA configure/PDI here").

  stepB   STEP B collapse: attention folded onto the c2 row-2 herd as wave 0,
          ONE device / ONE configure / 1 LoadPDI.  build_c2_attn_module.
          Verify configure/load_pdi counts == 1, re-validate numerics.

Run:
    source ~/npu-dev-pythoc/env.sh
    cd build_peano
    python ../tools/test_c2_attn.py stepA
    python ../tools/test_c2_attn.py stepB
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from test_decode_attn_npu import (  # noqa: E402
    N_HEADS, N_KV_HEADS, HEAD_DIM, EMB_DIM, GROUP_SIZE, TILE_ROWS,
    decode_attention_ref, decode_attention_npu_batched, _make_inputs,
)

HIDDEN_DIM = 8192
TILE_SIZE = TILE_ROWS * HEAD_DIM  # 4096


# ---------------------------------------------------------------------------
# One-layer FFN reference + synthetic weights (mirrors run_decode_block call 2)
# ---------------------------------------------------------------------------
def _make_layer_weights(seed=0):
    rng = np.random.default_rng(seed)
    s = 0.05
    wo = (rng.standard_normal((EMB_DIM, EMB_DIM)) * s).astype(bfloat16)
    x_resid = (rng.standard_normal(EMB_DIM) * 0.1).astype(bfloat16)
    ffn_norm_w = (np.abs(rng.standard_normal(EMB_DIM)) * 0.5 + 0.5).astype(bfloat16)
    wgate = (rng.standard_normal((HIDDEN_DIM, EMB_DIM)) * s).astype(bfloat16)
    wup = (rng.standard_normal((HIDDEN_DIM, EMB_DIM)) * s).astype(bfloat16)
    wdown = (rng.standard_normal((EMB_DIM, HIDDEN_DIM)) * s).astype(bfloat16)
    return dict(wo=wo, x_resid=x_resid, ffn_norm_w=ffn_norm_w,
                wgate=wgate, wup=wup, wdown=wdown)


def _rms_norm(x, w, eps=1e-5):
    x = x.astype(np.float32)
    var = np.mean(x * x)
    return (x / np.sqrt(var + eps) * w.astype(np.float32)).astype(np.float32)


def c2_ffn_ref(attn_vec, lw):
    """CPU reference for the c2_merged call: O / add1 / rms / gate / up /
    swiglu / down / add2.  attn_vec is the flat (2048,) head-major attn out.
    """
    attn = attn_vec.astype(np.float32)
    proj = lw["wo"].astype(np.float32) @ attn          # (2048,)
    res1 = proj + lw["x_resid"].astype(np.float32)     # add1
    normed2 = _rms_norm(res1, lw["ffn_norm_w"])        # rms
    gate = lw["wgate"].astype(np.float32) @ normed2    # (8192,)
    up = lw["wup"].astype(np.float32) @ normed2        # (8192,)
    swiglu = gate / (1.0 + np.exp(-gate)) * up         # swiglu
    down = lw["wdown"].astype(np.float32) @ swiglu     # (2048,)
    out = down + res1                                  # add2
    return out.astype(bfloat16)


# ---------------------------------------------------------------------------
# c2_merged host driver (UNMODIFIED device; 15-arg o_gemv_ffn ABI).
# ---------------------------------------------------------------------------
def _o_gemv_ffn_args(attn_vec, lw):
    return [
        np.ascontiguousarray(lw["wo"]),                  # 0 wo
        np.ascontiguousarray(attn_vec),                  # 1 attn_out
        np.zeros(EMB_DIM, dtype=bfloat16),               # 2 proj
        np.ascontiguousarray(lw["x_resid"]),             # 3 x_residual
        np.zeros(EMB_DIM, dtype=bfloat16),               # 4 res1
        np.ascontiguousarray(lw["ffn_norm_w"]),          # 5 ffn_norm_w
        np.zeros(EMB_DIM, dtype=bfloat16),               # 6 normed2
        np.ascontiguousarray(lw["wgate"]),               # 7 wgate
        np.zeros(HIDDEN_DIM, dtype=bfloat16),            # 8 gate
        np.ascontiguousarray(lw["wup"]),                 # 9 wup
        np.zeros(HIDDEN_DIM, dtype=bfloat16),            # 10 up
        np.zeros(HIDDEN_DIM, dtype=bfloat16),            # 11 swiglu
        np.ascontiguousarray(lw["wdown"]),               # 12 wdown
        np.zeros(EMB_DIM, dtype=bfloat16),               # 13 down
        np.zeros(EMB_DIM, dtype=bfloat16),               # 14 output
    ]


def c2_merged_npu(cache, attn_vec, lw, *, kernel, backend):
    args = _o_gemv_ffn_args(attn_vec, lw)
    res = cache.load_and_run(
        kernel, backend, *args,
        output_indices=[14],
        static_input_indices={0, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
    )
    return np.asarray(res[14], dtype=bfloat16)


def _build_c2_merged(cache):
    """Build the UNMODIFIED c2_merged device under its own cache instance.

    The dispatcher's runtime_sequence symbol is ``@o_gemv_ffn`` -> the XRT
    kernel identifier is ``main:o_gemv_ffn``, so instance_name must be that.
    """
    kernel = "c2_merged_ref"
    if kernel not in cache.artifacts:
        from builders.o_gemv_ffn import build_o_gemv_ffn_module
        ir = build_o_gemv_ffn_module(pack_mode="c2_merged")
        cache.compile_and_cache(kernel, ir, "o_gemv_ffn")
    return kernel, None


# ---------------------------------------------------------------------------
# STEP A
# ---------------------------------------------------------------------------
def cmd_stepA(args):
    from kernel_builder.cache import KernelCache

    cache = KernelCache(cache_dir="c2_attn_cache", verbose=True)
    kernel, backend = _build_c2_merged(cache)

    # Batched attention device (validated emitter).
    from builders.attn_decode import build_decode_attn_batched_module
    attn_kernel = f"decode_attn_b{args.seq_len}"
    if attn_kernel not in cache.artifacts:
        ir = build_decode_attn_batched_module(args.seq_len, N_KV_HEADS,
                                              verbose=True)
        cache.compile_and_cache(attn_kernel, ir, "decode_attn")

    ok = True
    for seed in range(args.seeds):
        q, k_cache, v_cache = _make_inputs(args.seq_len, seed=seed)
        pos = args.seq_len - 1
        lw = _make_layer_weights(seed=seed)

        # --- CPU-attention reference path ---
        attn_cpu = decode_attention_ref(q, k_cache, v_cache, pos)  # (32,64) f32
        attn_cpu_vec = attn_cpu.reshape(-1).astype(bfloat16)
        ref_out = c2_ffn_ref(attn_cpu_vec, lw).astype(np.float32)

        # sanity: c2_merged(CPU attn) ~= cpu c2 ref
        c2_cpu_out = c2_merged_npu(cache, attn_cpu_vec, lw,
                                   kernel=kernel, backend=backend).astype(np.float32)
        e_c2 = float(np.max(np.abs(c2_cpu_out - ref_out)))

        # --- NPU-attention path ---
        attn_npu = decode_attention_npu_batched(
            cache, q, k_cache, v_cache, pos,
            kernel=attn_kernel)            # bf16 (2048,) head-major
        attn_npu_vec = np.asarray(attn_npu, dtype=bfloat16)
        e_attn = float(np.max(np.abs(
            attn_npu_vec.astype(np.float32) - attn_cpu_vec.astype(np.float32))))

        npu_out = c2_merged_npu(cache, attn_npu_vec, lw,
                                kernel=kernel, backend=backend).astype(np.float32)

        # c2_attn(npu attn) vs c2_merged(cpu attn)  (the gate)
        e_end = float(np.max(np.abs(npu_out - c2_cpu_out)))
        rel = e_end / (float(np.max(np.abs(c2_cpu_out))) + 1e-9)
        good = e_end < 1e-2 or rel < 2e-2
        ok = ok and good
        print(f"[stepA seed={seed} seq={args.seq_len}] "
              f"attn_err={e_attn:.3e} c2(cpu)-cpuref={e_c2:.3e} "
              f"end_err={e_end:.3e} rel={rel:.3e} "
              f"{'PASS' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# STEP B
# ---------------------------------------------------------------------------
def cmd_stepB(args):
    from kernel_builder.cache import KernelCache
    from builders.c2_attn import build_c2_attn_module

    from builders.c2_attn import c2_attn_kernel_id

    cache = KernelCache(cache_dir="c2_attn_cache", verbose=True)
    kernel = f"c2_attn_s{args.seq_len}"
    backend = None
    if kernel not in cache.artifacts or args.rebuild:
        ir = build_c2_attn_module(args.seq_len, N_KV_HEADS, verbose=True)
        # dispatcher sym is unique per seq -> kernel id main:o_gemv_ffn_c2attn_sN
        cache.compile_and_cache(kernel, ir, c2_attn_kernel_id(args.seq_len))

    # Reference c2_merged for the gate.
    ref_kernel, ref_backend = _build_c2_merged(cache)

    ok = True
    for seed in range(args.seeds):
        q, k_cache, v_cache = _make_inputs(args.seq_len, seed=seed)
        pos = args.seq_len - 1
        lw = _make_layer_weights(seed=seed)

        attn_cpu = decode_attention_ref(q, k_cache, v_cache, pos)
        attn_cpu_vec = attn_cpu.reshape(-1).astype(bfloat16)
        c2_cpu_out = c2_merged_npu(cache, attn_cpu_vec, lw,
                                   kernel=ref_kernel,
                                   backend=ref_backend).astype(np.float32)

        npu_out = c2_attn_npu(cache, q, k_cache, v_cache, pos, lw,
                              kernel=kernel, backend=backend,
                              seq_len=args.seq_len).astype(np.float32)
        e_end = float(np.max(np.abs(npu_out - c2_cpu_out)))
        rel = e_end / (float(np.max(np.abs(c2_cpu_out))) + 1e-9)
        good = e_end < 1e-2 or rel < 2e-2
        ok = ok and good
        print(f"[stepB seed={seed} seq={args.seq_len}] "
              f"end_err={e_end:.3e} rel={rel:.3e} {'PASS' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


def _pack_attn_inputs(q, k_cache, v_cache, current_pos, n_groups, seq_len):
    """Pack q/k/v into the batched-attention concatenated BO layout."""
    group_size = N_HEADS // N_KV_HEADS
    n_chunks = (seq_len + TILE_ROWS - 1) // TILE_ROWS
    padded = n_chunks * TILE_ROWS
    kv_size = n_chunks * TILE_SIZE
    q_heads = np.asarray(q, dtype=bfloat16).reshape(N_HEADS, HEAD_DIM)
    q_all = np.zeros(n_groups * TILE_SIZE, dtype=bfloat16)
    k_all = np.zeros(n_groups * kv_size, dtype=bfloat16)
    v_all = np.zeros(n_groups * kv_size, dtype=bfloat16)
    for g in range(n_groups):
        q_pad = np.zeros((TILE_ROWS, HEAD_DIM), dtype=bfloat16)
        q_pad[:group_size] = q_heads[g * group_size:(g + 1) * group_size]
        q_all[g * TILE_SIZE:(g + 1) * TILE_SIZE] = q_pad.reshape(-1)
        k_pad = np.zeros((padded, HEAD_DIM), dtype=bfloat16)
        v_pad = np.zeros((padded, HEAD_DIM), dtype=bfloat16)
        k_pad[:seq_len] = k_cache[g, :seq_len, :].astype(bfloat16)
        v_pad[:seq_len] = v_cache[g, :seq_len, :].astype(bfloat16)
        k_all[g * kv_size:(g + 1) * kv_size] = k_pad.reshape(-1)
        v_all[g * kv_size:(g + 1) * kv_size] = v_pad.reshape(-1)
    return q_all, k_all, v_all


def c2_attn_npu(cache, q, k_cache, v_cache, current_pos, lw, *,
                kernel, backend, seq_len, n_groups=N_KV_HEADS):
    """Drive the c2_attn collapsed device.  ABI: arg1 (attn_out input) is
    REPLACED by q_roped/k_cache/v_cache; the rest of the 15-arg c2 layout is
    preserved (wo at 0, ... output at 14).  See builders/c2_attn.py for the
    exact extended host signature.
    """
    q_all, k_all, v_all = _pack_attn_inputs(q, k_cache, v_cache, current_pos,
                                            n_groups, seq_len)
    # Extended host args: c2's 15 args, but arg1 (attn_out) is WIDENED to the
    # per-group tiled scratch (n_groups*4096) that the attention wave writes;
    # q/k/v are appended (args 15/16/17).
    args = _o_gemv_ffn_args(np.zeros(EMB_DIM, dtype=bfloat16), lw)
    args[1] = np.zeros(n_groups * TILE_SIZE, dtype=bfloat16)   # wide attn_out
    args = args + [q_all, k_all, v_all]   # args 15,16,17
    res = cache.load_and_run(
        kernel, backend, *args,
        output_indices=[14],
        static_input_indices={0, 7, 9, 12},
        intermediate_indices={1, 2, 4, 6, 8, 10, 11, 13, 14},
    )
    return np.asarray(res[14], dtype=bfloat16)


# ---------------------------------------------------------------------------
# RESIDENT (stepR): ONE fixed-structure PDI (MAX_CHUNKS=4) reused for every
# position; the trailing-chunk mask is a runtime value L = current_pos+1.
# ---------------------------------------------------------------------------
import os as _os_r
# MEMKV (PYTHOC_C2_ATTN_MEMKV=1) lifts the 4-chunk shim-direct cap; the host
# pads to PYTHOC_C2_ATTN_MAX_CHUNKS*64.  Must mirror builders/o_gemv_ffn.py.
RES_MAX_CHUNKS = (int(_os_r.environ.get("PYTHOC_C2_ATTN_MAX_CHUNKS", "8"))
                  if _os_r.environ.get("PYTHOC_C2_ATTN_MEMKV", "0") == "1" else 4)
RES_PADDED = RES_MAX_CHUNKS * TILE_ROWS  # 256 (or MAX_CHUNKS*64 under MEMKV)


def _tile_8x8(mat):
    """Tile a (64,64) matrix into 8x8 column-block-major flat layout:
    element (row,col) -> flat (col//8)*512 + row*8 + (col%8).  Matches the
    on-device q/k/v tiled buffer the matmul kernels read."""
    out = np.zeros(64 * 64, dtype=mat.dtype)
    cols = np.arange(64)
    for row in range(64):
        flat = (cols // 8) * 512 + row * 8 + (cols % 8)
        out[flat] = mat[row, :]
    return out


def _pack_attn_inputs_resident(q, k_cache, v_cache, current_pos, n_groups):
    """Pack q/k/v with FIXED 4-chunk padding (256 KV rows, host zero-pads).
    K/V are PRE-TILED per 64-row chunk (the resident device flat-copies them
    into an i8 L1 buffer it views per-chunk; the DMA does no tiling transform).
    q keeps the untiled layout (its DMA tiles it).  Returns q_all,k_all,v_all,L.
    """
    group_size = N_HEADS // N_KV_HEADS
    seq_len = current_pos + 1
    kv_size = RES_MAX_CHUNKS * TILE_SIZE
    q_heads = np.asarray(q, dtype=bfloat16).reshape(N_HEADS, HEAD_DIM)
    q_all = np.zeros(n_groups * TILE_SIZE, dtype=bfloat16)
    k_all = np.zeros(n_groups * kv_size, dtype=bfloat16)
    v_all = np.zeros(n_groups * kv_size, dtype=bfloat16)
    for g in range(n_groups):
        q_pad = np.zeros((TILE_ROWS, HEAD_DIM), dtype=bfloat16)
        q_pad[:group_size] = q_heads[g * group_size:(g + 1) * group_size]
        q_all[g * TILE_SIZE:(g + 1) * TILE_SIZE] = q_pad.reshape(-1)
        k_pad = np.zeros((RES_PADDED, HEAD_DIM), dtype=bfloat16)
        v_pad = np.zeros((RES_PADDED, HEAD_DIM), dtype=bfloat16)
        k_pad[:seq_len] = k_cache[g, :seq_len, :].astype(bfloat16)
        v_pad[:seq_len] = v_cache[g, :seq_len, :].astype(bfloat16)
        for c in range(RES_MAX_CHUNKS):
            kt = _tile_8x8(k_pad[c * TILE_ROWS:(c + 1) * TILE_ROWS])
            vt = _tile_8x8(v_pad[c * TILE_ROWS:(c + 1) * TILE_ROWS])
            base = g * kv_size + c * TILE_SIZE
            k_all[base:base + TILE_SIZE] = kt
            v_all[base:base + TILE_SIZE] = vt
        # Fold runtime valid length L into q's padding.  q_all is UNTILED
        # (the device DMA tiles it 8x8); the core reads tiled offset 32 =
        # (row4,col0), whose UNTILED position is row4*64+col0 = 256.  Only
        # rows 0..3 hold real heads, so row 4 is free.
        q_all[g * TILE_SIZE + 256] = bfloat16(float(seq_len))
    return q_all, k_all, v_all


def c2_attn_resident_npu(cache, q, k_cache, v_cache, current_pos, lw, *,
                         kernel, backend, n_groups=N_KV_HEADS):
    q_all, k_all, v_all = _pack_attn_inputs_resident(
        q, k_cache, v_cache, current_pos, n_groups)
    args = _o_gemv_ffn_args(np.zeros(EMB_DIM, dtype=bfloat16), lw)
    args[1] = np.zeros(n_groups * TILE_SIZE, dtype=bfloat16)   # wide attn_out
    args = args + [q_all, k_all, v_all]   # args 15,16,17
    res = cache.load_and_run(
        kernel, backend, *args,
        output_indices=[14],
        static_input_indices={0, 7, 9, 12},
        intermediate_indices={1, 2, 4, 6, 8, 10, 11, 13, 14},
    )
    return np.asarray(res[14], dtype=bfloat16)


def cmd_stepR(args):
    from kernel_builder.cache import KernelCache
    from builders.c2_attn import (build_c2_attn_resident_module,
                                  c2_attn_resident_kernel_id)

    cache = KernelCache(cache_dir="c2_attn_cache", verbose=True)
    kernel = "c2_attn_resident"
    backend = None
    if kernel not in cache.artifacts or args.rebuild:
        ir = build_c2_attn_resident_module(N_KV_HEADS, verbose=True)
        cache.compile_and_cache(kernel, ir, c2_attn_resident_kernel_id())

    ref_kernel, ref_backend = _build_c2_merged(cache)

    # Positions spanning chunk boundaries + partial chunks.
    positions = args.positions or [0, 39, 63, 64, 65, 99, 127, 128, 200, 255]
    ok = True
    # Determinism: run each position twice, require bit-exact device output.
    for pos in positions:
        seq_len = pos + 1
        q, k_cache, v_cache = _make_inputs(seq_len, seed=pos)
        lw = _make_layer_weights(seed=pos)

        attn_cpu = decode_attention_ref(q, k_cache, v_cache, pos)
        attn_cpu_vec = attn_cpu.reshape(-1).astype(bfloat16)
        c2_cpu_out = c2_merged_npu(cache, attn_cpu_vec, lw, kernel=ref_kernel,
                                   backend=ref_backend).astype(np.float32)

        out1 = c2_attn_resident_npu(cache, q, k_cache, v_cache, pos, lw,
                                    kernel=kernel, backend=backend)
        out2 = c2_attn_resident_npu(cache, q, k_cache, v_cache, pos, lw,
                                    kernel=kernel, backend=backend)
        det = bool(np.array_equal(np.asarray(out1, dtype=bfloat16),
                                  np.asarray(out2, dtype=bfloat16)))
        npu_out = np.asarray(out1, dtype=bfloat16).astype(np.float32)
        e_end = float(np.max(np.abs(npu_out - c2_cpu_out)))
        rel = e_end / (float(np.max(np.abs(c2_cpu_out))) + 1e-9)
        good = (e_end < 1e-2 or rel < 2e-2) and det
        ok = ok and good
        print(f"[stepR pos={pos:3d} seq={seq_len:3d}] "
              f"end_err={e_end:.3e} rel={rel:.3e} det={det} "
              f"{'PASS' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


def _c2_merged_npu_full(cache, attn_vec, lw, *, kernel, backend):
    """Like c2_merged_npu but also reads back proj (idx 2) for drift probing.

    Returns (proj, output) both as float32. Reads idx 2 + 14.
    """
    import pyxrt as xrt
    args = _o_gemv_ffn_args(attn_vec, lw)
    res = cache.load_and_run(
        kernel, backend, *args,
        output_indices=[2, 14],
        static_input_indices={0, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
    )
    proj = np.asarray(res[2], dtype=bfloat16).astype(np.float32)
    out = np.asarray(res[14], dtype=bfloat16).astype(np.float32)
    return proj, out


def cmd_stepW(args):
    """Back-to-back WARM reuse of the c2_merged ELF in ONE process.

    Dispatch the SAME c2_merged PDI N times on the SAME input with NO
    intervening different-PDI dispatch.  REQUIRE call1==call2==...==call0
    bit-identical (proj idx 2 AND output idx 14), AND call0 correct vs the
    numpy CPU reference.  This is the quiescence gate: it proves the c2 fabric
    state returns to its cold post-LoadPDI init at end-of-dispatch so warm
    reuse == cold.
    """
    from kernel_builder.cache import KernelCache

    cache = KernelCache(cache_dir="c2_attn_cache", verbose=True)
    kernel, backend = _build_c2_merged(cache)

    n = args.reuse
    # A couple of "positions" = seeds (incl. one partial-chunk-shaped input via
    # different seed); each seed re-runs the same PDI n times back-to-back.
    seeds = args.seeds_list or [0, 1, 7]
    ok = True
    for seed in seeds:
        lw = _make_layer_weights(seed=seed)
        # Use a CPU attention vector as the (host-synced) activation input.
        q, k_cache, v_cache = _make_inputs(64, seed=seed)
        attn_cpu = decode_attention_ref(q, k_cache, v_cache, 63)
        attn_vec = attn_cpu.reshape(-1).astype(bfloat16)

        ref_proj = (lw["wo"].astype(np.float32) @ attn_vec.astype(np.float32))
        ref_out = c2_ffn_ref(attn_vec, lw).astype(np.float32)

        projs, outs = [], []
        for _ in range(n):
            p, o = _c2_merged_npu_full(cache, attn_vec, lw,
                                       kernel=kernel, backend=backend)
            projs.append(p)
            outs.append(o)

        # call0 correctness
        e_proj0 = float(np.max(np.abs(projs[0] - ref_proj)))
        rel_proj0 = e_proj0 / (float(np.max(np.abs(ref_proj))) + 1e-9)
        e_out0 = float(np.max(np.abs(outs[0] - ref_out)))
        rel_out0 = e_out0 / (float(np.max(np.abs(ref_out))) + 1e-9)
        cold_ok = rel_proj0 < 5e-2 and rel_out0 < 5e-2

        # warm == cold bit-identity
        proj_bits = [projs[i].view(np.uint32) for i in range(n)]
        out_bits = [outs[i].view(np.uint32) for i in range(n)]
        proj_id = all(np.array_equal(proj_bits[i], proj_bits[0])
                      for i in range(1, n))
        out_id = all(np.array_equal(out_bits[i], out_bits[0])
                     for i in range(1, n))
        # how far do warm calls drift from cold (diagnostic)
        max_proj_drift = max(
            (float(np.max(np.abs(projs[i] - projs[0]))) for i in range(1, n)),
            default=0.0)
        n_proj_diff = max(
            (int(np.sum(projs[i] != projs[0])) for i in range(1, n)),
            default=0)

        good = cold_ok and proj_id and out_id
        ok = ok and good
        print(f"[stepW seed={seed} n={n}] cold:rel_proj={rel_proj0:.3e} "
              f"rel_out={rel_out0:.3e} | warm==cold proj={proj_id} out={out_id} "
              f"| drift_proj_max={max_proj_drift:.3e} "
              f"n_proj_diff={n_proj_diff} {'PASS' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stepA", "stepB"):
        p = sub.add_parser(name)
        p.add_argument("--seq-len", dest="seq_len", type=int, default=64)
        p.add_argument("--seeds", type=int, default=3)
        p.add_argument("--rebuild", action="store_true")
    pr = sub.add_parser("stepR")
    pr.add_argument("--rebuild", action="store_true")
    pr.add_argument("--positions", type=int, nargs="*", default=None)
    pw = sub.add_parser("stepW")
    pw.add_argument("--reuse", type=int, default=4)
    pw.add_argument("--seeds-list", dest="seeds_list", type=int, nargs="*",
                    default=None)
    args = ap.parse_args()
    {"stepA": cmd_stepA, "stepB": cmd_stepB, "stepR": cmd_stepR,
     "stepW": cmd_stepW}[args.cmd](args)


if __name__ == "__main__":
    main()
