#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Standalone numerics test + drop-in wrapper for NPU decode attention.

This is the *contract* for the GQA-batched BFP576 decode attention (see
``ATTN_DECODE_GQA_SCOPE.md``).  It pins down, before any kernel/builder
exists:

  1. ``decode_attention_ref`` -- the float32 reference (identical math to
     ``llama32_1b_decode.decode_attention_cpu``, which is the thing we are
     replacing).  Used as ground truth.

  2. ``decode_attention_npu`` -- the drop-in wrapper with the SAME signature
     as ``decode_attention_cpu``.  It packs the per-layer KV cache into the
     per-group natural-layout buffers the kernel expects, runs the cached
     ``decode_attn`` kernel, and unpacks ``(n_heads, head_dim)``.  Host I/O
     is natural-layout (the kernel does the column-major 8x8 tiling
     internally via DMA, mirroring how ``flash_attn`` takes plain
     ``(seq, emb)`` q/k/v -- see llama32_1b_prefill.py:283).

  3. ``main`` -- compiles the kernel via the cache, runs both paths over a
     sweep of (seq_len) cases, reports max-abs error.

Usage:
    source env.sh
    python tools/test_decode_attn_npu.py            # full sweep on HW
    python tools/test_decode_attn_npu.py --ref-only  # reference self-check, no HW

Validation ladder (matches the scope doc):
    stage 0  seq_len == 64, single KV tile, no online rescale, one group
    stage 1  + multi-group (all 8 KV heads / 32 Q heads)
    stage 2  + online-softmax KV tiling for seq_len > 64 (arbitrary context)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16


# ---------------------------------------------------------------------------
# Config (llama-3.2-1B)
# ---------------------------------------------------------------------------
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
EMB_DIM = N_HEADS * HEAD_DIM          # 2048
KV_DIM = N_KV_HEADS * HEAD_DIM        # 512
GROUP_SIZE = N_HEADS // N_KV_HEADS    # 4  <- the M dimension we fill
TILE_ROWS = 64                        # BFP576 tile M (4 used, 60 zero-pad)


# ---------------------------------------------------------------------------
# Reference (float32) -- mirrors decode_attention_cpu exactly.
# ---------------------------------------------------------------------------
def decode_attention_ref(q, k_cache, v_cache, current_pos,
                         n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                         head_dim=HEAD_DIM):
    """Single-query attention with KV cache, float32 reference.

    Args:
        q: (emb_dim,) query vector for the current token (RoPE'd).
        k_cache: (n_kv_heads, max_seq, head_dim) cached keys.
        v_cache: (n_kv_heads, max_seq, head_dim) cached values.
        current_pos: 0-indexed position of the current token.
    Returns:
        (n_heads, head_dim) float32 attention output.
    """
    group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = current_pos + 1

    q_heads = q.astype(np.float32).reshape(n_heads, head_dim)
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale       # (seq,)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]                        # (head_dim,)
    return out


# ---------------------------------------------------------------------------
# Drop-in NPU wrapper -- same signature as decode_attention_cpu.
# ---------------------------------------------------------------------------
def decode_attention_npu(cache, q, k_cache, v_cache, current_pos,
                         n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                         head_dim=HEAD_DIM, *, kernel="decode_attn",
                         backend=None):
    """Runs the cached NPU decode-attention kernel.  Drop-in for
    ``llama32_1b_decode.decode_attention_cpu`` (returns bf16 (n_heads*hd,)).

    Packing contract (natural layout, kernel tiles internally):
      * q:  per group g, the GROUP_SIZE query heads -> (group_size, head_dim)
      * k:  k_cache[g, :seq_len, :]  -> (seq_len, head_dim)
      * v:  v_cache[g, :seq_len, :]  -> (seq_len, head_dim)
      * out: (group_size, head_dim) per group, reassembled to (n_heads, hd)

    The current build dispatches one group at a time (8 dispatches/token);
    a batched single-device variant is a follow-on (see scope doc).
    """
    group_size = n_heads // n_kv_heads
    seq_len = current_pos + 1
    n_chunks = (seq_len + TILE_ROWS - 1) // TILE_ROWS
    padded = n_chunks * TILE_ROWS

    q_heads = np.asarray(q, dtype=bfloat16).reshape(n_heads, head_dim)
    out = np.zeros((n_heads, head_dim), dtype=bfloat16)

    # The kernel takes a full TILE_ROWS x head_dim (64x64) q tile (4 real
    # heads in rows 0..3, zero pad) plus n_chunks K/V tiles, each 64 KV
    # positions x head_dim.  The trailing partial chunk is zero-padded; the
    # device masks the invalid score columns to -inf before softmax.  Host
    # feeds q UNSCALED (the 1/sqrt(head_dim) scale is folded into the
    # device-side softmax exp2).
    for g in range(n_kv_heads):
        q_pad = np.zeros((TILE_ROWS, head_dim), dtype=bfloat16)
        q_pad[:group_size] = q_heads[g * group_size:(g + 1) * group_size]
        q_g = np.ascontiguousarray(q_pad).reshape(-1)              # (4096,)

        k_pad = np.zeros((padded, head_dim), dtype=bfloat16)
        v_pad = np.zeros((padded, head_dim), dtype=bfloat16)
        k_pad[:seq_len] = k_cache[g, :seq_len, :].astype(bfloat16)
        v_pad[:seq_len] = v_cache[g, :seq_len, :].astype(bfloat16)
        # Pack each 64-row KV chunk as a contiguous (64,64) tile.
        k_g = np.ascontiguousarray(k_pad).reshape(-1)             # (n*4096,)
        v_g = np.ascontiguousarray(v_pad).reshape(-1)

        out_g = np.zeros(TILE_ROWS * head_dim, dtype=bfloat16)
        results = cache.load_and_run(
            kernel, backend, q_g, k_g, v_g, out_g,
            output_indices=[3],
        )
        ctx = results[-1].reshape(TILE_ROWS, head_dim)
        out[g * group_size:(g + 1) * group_size] = ctx[:group_size]
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# BATCHED single-dispatch NPU wrapper -- all n_groups in ONE load_and_run.
# ---------------------------------------------------------------------------
def decode_attention_npu_batched(cache, q, k_cache, v_cache, current_pos,
                                 n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                                 head_dim=HEAD_DIM, *, kernel,
                                 n_groups=N_KV_HEADS, backend=None):
    """Runs the batched NPU decode-attention kernel (1 dispatch, all groups).

    Packs all ``n_groups`` groups' q/k/v into single concatenated host BOs
    (group g at offset g*tile) and does ONE ``cache.load_and_run`` returning
    all groups' outputs.  Returns bf16 (n_heads*head_dim,).
    """
    group_size = n_heads // n_kv_heads
    seq_len = current_pos + 1
    n_chunks = (seq_len + TILE_ROWS - 1) // TILE_ROWS
    padded = n_chunks * TILE_ROWS
    tile_size = TILE_ROWS * head_dim
    kv_size = n_chunks * tile_size

    q_heads = np.asarray(q, dtype=bfloat16).reshape(n_heads, head_dim)
    out = np.zeros((n_heads, head_dim), dtype=bfloat16)

    q_all = np.zeros(n_groups * tile_size, dtype=bfloat16)
    k_all = np.zeros(n_groups * kv_size, dtype=bfloat16)
    v_all = np.zeros(n_groups * kv_size, dtype=bfloat16)
    out_all = np.zeros(n_groups * tile_size, dtype=bfloat16)

    for g in range(n_groups):
        q_pad = np.zeros((TILE_ROWS, head_dim), dtype=bfloat16)
        q_pad[:group_size] = q_heads[g * group_size:(g + 1) * group_size]
        q_all[g * tile_size:(g + 1) * tile_size] = q_pad.reshape(-1)

        k_pad = np.zeros((padded, head_dim), dtype=bfloat16)
        v_pad = np.zeros((padded, head_dim), dtype=bfloat16)
        k_pad[:seq_len] = k_cache[g, :seq_len, :].astype(bfloat16)
        v_pad[:seq_len] = v_cache[g, :seq_len, :].astype(bfloat16)
        k_all[g * kv_size:(g + 1) * kv_size] = k_pad.reshape(-1)
        v_all[g * kv_size:(g + 1) * kv_size] = v_pad.reshape(-1)

    results = cache.load_and_run(
        kernel, backend, q_all, k_all, v_all, out_all,
        output_indices=[3],
    )
    out_flat = np.asarray(results[-1], dtype=bfloat16).reshape(
        n_groups, TILE_ROWS, head_dim)
    for g in range(n_groups):
        out[g * group_size:(g + 1) * group_size] = out_flat[g, :group_size]
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
def _make_inputs(seq_len, seed=0):
    rng = np.random.default_rng(seed)
    max_seq = max(seq_len, 64)
    q = rng.standard_normal(EMB_DIM).astype(np.float32) * 0.1
    k_cache = rng.standard_normal(
        (N_KV_HEADS, max_seq, HEAD_DIM)).astype(np.float32) * 0.1
    v_cache = rng.standard_normal(
        (N_KV_HEADS, max_seq, HEAD_DIM)).astype(np.float32) * 0.1
    return (q.astype(bfloat16), k_cache.astype(bfloat16),
            v_cache.astype(bfloat16))


def _ref_self_check():
    """No-HW sanity: reference vs a second naive einsum implementation."""
    q, k, v = _make_inputs(40)
    pos = 39
    ref = decode_attention_ref(q, k, v, pos)
    # Independent einsum reimplementation.
    sl = pos + 1
    qh = q.astype(np.float32).reshape(N_HEADS, HEAD_DIM)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    out = np.zeros((N_HEADS, HEAD_DIM), np.float32)
    for h in range(N_HEADS):
        kv = h // GROUP_SIZE
        s = np.einsum("d,sd->s", qh[h], k[kv, :sl].astype(np.float32)) * scale
        p = np.exp(s - s.max()); p /= p.sum()
        out[h] = np.einsum("s,sd->d", p, v[kv, :sl].astype(np.float32))
    err = np.max(np.abs(ref - out))
    print(f"[ref-self-check] max-abs err = {err:.2e}", "OK" if err < 1e-4 else "FAIL")
    return err < 1e-4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-only", action="store_true",
                    help="reference self-check only, no hardware")
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[64, 128, 200, 256],
                    help="context lengths to test (validated n_chunks<=4, "
                         "i.e. seq_len<=256; larger needs a memtile KV feed)")
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="max-abs error tolerance (bf16)")
    args = ap.parse_args()

    if args.ref_only:
        sys.exit(0 if _ref_self_check() else 1)

    # HW path: build/compile the kernel via the KernelCache, then sweep.
    # Ensure the project root is importable and is cwd (link_with .o files
    # resolve relative to cwd inside compile_aie_to_elf).
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    os.chdir(_ROOT)

    from kernel_builder.cache import KernelCache
    from builders.attn_decode import build_decode_attn_module

    cache = KernelCache(cache_dir="decode_attn_test_cache", verbose=True)

    overall_ok = True
    for seq_len in args.seq_lens:
        # One compiled kernel per seq_len (chunk count is baked into the IR).
        kernel = f"decode_attn_s{seq_len}"
        if kernel not in cache.artifacts:
            try:
                ir = build_decode_attn_module(seq_len, verbose=True)
            except NotImplementedError as e:
                print(f"[seq_len={seq_len}] SKIP -- {e}")
                overall_ok = False
                continue
            cache.compile_and_cache(kernel, ir, "decode_attn")

        q, k_cache, v_cache = _make_inputs(seq_len, seed=seq_len)
        pos = seq_len - 1
        ref = decode_attention_ref(q, k_cache, v_cache, pos)
        npu = decode_attention_npu(cache, q, k_cache, v_cache, pos, kernel=kernel)
        npu = np.asarray(npu, dtype=bfloat16).reshape(N_HEADS, HEAD_DIM)
        npu_f = npu.astype(np.float32)

        err = float(np.max(np.abs(npu_f - ref)))
        per_head = [float(np.max(np.abs(npu_f[h] - ref[h]))) for h in range(N_HEADS)]
        ok = err < args.tol
        overall_ok = overall_ok and ok
        print(f"[seq_len={seq_len}] max-abs err = {err:.3e}  tol={args.tol:.1e}  "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"  worst head = {int(np.argmax(per_head))} (err {max(per_head):.3e})")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
