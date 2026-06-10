#!/usr/bin/env python3
# rn3_chain_pythoc.py — chained rn3-pair iteration kernels (one per stage),
# geometry-generic: ic is a runtime arg (32 = re4, 48 = re6, 64 = re8).
#
# Mode-free variants of rn3_pair_vector_stage_bf16: each stage is its own
# @aie_kernel so the per-stage loops carry no mode branches (mode-branched
# variants measured 4-5x slower). Layout:
#   * 8x8 tiles, 12-wide patches (8 + 2x2 halo) at every geometry;
#   * scratch = (ic/16) planes of 1600 u16 (10x10x16), worker-local Buffer;
#   * finals = compact 64*ic u16 HWC tile in the out FIFO;
#   * weight slot = 16*ic*9 wts + bn_w(16) + bn_b(16); (ic/16) slots/conv.

from __future__ import annotations

from pythoc import ptr, i16, i32, bf16, f32, void
from pythoc.aie import (
    aie_vector, load_v, store_v, vector_add, vector_sub, vector_mul,
    vector_and, vector_cast, vector_extract, broadcast, concat, zeros,
)
from pythoc.aie.operations import write_tm, read_tm, lock_acquire, lock_release
from pythoc.aie.mmul import acc_to_bf16
from pythoc.aie.profiling import event0, event1

from aie.iron.pythoc import aie_kernel

from kernels.rep_elan_bf16_pythoc import (  # noqa: F401
    KERNEL_EXTRA_GLOBALS,
    _MMUL_HELPERS,
    _build_a32_3x3,
    _mul_4x8x8_bf16,
    _mac_4x8x8_bf16,
    _store_bn_silu_4x8_rows,
)


@aie_kernel
def chain_conv1_bf16(
    arena_base: ptr[bf16, True],
    weight: ptr[bf16, True],
    scratch: ptr[bf16, True],
    block: i32,
    t: i32,
    ic: i32,
) -> void:
    """Conv1: 12x12xic patch t -> 10x10x16 scratch plane `block` (no mask)."""
    event0()
    arena_in: ptr[bf16] = arena_base + t * (96 * ic)
    scr: ptr[bf16] = scratch + block * 1600
    kkmax: i32 = (ic // 8) * 9
    bn1_w: ptr[bf16] = weight + 16 * ic * 9
    bn1_b: ptr[bf16] = bn1_w + 16
    woff_b: i32 = kkmax * 64
    sp: i32 = 0
    while sp < 100:
        A0: aie_vector[bf16, 32] = _build_a32_3x3(arena_in, sp, 10, 1, 12, ic, 0, 0, 0)
        acc_a: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, load_v(weight, 64))
        acc_b: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, load_v(weight + woff_b, 64))
        kk: i32 = 1
        while kk < kkmax:
            ic_blk: i32 = kk // 9
            kk_in_blk: i32 = kk - ic_blk * 9
            kh: i32 = kk_in_blk // 3
            kw: i32 = kk_in_blk - kh * 3
            A: aie_vector[bf16, 32] = _build_a32_3x3(arena_in, sp, 10, 1, 12, ic, ic_blk, kh, kw)
            acc_a = _mac_4x8x8_bf16(A, load_v(weight + kk * 64, 64), acc_a)
            acc_b = _mac_4x8x8_bf16(A, load_v(weight + woff_b + kk * 64, 64), acc_b)
            kk = kk + 1
        _store_bn_silu_4x8_rows(acc_to_bf16(acc_a), scr, bn1_w, bn1_b, sp, 100, 16, 0)
        _store_bn_silu_4x8_rows(acc_to_bf16(acc_b), scr, bn1_w, bn1_b, sp, 100, 16, 1)
        sp = sp + 4
    event1()


@aie_kernel
def chain_mask_bf16(
    scratch: ptr[bf16, True],
    grow: i32,
    gcol: i32,
    gbound: i32,
    n_planes: i32,
) -> void:
    """Zero conv1 scratch positions outside the gbound x gbound image domain.

    Scratch (r,c) maps to image (grow + r - 1, gcol + c - 1).
    """
    event0()
    z16: aie_vector[bf16, 16] = zeros(bf16, 16)
    mz: i32 = 0
    while mz < 100:
        r: i32 = mz // 10
        c: i32 = mz - r * 10
        gr: i32 = grow + r - 1
        gc: i32 = gcol + c - 1
        if gr < 0 or gr >= gbound or gc < 0 or gc >= gbound:
            p: i32 = 0
            while p < n_planes:
                store_v(scratch + p * 1600 + mz * 16, z16)
                p = p + 1
        mz = mz + 1
    event1()


@aie_kernel
def _store_bn_silu_res_4x8_rows(
    result32: aie_vector[bf16, 32],
    output: ptr[bf16, True],
    bn_w_ptr: ptr[bf16, True],
    bn_b_ptr: ptr[bf16, True],
    sp: i32,
    oc_blk: i32,
    arena_in: ptr[bf16, True],
    oc: i32,
    grow: i32,
    gcol: i32,
    gbound: i32,
) -> void:
    """BN + SiLU + residual (out = silu(bn(acc)) + patch center) for 4x8 rows.

    Same math as _store_bn_silu_4x8_rows plus the 8-lane patch-center residual
    folded in at store time — avoids reading finals back after conv stores.
    oc_blk indexes the FULL oc-channel output; the slot's bn arrays only cover
    its own 16 channels, so bn is indexed by the slot-local block (oc_blk % 2).
    """
    bn_blk: i32 = oc_blk - (oc_blk // 2) * 2
    bn_w8: aie_vector[bf16, 8] = load_v(bn_w_ptr + bn_blk * 8, 8)
    bn_b8: aie_vector[bf16, 8] = load_v(bn_b_ptr + bn_blk * 8, 8)
    bn_w16: aie_vector[bf16, 16] = concat(bn_w8, bn_w8)
    bn_b16: aie_vector[bf16, 16] = concat(bn_b8, bn_b8)
    bnw32: aie_vector[bf16, 32] = concat(bn_w16, bn_w16)
    bnb32: aie_vector[bf16, 32] = concat(bn_b16, bn_b16)

    two32: aie_vector[bf16, 32] = vector_cast(broadcast(i16, 32, 0x4000), bf16, 32)
    one32: aie_vector[bf16, 32] = vector_cast(broadcast(i16, 32, 0x3F80), bf16, 32)

    t1: aie_vector[bf16, 32] = vector_mul(result32, bnw32)
    t2: aie_vector[bf16, 32] = vector_add(t1, bnb32)

    abits: aie_vector[i16, 32] = vector_and(vector_cast(t2, i16, 32), broadcast(i16, 32, 0x7FFF))
    a: aie_vector[bf16, 32] = vector_cast(abits, bf16, 32)
    d: aie_vector[bf16, 32] = vector_add(vector_add(a, a), two32)

    rbits: aie_vector[i16, 32] = vector_sub(broadcast(i16, 32, 0x7EF4), vector_cast(d, i16, 32))
    r: aie_vector[bf16, 32] = vector_cast(rbits, bf16, 32)
    r = vector_mul(r, vector_sub(two32, vector_mul(d, r)))
    r = vector_mul(r, vector_sub(two32, vector_mul(d, r)))

    n: aie_vector[bf16, 32] = vector_add(vector_add(a, t2), one32)
    out32: aie_vector[bf16, 32] = vector_mul(t2, vector_mul(n, r))

    pos0: i32 = sp
    r0: i32 = pos0 // 8
    off0: i32 = ((r0 + 2) * 12 + (pos0 - r0 * 8 + 2)) * oc + oc_blk * 8
    pos1: i32 = sp + 1
    r1: i32 = pos1 // 8
    off1: i32 = ((r1 + 2) * 12 + (pos1 - r1 * 8 + 2)) * oc + oc_blk * 8
    pos2: i32 = sp + 2
    r2: i32 = pos2 // 8
    off2: i32 = ((r2 + 2) * 12 + (pos2 - r2 * 8 + 2)) * oc + oc_blk * 8
    pos3: i32 = sp + 3
    r3: i32 = pos3 // 8
    off3: i32 = ((r3 + 2) * 12 + (pos3 - r3 * 8 + 2)) * oc + oc_blk * 8
    c01: aie_vector[bf16, 16] = concat(load_v(arena_in + off0, 8), load_v(arena_in + off1, 8))
    c23: aie_vector[bf16, 16] = concat(load_v(arena_in + off2, 8), load_v(arena_in + off3, 8))
    res32: aie_vector[bf16, 32] = concat(c01, c23)
    sum32: aie_vector[bf16, 32] = vector_add(out32, res32)
    z8: aie_vector[bf16, 8] = zeros(bf16, 8)
    # rows past the image bound store zero so drained junk never reaches
    # the next iteration's halo reads
    if grow + r0 < gbound and gcol + (pos0 - r0 * 8) < gbound:
        store_v(output + pos0 * oc + oc_blk * 8, vector_extract(sum32, 0, 8))
    else:
        store_v(output + pos0 * oc + oc_blk * 8, z8)
    if grow + r1 < gbound and gcol + (pos1 - r1 * 8) < gbound:
        store_v(output + pos1 * oc + oc_blk * 8, vector_extract(sum32, 8, 8))
    else:
        store_v(output + pos1 * oc + oc_blk * 8, z8)
    if grow + r2 < gbound and gcol + (pos2 - r2 * 8) < gbound:
        store_v(output + pos2 * oc + oc_blk * 8, vector_extract(sum32, 16, 8))
    else:
        store_v(output + pos2 * oc + oc_blk * 8, z8)
    if grow + r3 < gbound and gcol + (pos3 - r3 * 8) < gbound:
        store_v(output + pos3 * oc + oc_blk * 8, vector_extract(sum32, 24, 8))
    else:
        store_v(output + pos3 * oc + oc_blk * 8, z8)


# wt replay: the core arms its own S2MM ch1 into a fixed-address slot buffer
# (address patched onto the buffer op post-resolve), released on lock WT_LOCK.
WT_BD = 15
WT_LOCK = 12              # core lock id; localized acquire id = 48 + 12
WT_BUF_ADDR = 0xC800     # after iron buffers (cw_out 0xC000+2K), ends 0xECC0 < counters 0xF000
# 0x80000-based own-tile alias: bare-micro-proven base. The raw 0x1D000
# constants sext-lower to 0xFFF9xxxx pointers, which wedge the core.
DMA_BD_BASE = 0x0009D000
DMA_S2MM_1_START_QUEUE = 0x0009DE0C
DMA_MM2S_1_START_QUEUE = 0x0009DE1C
TOK_ADDR = 0xEFE0          # 1-word credit source, between wbuf end and counters


@aie_kernel
def chain_wt_arm_tok(pkt_id: i32) -> void:
    """Arm wt S2MM ch1 + send a 1-word credit packet (MM2S ch1, BD14).

    The memtile emits one slot only when NWORK credits arrive — the stream
    never parks (parked circuit beats head-of-line block the chain fills)."""
    bd: i32 = DMA_BD_BASE + WT_BD * 32
    write_tm(((WT_BUF_ADDR // 4) << 14) | WT_SLOT_I32, bd)
    write_tm(0, bd + 4)
    write_tm(0, bd + 8)
    write_tm(0, bd + 12)
    write_tm(0, bd + 16)
    write_tm((1 << 25) | (1 << 18) | (WT_LOCK << 13), bd + 20)
    write_tm(WT_BD, DMA_S2MM_1_START_QUEUE)
    tb: i32 = DMA_BD_BASE + 14 * 32
    write_tm(((TOK_ADDR // 4) << 14) | 1, tb)
    write_tm((1 << 30) | (pkt_id << 19), tb + 4)
    write_tm(0, tb + 8)
    write_tm(0, tb + 12)
    write_tm(0, tb + 16)
    write_tm((1 << 25), tb + 20)
    write_tm(14, DMA_MM2S_1_START_QUEUE)


@aie_kernel
def chain_wt_arm() -> void:
    """Arm S2MM ch1: one weight slot into the fixed L1 buffer; release lock 12.

    ALL BD words are compile-time constants (WT_SLOT_I32 via extra_globals):
    Peano doesn't model the write_tm(start-queue) -> DMA dependency and a
    runtime-valued BD store gets interleaved past the launch — the DMA reads
    a stale BD and wedges (see microbench/ctrl_packet_dma/issue_peano_dma_sched)."""
    bd: i32 = DMA_BD_BASE + WT_BD * 32
    write_tm(((WT_BUF_ADDR // 4) << 14) | WT_SLOT_I32, bd)
    write_tm(0, bd + 4)
    write_tm(0, bd + 8)
    write_tm(0, bd + 12)
    write_tm(0, bd + 16)
    write_tm((1 << 25) | (1 << 18) | (WT_LOCK << 13), bd + 20)
    write_tm(WT_BD, DMA_S2MM_1_START_QUEUE)


@aie_kernel
def chain_wt_arm_nq(dummy: i32) -> void:
    """Empty probe kernel — isolates kernel-call vs BD-write wedge."""
    z: i32 = dummy


@aie_kernel
def chain_wt_wait() -> void:
    """Block until the armed slot landed: acquire-GE 1 then take the token
    (lock stays 0/1 — cumulative counting saturates at 63 and wedges)."""
    lock_acquire(48 + WT_LOCK, -1)


@aie_kernel
def chain_copy_bf16(
    src: ptr[bf16, True],
    dst: ptr[bf16, True],
    ic: i32,
) -> void:
    """Park one x2 tile (8 rows x 12 x ic) in scratch for the rnm epilogue.

    8 rows is all the 1x1 reads; fits the existing chain scratch (no L1
    growth — the conv planes are dead during the epilogue).
    """
    n: i32 = 96 * ic
    i: i32 = 0
    while i < n:
        store_v(dst + i, load_v(src + i, 16))
        i = i + 16


@aie_kernel
def chain_gemm_bf16(
    pa: ptr[bf16, True],
    pb: ptr[bf16, True],
    weight: ptr[bf16, True],
    outp: ptr[bf16, True],
    t: i32,
    slot: i32,
    ic: i32,
) -> void:
    """rnm epilogue: 1x1 GEMM over concat(cur, x2), one 16-oc slot per call.

    pa = cur tile (12-wide rows, 8 valid cols at offset 2) from the patch
    FIFO; pb = x2 tile (same layout) shipped through the wt FIFO; out is
    8x8xic flat (one oc-pass of ic channels), slot picks 16 channels within
    the pass. Weight slot = [2*(ic/8)][64] mmul blocks for oc 0-7 then 8-15,
    + bn 16+16.
    """
    event0()
    px2: ptr[bf16] = pb
    kkh: i32 = ic // 8
    woff_b: i32 = 2 * kkh * 64
    bn_w: ptr[bf16] = weight + 2 * ic * 16
    bn_b: ptr[bf16] = bn_w + 16
    outt: ptr[bf16] = outp + t * (64 * ic) + slot * 16
    sp: i32 = 0
    while sp < 64:
        A0: aie_vector[bf16, 32] = _build_a32_3x3(pa, sp, 8, 1, 12, ic, 0, 0, 2)
        acc_a: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, load_v(weight, 64))
        acc_b: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, load_v(weight + woff_b, 64))
        kk: i32 = 1
        while kk < kkh:
            A: aie_vector[bf16, 32] = _build_a32_3x3(pa, sp, 8, 1, 12, ic, kk, 0, 2)
            acc_a = _mac_4x8x8_bf16(A, load_v(weight + kk * 64, 64), acc_a)
            acc_b = _mac_4x8x8_bf16(A, load_v(weight + woff_b + kk * 64, 64), acc_b)
            kk = kk + 1
        kk = 0
        while kk < kkh:
            A2: aie_vector[bf16, 32] = _build_a32_3x3(px2, sp, 8, 1, 12, ic, kk, 0, 2)
            acc_a = _mac_4x8x8_bf16(A2, load_v(weight + (kkh + kk) * 64, 64), acc_a)
            acc_b = _mac_4x8x8_bf16(A2, load_v(weight + woff_b + (kkh + kk) * 64, 64), acc_b)
            kk = kk + 1
        _store_bn_silu_4x8_rows(acc_to_bf16(acc_a), outt, bn_w, bn_b, sp, 64, ic, 0)
        _store_bn_silu_4x8_rows(acc_to_bf16(acc_b), outt, bn_w, bn_b, sp, 64, ic, 1)
        sp = sp + 4
    event1()


@aie_kernel
def chain_conv2res_bf16(
    scratch: ptr[bf16, True],
    weight: ptr[bf16, True],
    finals_base: ptr[bf16, True],
    arena_base: ptr[bf16, True],
    block: i32,
    t: i32,
    ic: i32,
    grow: i32,
    gcol: i32,
    gbound: i32,
) -> void:
    """Conv2 + BN + SiLU + residual: finals = pair(x) + center(x), one store."""
    event0()
    outp: ptr[bf16] = finals_base + t * (64 * ic)
    arena_in: ptr[bf16] = arena_base + t * (96 * ic)
    kkmax: i32 = (ic // 8) * 9
    bn2_w: ptr[bf16] = weight + 16 * ic * 9
    bn2_b: ptr[bf16] = bn2_w + 16
    woff_b: i32 = kkmax * 64
    sp2: i32 = 0
    while sp2 < 64:
        A02: aie_vector[bf16, 32] = _build_a32_3x3(scratch, sp2, 8, 1, 10, 16, 0, 0, 0)
        acc2a: aie_vector[f32, 32] = _mul_4x8x8_bf16(A02, load_v(weight, 64))
        acc2b: aie_vector[f32, 32] = _mul_4x8x8_bf16(A02, load_v(weight + woff_b, 64))
        kk2: i32 = 1
        while kk2 < kkmax:
            ic_blk2: i32 = kk2 // 9
            kk_in_blk2: i32 = kk2 - ic_blk2 * 9
            mid_plane2: i32 = ic_blk2 // 2
            sub_ic_blk2: i32 = ic_blk2 - mid_plane2 * 2
            kh2: i32 = kk_in_blk2 // 3
            kw2: i32 = kk_in_blk2 - kh2 * 3
            plane2: ptr[bf16] = scratch + mid_plane2 * 1600
            A2: aie_vector[bf16, 32] = _build_a32_3x3(plane2, sp2, 8, 1, 10, 16, sub_ic_blk2, kh2, kw2)
            acc2a = _mac_4x8x8_bf16(A2, load_v(weight + kk2 * 64, 64), acc2a)
            acc2b = _mac_4x8x8_bf16(A2, load_v(weight + woff_b + kk2 * 64, 64), acc2b)
            kk2 = kk2 + 1
        _store_bn_silu_res_4x8_rows(acc_to_bf16(acc2a), outp, bn2_w, bn2_b, sp2, block * 2, arena_in, ic, grow, gcol, gbound)
        _store_bn_silu_res_4x8_rows(acc_to_bf16(acc2b), outp, bn2_w, bn2_b, sp2, block * 2 + 1, arena_in, ic, grow, gcol, gbound)
        sp2 = sp2 + 4
    event1()
