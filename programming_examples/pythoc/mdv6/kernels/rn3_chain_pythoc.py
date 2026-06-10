#!/usr/bin/env python3
# rn3_chain_pythoc.py — chained rn3-pair iteration kernels (one per stage).
#
# Mode-free variants of rn3_pair_vector_stage_bf16: each stage is its own
# @aie_kernel so the per-stage loops carry no mode branches (mode-branched
# variants measured 4-5x slower). Layout differences vs the standalone pair
# kernel:
#   * scratch is a separate 4800 u16 worker-local Buffer (not the out arena);
#   * finals live at offset 0 in a compact 3072 u16 out FIFO;
#   * mask is a separate 400 u16 FIFO elem (input arenas carry no mask).

from __future__ import annotations

from pythoc import ptr, i16, i32, bf16, f32, void
from pythoc.aie import (
    aie_vector, load_v, store_v, vector_add, vector_sub, vector_mul,
    vector_and, vector_cast, vector_extract, broadcast, concat, zeros,
)
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
) -> void:
    """Conv1: 12x12x48 patch t -> 10x10x16 scratch plane `block` (no mask)."""
    event0()
    arena_in: ptr[bf16] = arena_base + t * 4608
    scr: ptr[bf16] = scratch + block * 1600
    bn1_w: ptr[bf16] = weight + 16 * 48 * 9
    bn1_b: ptr[bf16] = bn1_w + 16
    sp: i32 = 0
    while sp < 100:
        A0: aie_vector[bf16, 32] = _build_a32_3x3(arena_in, sp, 10, 1, 12, 48, 0, 0, 0)
        acc_a: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, load_v(weight, 64))
        acc_b: aie_vector[f32, 32] = _mul_4x8x8_bf16(A0, load_v(weight + 3456, 64))
        kk: i32 = 1
        while kk < 54:
            ic_blk: i32 = kk // 9
            kk_in_blk: i32 = kk - ic_blk * 9
            kh: i32 = kk_in_blk // 3
            kw: i32 = kk_in_blk - kh * 3
            A: aie_vector[bf16, 32] = _build_a32_3x3(arena_in, sp, 10, 1, 12, 48, ic_blk, kh, kw)
            acc_a = _mac_4x8x8_bf16(A, load_v(weight + kk * 64, 64), acc_a)
            acc_b = _mac_4x8x8_bf16(A, load_v(weight + 3456 + kk * 64, 64), acc_b)
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
) -> void:
    """Zero conv1 scratch positions outside the 40x40 image domain.

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
        if gr < 0 or gr >= 40 or gc < 0 or gc >= 40:
            store_v(scratch + mz * 16, z16)
            store_v(scratch + 1600 + mz * 16, z16)
            store_v(scratch + 3200 + mz * 16, z16)
        mz = mz + 1
    event1()


@aie_kernel
def chain_conv2_bf16(
    scratch: ptr[bf16, True],
    weight: ptr[bf16, True],
    finals_base: ptr[bf16, True],
    block: i32,
    t: i32,
) -> void:
    """Conv2: scratch planes -> HWC finals (8x8x48), channels block*16."""
    event0()
    outp: ptr[bf16] = finals_base + t * 3072
    # the helper indexes bn at oc_blk*8 over a 48-ch output but the slot's bn
    # arrays only cover its own 16 channels — pre-subtract block*16
    bn2_w: ptr[bf16] = weight + 16 * 48 * 9 - block * 16
    bn2_b: ptr[bf16] = bn2_w + 16
    sp2: i32 = 0
    while sp2 < 64:
        A02: aie_vector[bf16, 32] = _build_a32_3x3(scratch, sp2, 8, 1, 10, 16, 0, 0, 0)
        acc2a: aie_vector[f32, 32] = _mul_4x8x8_bf16(A02, load_v(weight, 64))
        acc2b: aie_vector[f32, 32] = _mul_4x8x8_bf16(A02, load_v(weight + 3456, 64))
        kk2: i32 = 1
        while kk2 < 54:
            ic_blk2: i32 = kk2 // 9
            kk_in_blk2: i32 = kk2 - ic_blk2 * 9
            mid_plane2: i32 = ic_blk2 // 2
            sub_ic_blk2: i32 = ic_blk2 - mid_plane2 * 2
            kh2: i32 = kk_in_blk2 // 3
            kw2: i32 = kk_in_blk2 - kh2 * 3
            plane2: ptr[bf16] = scratch + mid_plane2 * 1600
            A2: aie_vector[bf16, 32] = _build_a32_3x3(plane2, sp2, 8, 1, 10, 16, sub_ic_blk2, kh2, kw2)
            acc2a = _mac_4x8x8_bf16(A2, load_v(weight + kk2 * 64, 64), acc2a)
            acc2b = _mac_4x8x8_bf16(A2, load_v(weight + 3456 + kk2 * 64, 64), acc2b)
            kk2 = kk2 + 1
        _store_bn_silu_4x8_rows(acc_to_bf16(acc2a), outp, bn2_w, bn2_b, sp2, 64, 48, block * 2)
        _store_bn_silu_4x8_rows(acc_to_bf16(acc2b), outp, bn2_w, bn2_b, sp2, 64, 48, block * 2 + 1)
        sp2 = sp2 + 4

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
) -> void:
    """BN + SiLU + residual (out = silu(bn(acc)) + patch center) for 4x8 rows.

    Same math as _store_bn_silu_4x8_rows (oc=48, spatial_out=64) plus the
    8-lane patch-center residual folded in at store time — avoids reading
    finals back after conv stores. oc_blk indexes the FULL 48-channel output;
    the slot's bn arrays only cover its own 16 channels, so bn is indexed by
    the slot-local block (oc_blk % 2 — chain conv2 passes block*2/+1).
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
    off0: i32 = ((r0 + 2) * 12 + (pos0 - r0 * 8 + 2)) * 48 + oc_blk * 8
    pos1: i32 = sp + 1
    r1: i32 = pos1 // 8
    off1: i32 = ((r1 + 2) * 12 + (pos1 - r1 * 8 + 2)) * 48 + oc_blk * 8
    pos2: i32 = sp + 2
    r2: i32 = pos2 // 8
    off2: i32 = ((r2 + 2) * 12 + (pos2 - r2 * 8 + 2)) * 48 + oc_blk * 8
    pos3: i32 = sp + 3
    r3: i32 = pos3 // 8
    off3: i32 = ((r3 + 2) * 12 + (pos3 - r3 * 8 + 2)) * 48 + oc_blk * 8
    c01: aie_vector[bf16, 16] = concat(load_v(arena_in + off0, 8), load_v(arena_in + off1, 8))
    c23: aie_vector[bf16, 16] = concat(load_v(arena_in + off2, 8), load_v(arena_in + off3, 8))
    res32: aie_vector[bf16, 32] = concat(c01, c23)
    sum32: aie_vector[bf16, 32] = vector_add(out32, res32)
    store_v(output + pos0 * 48 + oc_blk * 8, vector_extract(sum32, 0, 8))
    store_v(output + pos1 * 48 + oc_blk * 8, vector_extract(sum32, 8, 8))
    store_v(output + pos2 * 48 + oc_blk * 8, vector_extract(sum32, 16, 8))
    store_v(output + pos3 * 48 + oc_blk * 8, vector_extract(sum32, 24, 8))


@aie_kernel
def chain_conv2res_bf16(
    scratch: ptr[bf16, True],
    weight: ptr[bf16, True],
    finals_base: ptr[bf16, True],
    arena_base: ptr[bf16, True],
    block: i32,
    t: i32,
) -> void:
    """Conv2 + BN + SiLU + residual: finals = pair(x) + center(x), one store."""
    event0()
    outp: ptr[bf16] = finals_base + t * 3072
    arena_in: ptr[bf16] = arena_base + t * 4608
    bn2_w: ptr[bf16] = weight + 16 * 48 * 9
    bn2_b: ptr[bf16] = bn2_w + 16
    sp2: i32 = 0
    while sp2 < 64:
        A02: aie_vector[bf16, 32] = _build_a32_3x3(scratch, sp2, 8, 1, 10, 16, 0, 0, 0)
        acc2a: aie_vector[f32, 32] = _mul_4x8x8_bf16(A02, load_v(weight, 64))
        acc2b: aie_vector[f32, 32] = _mul_4x8x8_bf16(A02, load_v(weight + 3456, 64))
        kk2: i32 = 1
        while kk2 < 54:
            ic_blk2: i32 = kk2 // 9
            kk_in_blk2: i32 = kk2 - ic_blk2 * 9
            mid_plane2: i32 = ic_blk2 // 2
            sub_ic_blk2: i32 = ic_blk2 - mid_plane2 * 2
            kh2: i32 = kk_in_blk2 // 3
            kw2: i32 = kk_in_blk2 - kh2 * 3
            plane2: ptr[bf16] = scratch + mid_plane2 * 1600
            A2: aie_vector[bf16, 32] = _build_a32_3x3(plane2, sp2, 8, 1, 10, 16, sub_ic_blk2, kh2, kw2)
            acc2a = _mac_4x8x8_bf16(A2, load_v(weight + kk2 * 64, 64), acc2a)
            acc2b = _mac_4x8x8_bf16(A2, load_v(weight + 3456 + kk2 * 64, 64), acc2b)
            kk2 = kk2 + 1
        _store_bn_silu_res_4x8_rows(acc_to_bf16(acc2a), outp, bn2_w, bn2_b, sp2, block * 2, arena_in)
        _store_bn_silu_res_4x8_rows(acc_to_bf16(acc2b), outp, bn2_w, bn2_b, sp2, block * 2 + 1, arena_in)
        sp2 = sp2 + 4
    event1()


@aie_kernel
def chain_wtprobe_bf16(
    weight: ptr[bf16, True],
    finals_base: ptr[bf16, True],
    block: i32,
    t: i32,
) -> void:
    """DEBUG: dump first 16 weight elems of the held slot to finals ch block*16."""
    event0()
    finals: ptr[bf16] = finals_base + t * 3072
    p: i32 = 0
    while p < 64:
        w16: aie_vector[bf16, 16] = load_v(weight + p * 108, 16)
        store_v(finals + p * 48 + block * 16, w16)
        p = p + 1
    event1()


@aie_kernel
def chain_residual_bf16(
    arena_base: ptr[bf16, True],
    finals_base: ptr[bf16, True],
    block: i32,
    t: i32,
) -> void:
    """finals(HWC 8x8x48) += center 8x8x48 of the 12x12x48 input patch."""
    event0()
    arena_in: ptr[bf16] = arena_base + t * 4608
    finals: ptr[bf16] = finals_base + t * 3072
    rsp: i32 = 0
    while rsp < 64:
        rrow: i32 = rsp // 8
        rcol: i32 = rsp - rrow * 8
        in_off: i32 = ((rrow + 2) * 12 + (rcol + 2)) * 48
        f_off: i32 = rsp * 48
        ch: i32 = 0
        while ch < 48:
            finals[f_off + ch] = bf16(f32(finals[f_off + ch]) + f32(arena_in[in_off + ch]))
            ch = ch + 1
        rsp = rsp + 1
    event1()
