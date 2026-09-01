#!/usr/bin/env python3
"""One-tile vector rn3pair composition smoke test.

This is the first composed vector-fused rn3pair wrapper. It validates that the
existing fast conv3x3 kernel can materialize the conv1 10x10x16 scratch on tile
and that the new conv3x3_kblocked_accum_bf16 primitive can consume that scratch
for conv2 to produce one 8x8x16 output tile.

Scope deliberately stays at one mid block (16 channels) and one output block
(16 channels). Passing this proves the on-tile scratch/dataflow composition
before extending to three mid blocks with partial accumulation.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))

import aie.iron as iron
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KERNELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "build"))
from kernels.rep_elan_bf16_pythoc import (  # noqa: E402
    KERNEL_EXTRA_GLOBALS,
    _MMUL_HELPERS,
    rn3_pair_vector_stage_bf16,
)


def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7FFF) + lsb.astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


def bf16_u16_to_f32(x: np.ndarray) -> np.ndarray:
    u = np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def pack_3x3_weights_u16(w_oi3x3_f32: np.ndarray, bn_w_f32: np.ndarray, bn_b_f32: np.ndarray) -> np.ndarray:
    oc, ic, kh, kw = w_oi3x3_f32.shape
    assert kh == 3 and kw == 3
    assert oc % 8 == 0 and ic % 8 == 0
    w_u16 = f32_to_bf16_u16(w_oi3x3_f32).reshape(oc, ic, 9)
    w_f = bf16_u16_to_f32(w_u16).reshape(oc, ic, 9)
    oc_blks = oc // 8
    ic_blks = ic // 8
    blocked = w_f.reshape(oc_blks, 8, ic_blks, 8, 9).transpose(0, 2, 4, 3, 1).copy()
    return np.concatenate([
        f32_to_bf16_u16(blocked.reshape(-1)),
        f32_to_bf16_u16(bn_w_f32),
        f32_to_bf16_u16(bn_b_f32),
    ]).astype(np.uint16)


def unpack_packed_3x3_weights_f32(packed_u16: np.ndarray, oc: int, ic: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wt_size = oc * ic * 9
    packed = bf16_u16_to_f32(packed_u16[:wt_size])
    bn_w = bf16_u16_to_f32(packed_u16[wt_size:wt_size + oc])
    bn_b = bf16_u16_to_f32(packed_u16[wt_size + oc:wt_size + 2 * oc])
    blocked = packed.reshape(oc // 8, ic // 8, 9, 8, 8)
    w = blocked.transpose(0, 4, 1, 3, 2).reshape(oc, ic, 3, 3)
    return w, bn_w, bn_b


def conv3x3_bnsilu_cpu(inp: np.ndarray, packed_w_u16: np.ndarray, tile_h: int, tile_w: int, ic: int, oc: int) -> np.ndarray:
    w, bn_w, bn_b = unpack_packed_3x3_weights_f32(packed_w_u16, oc, ic)
    out = np.zeros((tile_h, tile_w, oc), dtype=np.float32)
    for oh in range(tile_h):
        for ow in range(tile_w):
            patch = inp[oh:oh + 3, ow:ow + 3, :]
            for co in range(oc):
                acc = np.float32(0.0)
                for ci in range(ic):
                    for kh in range(3):
                        for kw in range(3):
                            acc += np.float32(patch[kh, kw, ci] * w[co, ci, kh, kw])
                out[oh, ow, co] = silu(acc * bn_w[co] + bn_b[co])
    return out


def cpu_oracle(input_u16: np.ndarray, w1_u16: np.ndarray, w2_u16: np.ndarray, ic=48, mid=16, oc=16) -> np.ndarray:
    inp = bf16_u16_to_f32(input_u16).reshape(12, 12, ic)
    mid_f32 = conv3x3_bnsilu_cpu(inp, w1_u16, 10, 10, ic, mid)
    # Hardware conv1 scratch is bf16. Round before conv2.
    mid_bf16_f32 = bf16_u16_to_f32(f32_to_bf16_u16(mid_f32.reshape(-1))).reshape(10, 10, mid)
    return conv3x3_bnsilu_cpu(mid_bf16_f32, w2_u16, 8, 8, mid, oc)


def build_module(ic=48, mid=16, oc=16, stack_size=4096):
    input_size = 12 * 12 * ic
    scratch_size = 10 * 10 * mid
    output_size = 8 * 8 * oc
    w1_size = mid * ic * 9 + 2 * mid
    w2_size = oc * mid * 9 + 2 * oc
    weight_slot_size = max(w1_size, w2_size)

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    scratch_ty = np.ndarray[(scratch_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_slot_size,), np.dtype[np.uint16]]

    # Use a single PythoC object to avoid duplicate helper symbols at link time.
    # Slot buffers are padded to the largest tensor shape (12*12*48 = 6912 ui16).
    # Conv1 writes the first 10*10*16 elements of scratch; conv2 reads those and
    # writes the first 8*8*16 elements of output.
    slot_size = input_size
    slot_ty = np.ndarray[(slot_size,), np.dtype[np.uint16]]
    kernel = PythocKernel(
        "conv3x3_kblocked_accum_bf16",
        os.path.join(KERNELS_DIR, "conv3x3_kblocked_accum_bf16.o"),
        [slot_ty, weight_ty, slot_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    in_fifo = ObjectFifo(slot_ty, depth=1, name="rn3v_in")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3v_wt_seq")
    scratch_buf = Buffer(slot_ty, name="rn3v_scratch_buf")
    out_fifo = ObjectFifo(slot_ty, depth=1, name="rn3v_out")

    def core_fn(a, w, scratch, c, kern):
        ein = a.acquire(1)
        ew1 = w.acquire(1)
        # conv1: 12x12x48 -> 10x10x16, final block so BN+SiLU applies.
        kern(ein, ew1, scratch, 10, 10, ic, mid, 0, ic, 1, 0)
        a.release(1)
        w.release(1)

        ew2 = w.acquire(1)
        eout = c.acquire(1)
        # conv2: 10x10x16 scratch -> 8x8x16 final output.
        kern(scratch, ew2, eout, 8, 8, mid, oc, 0, mid, 1, 0)
        w.release(1)
        c.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), scratch_buf, out_fifo.prod(), kernel], stack_size=stack_size)
    def sequence(I, W1, W2, O, in_fifo_prod, wt_fifo_prod, out_fifo_cons):
        in_fifo_prod.fill(I, TensorAccessPattern((slot_size,), offset=0, sizes=[1, slot_size], strides=[0, 1]))
        wt_fifo_prod.fill(W1, TensorAccessPattern((weight_slot_size,), offset=0, sizes=[1, weight_slot_size], strides=[0, 1]))
        wt_fifo_prod.fill(W2, TensorAccessPattern((weight_slot_size,), offset=0, sizes=[1, weight_slot_size], strides=[0, 1]))
        out_fifo_cons.drain(O, TensorAccessPattern((slot_size,), offset=0, sizes=[1, slot_size], strides=[0, 1]), wait=True)

    rt = Runtime(
        sequence,
        [slot_ty, weight_ty, weight_ty, slot_ty, in_fifo.prod(), wt_fifo.prod(), out_fifo.cons()],
    )
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program(), weight_slot_size, w1_size, w2_size


def compile_module(module, workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    mlir_path = workdir / "kernel.mlir"
    with open(mlir_path, "w", encoding="utf-8") as f:
        print(module, file=f)
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(workdir / "insts.bin"),
        xclbin_path=str(workdir / "final.xclbin"),
        work_dir=str(workdir),
        verbose=False,
    )
    return workdir / "final.xclbin", workdir / "insts.bin", mlir_path


def run_kernel(xclbin: Path, insts: Path, weight_slot_size: int, w1_size: int, w2_size: int, ic=48, mid=16, oc=16):
    rng = np.random.default_rng(789)
    inp = rng.normal(0, 0.12, size=(12, 12, ic)).astype(np.float32)
    w1 = rng.normal(0, 0.035, size=(mid, ic, 3, 3)).astype(np.float32)
    bn1w = rng.normal(1.0, 0.02, size=(mid,)).astype(np.float32)
    bn1b = rng.normal(0.0, 0.01, size=(mid,)).astype(np.float32)
    w2 = rng.normal(0, 0.035, size=(oc, mid, 3, 3)).astype(np.float32)
    bn2w = rng.normal(1.0, 0.02, size=(oc,)).astype(np.float32)
    bn2b = rng.normal(0.0, 0.01, size=(oc,)).astype(np.float32)

    input_u16 = f32_to_bf16_u16(inp.reshape(-1))
    w1_u16 = pack_3x3_weights_u16(w1, bn1w, bn1b)
    w2_u16 = pack_3x3_weights_u16(w2, bn2w, bn2b)
    assert len(w1_u16) == w1_size and len(w2_u16) == w2_size
    w1_slot = np.zeros(weight_slot_size, dtype=np.uint16)
    w2_slot = np.zeros(weight_slot_size, dtype=np.uint16)
    w1_slot[:w1_size] = w1_u16
    w2_slot[:w2_size] = w2_u16

    exp = cpu_oracle(input_u16, w1_u16, w2_u16, ic, mid, oc)

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(input_u16, dtype=np.uint16)
    W1 = iron.tensor(w1_slot, dtype=np.uint16)
    W2 = iron.tensor(w2_slot, dtype=np.uint16)
    C = iron.zeros(12 * 12 * ic, dtype=np.uint16)
    t0 = time.perf_counter()
    DefaultNPURuntime.run(handle, [A, W1, W2, C])
    run_ms = (time.perf_counter() - t0) * 1000
    print(f"run_ms={run_ms:.2f}")
    got = bf16_u16_to_f32(C.numpy().copy()[:8 * 8 * oc]).reshape(8, 8, oc)
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp, rtol=5e-2, atol=5e-2)
    return got



def build_module_three_midblocks(ic=48, mid_block=16, oc=16, stack_size=4096):
    input_size = 12 * 12 * ic
    slot_size = input_size
    scratch_size = 10 * 10 * mid_block
    w1_size = mid_block * ic * 9 + 2 * mid_block
    w2_size = oc * mid_block * 9 + 2 * oc
    weight_slot_size = max(w1_size, w2_size)

    slot_ty = np.ndarray[(slot_size,), np.dtype[np.uint16]]
    scratch_ty = np.ndarray[(scratch_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_slot_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        "conv3x3_kblocked_accum_bf16",
        os.path.join(KERNELS_DIR, "conv3x3_kblocked_accum_bf16.o"),
        [slot_ty, weight_ty, slot_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )
    # Conv1 scratch only needs 1600 elements, but the shared kernel signature uses
    # slot_ty. A full slot buffer keeps type-compatible calls while conv1/conv2
    # only touch the prefix dictated by active tile/ic/oc RTP values.
    scratch_buf = Buffer(slot_ty, name="rn3v3_scratch_buf")
    in_fifo = ObjectFifo(slot_ty, depth=1, name="rn3v3_in")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3v3_wt_seq")
    out_fifo = ObjectFifo(slot_ty, depth=1, name="rn3v3_out")

    def core_fn(a, w, scratch, c, kern):
        ein = a.acquire(1)
        eout = c.acquire(1)
        # block 0
        ew = w.acquire(1)
        kern(ein, ew, scratch, 10, 10, ic, mid_block, 0, ic, 1, 0)
        w.release(1)
        ew = w.acquire(1)
        kern(scratch, ew, eout, 8, 8, mid_block, oc, 0, mid_block * 3, 1, 0)
        w.release(1)
        # block 1
        ew = w.acquire(1)
        kern(ein, ew, scratch, 10, 10, ic, mid_block, 0, ic, 1, 0)
        w.release(1)
        ew = w.acquire(1)
        kern(scratch, ew, eout, 8, 8, mid_block, oc, mid_block, mid_block * 3, 1, 0)
        w.release(1)
        # block 2, final conv2 applies BN+SiLU
        ew = w.acquire(1)
        kern(ein, ew, scratch, 10, 10, ic, mid_block, 0, ic, 1, 0)
        w.release(1)
        ew = w.acquire(1)
        kern(scratch, ew, eout, 8, 8, mid_block, oc, mid_block * 2, mid_block * 3, 1, 0)
        w.release(1)
        a.release(1)
        c.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), scratch_buf, out_fifo.prod(), kernel], stack_size=stack_size)
    weight_all_size = weight_slot_size * 6
    weight_all_ty = np.ndarray[(weight_all_size,), np.dtype[np.uint16]]
    def sequence(I, WALL, O, in_fifo_prod, wt_fifo_prod, out_fifo_cons):
        tap_s = TensorAccessPattern((slot_size,), offset=0, sizes=[1, slot_size], strides=[0, 1])
        in_fifo_prod.fill(I, tap_s)
        for wi in range(6):
            wt_fifo_prod.fill(WALL, TensorAccessPattern((weight_all_size,), offset=wi * weight_slot_size, sizes=[1, weight_slot_size], strides=[0, 1]))
        out_fifo_cons.drain(O, tap_s, wait=True)

    rt = Runtime(
        sequence,
        [slot_ty, weight_all_ty, slot_ty, in_fifo.prod(), wt_fifo.prod(), out_fifo.cons()],
    )
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program(), weight_slot_size, w1_size, w2_size


def run_kernel_three_midblocks(xclbin: Path, insts: Path, weight_slot_size: int, w1_size: int, w2_size: int, ic=48, mid_block=16, oc=16):
    rng = np.random.default_rng(987)
    inp = rng.normal(0, 0.12, size=(12, 12, ic)).astype(np.float32)
    input_u16 = f32_to_bf16_u16(inp.reshape(-1))
    w1_slots = []
    w2_slots = []
    mid_parts = []
    w2_chunks = []
    bn2w = rng.normal(1.0, 0.02, size=(oc,)).astype(np.float32)
    bn2b = rng.normal(0.0, 0.01, size=(oc,)).astype(np.float32)
    for mb in range(3):
        w1 = rng.normal(0, 0.035, size=(mid_block, ic, 3, 3)).astype(np.float32)
        bn1w = rng.normal(1.0, 0.02, size=(mid_block,)).astype(np.float32)
        bn1b = rng.normal(0.0, 0.01, size=(mid_block,)).astype(np.float32)
        w1_u16 = pack_3x3_weights_u16(w1, bn1w, bn1b)
        w1_slot = np.zeros(weight_slot_size, dtype=np.uint16)
        w1_slot[:w1_size] = w1_u16
        w1_slots.append(w1_slot)
        mid_f32 = conv3x3_bnsilu_cpu(bf16_u16_to_f32(input_u16).reshape(12, 12, ic), w1_u16, 10, 10, ic, mid_block)
        mid_parts.append(bf16_u16_to_f32(f32_to_bf16_u16(mid_f32.reshape(-1))).reshape(10, 10, mid_block))

        w2 = rng.normal(0, 0.035, size=(oc, mid_block, 3, 3)).astype(np.float32)
        # BN is only used by the final k-block in hardware, but pack identical BN
        # into every chunk so each slot has canonical layout.
        w2_u16 = pack_3x3_weights_u16(w2, bn2w, bn2b)
        w2_slot = np.zeros(weight_slot_size, dtype=np.uint16)
        w2_slot[:w2_size] = w2_u16
        w2_slots.append(w2_slot)
        w2_chunks.append(unpack_packed_3x3_weights_f32(w2_u16, oc, mid_block)[0])

    mid_full = np.concatenate(mid_parts, axis=2)
    w2_full = np.concatenate(w2_chunks, axis=1)
    bn2w_r = bf16_u16_to_f32(f32_to_bf16_u16(bn2w))
    bn2b_r = bf16_u16_to_f32(f32_to_bf16_u16(bn2b))
    exp = np.zeros((8, 8, oc), dtype=np.float32)
    for oh in range(8):
        for ow in range(8):
            patch = mid_full[oh:oh + 3, ow:ow + 3, :]
            for co in range(oc):
                acc = np.float32(0.0)
                for ci in range(mid_block * 3):
                    for kh in range(3):
                        for kw in range(3):
                            acc += np.float32(patch[kh, kw, ci] * w2_full[co, ci, kh, kw])
                exp[oh, ow, co] = silu(acc * bn2w_r[co] + bn2b_r[co])

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(input_u16, dtype=np.uint16)
    weight_all = np.zeros(weight_slot_size * 6, dtype=np.uint16)
    for mb in range(3):
        weight_all[(2 * mb + 0) * weight_slot_size:(2 * mb + 1) * weight_slot_size] = w1_slots[mb]
        weight_all[(2 * mb + 1) * weight_slot_size:(2 * mb + 2) * weight_slot_size] = w2_slots[mb]
    WALL = iron.tensor(weight_all, dtype=np.uint16)
    C = iron.zeros(12 * 12 * ic, dtype=np.uint16)
    t0 = time.perf_counter()
    DefaultNPURuntime.run(handle, [A, WALL, C])
    run_ms = (time.perf_counter() - t0) * 1000
    print(f"run_ms={run_ms:.2f}")
    got = bf16_u16_to_f32(C.numpy().copy()[:8 * 8 * oc]).reshape(8, 8, oc)
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp, rtol=6e-2, atol=6e-2)
    return got


def build_module_three_ocblocks(ic=48, mid_block=16, oc_block=16, n_ocblocks=3, stack_size=4096):
    input_size = 12 * 12 * ic
    slot_size = input_size
    w1_size = mid_block * ic * 9 + 2 * mid_block
    w2_size = oc_block * mid_block * 9 + 2 * oc_block
    weight_slot_size = max(w1_size, w2_size)

    slot_ty = np.ndarray[(slot_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_slot_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        "conv3x3_kblocked_accum_bf16",
        os.path.join(KERNELS_DIR, "conv3x3_kblocked_accum_bf16.o"),
        [slot_ty, weight_ty, slot_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )
    scratch_buf = Buffer(slot_ty, name="rn3v_oc_scratch_buf")
    in_fifo = ObjectFifo(slot_ty, depth=1, name="rn3v_oc_in")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3v_oc_wt_seq")
    out_fifo = ObjectFifo(slot_ty, depth=1, name="rn3v_oc_out")

    def core_fn(a, w, scratch, c, kern):
        ein = a.acquire(1)
        for _ob in range(n_ocblocks):
            eout = c.acquire(1)
            for mb in range(3):
                ew = w.acquire(1)
                kern(ein, ew, scratch, 10, 10, ic, mid_block, 0, ic, 1, 0)
                w.release(1)
                ew = w.acquire(1)
                kern(scratch, ew, eout, 8, 8, mid_block, oc_block, mb * mid_block, mid_block * 3, 1, 0)
                w.release(1)
            c.release(1)
        a.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), scratch_buf, out_fifo.prod(), kernel], stack_size=stack_size)
    n_weight_slots = n_ocblocks * 6
    weight_all_size = weight_slot_size * n_weight_slots
    out_all_size = slot_size * n_ocblocks
    weight_all_ty = np.ndarray[(weight_all_size,), np.dtype[np.uint16]]
    out_all_ty = np.ndarray[(out_all_size,), np.dtype[np.uint16]]
    def sequence(I, WALL, OALL, in_fifo_prod, wt_fifo_prod, out_fifo_cons):
        tap_s = TensorAccessPattern((slot_size,), offset=0, sizes=[1, slot_size], strides=[0, 1])
        in_fifo_prod.fill(I, tap_s)
        # One repeated input-side DMA task feeds all sequential weight slots.
        # Repeated output drains are unsafe in this tree, but repeated input-side
        # fills have been validated by the repeat-DMA microtest.
        wt_fifo_prod.fill(WALL, TensorAccessPattern((weight_all_size,), offset=0, sizes=[n_weight_slots, weight_slot_size], strides=[weight_slot_size, 1]))
        for ob in range(n_ocblocks):
            out_fifo_cons.drain(OALL, TensorAccessPattern((out_all_size,), offset=ob * slot_size, sizes=[1, slot_size], strides=[0, 1]), wait=ob == n_ocblocks - 1)

    rt = Runtime(
        sequence,
        [slot_ty, weight_all_ty, out_all_ty, in_fifo.prod(), wt_fifo.prod(), out_fifo.cons()],
    )
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program(), weight_slot_size, w1_size, w2_size


def run_kernel_three_ocblocks(xclbin: Path, insts: Path, weight_slot_size: int, w1_size: int, w2_size: int, ic=48, mid_block=16, oc_block=16, n_ocblocks=3):
    rng = np.random.default_rng(1357)
    inp = rng.normal(0, 0.12, size=(12, 12, ic)).astype(np.float32)
    input_u16 = f32_to_bf16_u16(inp.reshape(-1))
    inp_f32 = bf16_u16_to_f32(input_u16).reshape(12, 12, ic)

    # Conv1 is shared across output OC blocks in the real rn3 pair. The current
    # one-tile wrapper recomputes the same three scratch chunks for each output
    # block to keep L1 state and output placement simple.
    w1_slots = []
    mid_parts = []
    for mb in range(3):
        w1 = rng.normal(0, 0.035, size=(mid_block, ic, 3, 3)).astype(np.float32)
        bn1w = rng.normal(1.0, 0.02, size=(mid_block,)).astype(np.float32)
        bn1b = rng.normal(0.0, 0.01, size=(mid_block,)).astype(np.float32)
        w1_u16 = pack_3x3_weights_u16(w1, bn1w, bn1b)
        w1_slot = np.zeros(weight_slot_size, dtype=np.uint16)
        w1_slot[:w1_size] = w1_u16
        w1_slots.append(w1_slot)
        mid_f32 = conv3x3_bnsilu_cpu(inp_f32, w1_u16, 10, 10, ic, mid_block)
        mid_parts.append(bf16_u16_to_f32(f32_to_bf16_u16(mid_f32.reshape(-1))).reshape(10, 10, mid_block))
    mid_full = np.concatenate(mid_parts, axis=2)

    w2_slots_by_ob = []
    exp_blocks = []
    for ob in range(n_ocblocks):
        bn2w = rng.normal(1.0, 0.02, size=(oc_block,)).astype(np.float32)
        bn2b = rng.normal(0.0, 0.01, size=(oc_block,)).astype(np.float32)
        w2_chunks = []
        slots = []
        for mb in range(3):
            w2 = rng.normal(0, 0.035, size=(oc_block, mid_block, 3, 3)).astype(np.float32)
            w2_u16 = pack_3x3_weights_u16(w2, bn2w, bn2b)
            w2_slot = np.zeros(weight_slot_size, dtype=np.uint16)
            w2_slot[:w2_size] = w2_u16
            slots.append(w2_slot)
            w2_chunks.append(unpack_packed_3x3_weights_f32(w2_u16, oc_block, mid_block)[0])
        w2_slots_by_ob.append(slots)
        w2_full = np.concatenate(w2_chunks, axis=1)
        bn2w_r = bf16_u16_to_f32(f32_to_bf16_u16(bn2w))
        bn2b_r = bf16_u16_to_f32(f32_to_bf16_u16(bn2b))
        exp = np.zeros((8, 8, oc_block), dtype=np.float32)
        for oh in range(8):
            for ow in range(8):
                patch = mid_full[oh:oh + 3, ow:ow + 3, :]
                for co in range(oc_block):
                    acc = np.float32(0.0)
                    for ci in range(mid_block * 3):
                        for kh in range(3):
                            for kw in range(3):
                                acc += np.float32(patch[kh, kw, ci] * w2_full[co, ci, kh, kw])
                    exp[oh, ow, co] = silu(acc * bn2w_r[co] + bn2b_r[co])
        exp_blocks.append(exp)
    exp_full = np.concatenate(exp_blocks, axis=2)

    n_weight_slots = n_ocblocks * 6
    weight_all = np.zeros(weight_slot_size * n_weight_slots, dtype=np.uint16)
    wi = 0
    for ob in range(n_ocblocks):
        for mb in range(3):
            weight_all[wi * weight_slot_size:(wi + 1) * weight_slot_size] = w1_slots[mb]
            wi += 1
            weight_all[wi * weight_slot_size:(wi + 1) * weight_slot_size] = w2_slots_by_ob[ob][mb]
            wi += 1

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(input_u16, dtype=np.uint16)
    WALL = iron.tensor(weight_all, dtype=np.uint16)
    C = iron.zeros((12 * 12 * ic) * n_ocblocks, dtype=np.uint16)
    t0 = time.perf_counter()
    DefaultNPURuntime.run(handle, [A, WALL, C])
    run_ms = (time.perf_counter() - t0) * 1000
    print(f"run_ms={run_ms:.2f}")
    raw = C.numpy().copy()
    got_blocks = []
    slot_size = 12 * 12 * ic
    for ob in range(n_ocblocks):
        got_blocks.append(bf16_u16_to_f32(raw[ob * slot_size:ob * slot_size + 8 * 8 * oc_block]).reshape(8, 8, oc_block))
    got = np.concatenate(got_blocks, axis=2)
    max_abs = float(np.max(np.abs(got - exp_full)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp_full, rtol=7e-2, atol=7e-2)
    return got


def build_module_shared_conv1(stack_size=4096):
    arena_size = 4800 + 3 * 8 * 8 * 16
    weight_slot_size = 16 * 48 * 9 + 2 * 16
    w1_size = weight_slot_size
    w2_size = 16 * 16 * 9 + 2 * 16
    n_weight_slots = 12

    arena_ty = np.ndarray[(arena_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_slot_size,), np.dtype[np.uint16]]
    kernel = PythocKernel(
        rn3_pair_vector_stage_bf16,
        [arena_ty, weight_ty, arena_ty, np.int32, np.int32, np.int32],
        extra_globals=KERNEL_EXTRA_GLOBALS,
        helpers=_MMUL_HELPERS,
    )
    in_fifo = ObjectFifo(arena_ty, depth=1, name="rn3vs_in")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3vs_wt_seq")
    out_fifo = ObjectFifo(arena_ty, depth=1, name="rn3vs_arena")

    def core_fn(a, w, c, kern):
        ein = a.acquire(1)
        eout = c.acquire(1)
        mb = 0
        while mb < 3:
            ew = w.acquire(1)
            kern(ein, ew, eout, 0, mb, 0)
            w.release(1)
            mb = mb + 1
        a.release(1)

        ob = 0
        while ob < 3:
            mb2 = 0
            while mb2 < 3:
                ew = w.acquire(1)
                kern(eout, ew, eout, 1, mb2, ob)
                w.release(1)
                mb2 = mb2 + 1
            ob = ob + 1
        c.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), out_fifo.prod(), kernel], stack_size=stack_size)
    weight_all_size = weight_slot_size * n_weight_slots
    weight_all_ty = np.ndarray[(weight_all_size,), np.dtype[np.uint16]]
    def sequence(I, WALL, O, in_fifo_prod, wt_fifo_prod, out_fifo_cons):
        in_fifo_prod.fill(I, TensorAccessPattern((arena_size,), offset=0, sizes=[1, arena_size], strides=[0, 1]))
        wt_fifo_prod.fill(WALL, TensorAccessPattern((weight_all_size,), offset=0, sizes=[n_weight_slots, weight_slot_size], strides=[weight_slot_size, 1]))
        out_fifo_cons.drain(O, TensorAccessPattern((arena_size,), offset=0, sizes=[1, arena_size], strides=[0, 1]), wait=True)

    rt = Runtime(
        sequence,
        [arena_ty, weight_all_ty, arena_ty, in_fifo.prod(), wt_fifo.prod(), out_fifo.cons()],
    )
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program(), weight_slot_size, w1_size, w2_size


def run_kernel_shared_conv1(xclbin: Path, insts: Path, weight_slot_size: int, w1_size: int, w2_size: int):
    ic = 48
    mid_block = 16
    oc_block = 16
    n_ocblocks = 3
    arena_size = 4800 + 3 * 8 * 8 * 16
    final_base = 4800
    rng = np.random.default_rng(24680)
    inp = rng.normal(0, 0.12, size=(12, 12, ic)).astype(np.float32)
    input_u16 = f32_to_bf16_u16(inp.reshape(-1))
    inp_f32 = bf16_u16_to_f32(input_u16).reshape(12, 12, ic)
    input_arena = np.zeros(arena_size, dtype=np.uint16)
    input_arena[:12 * 12 * ic] = input_u16

    w1_slots = []
    mid_parts = []
    for mb in range(3):
        w1 = rng.normal(0, 0.035, size=(mid_block, ic, 3, 3)).astype(np.float32)
        bn1w = rng.normal(1.0, 0.02, size=(mid_block,)).astype(np.float32)
        bn1b = rng.normal(0.0, 0.01, size=(mid_block,)).astype(np.float32)
        w1_u16 = pack_3x3_weights_u16(w1, bn1w, bn1b)
        assert len(w1_u16) == w1_size
        w1_slot = np.zeros(weight_slot_size, dtype=np.uint16)
        w1_slot[:w1_size] = w1_u16
        w1_slots.append(w1_slot)
        mid_f32 = conv3x3_bnsilu_cpu(inp_f32, w1_u16, 10, 10, ic, mid_block)
        mid_parts.append(bf16_u16_to_f32(f32_to_bf16_u16(mid_f32.reshape(-1))).reshape(10, 10, mid_block))
    mid_full = np.concatenate(mid_parts, axis=2)

    w2_slots_by_ob = []
    exp_blocks = []
    for ob in range(n_ocblocks):
        bn2w = rng.normal(1.0, 0.02, size=(oc_block,)).astype(np.float32)
        bn2b = rng.normal(0.0, 0.01, size=(oc_block,)).astype(np.float32)
        w2_chunks = []
        slots = []
        for mb in range(3):
            w2 = rng.normal(0, 0.035, size=(oc_block, mid_block, 3, 3)).astype(np.float32)
            w2_u16 = pack_3x3_weights_u16(w2, bn2w, bn2b)
            assert len(w2_u16) == w2_size
            w2_slot = np.zeros(weight_slot_size, dtype=np.uint16)
            w2_slot[:w2_size] = w2_u16
            slots.append(w2_slot)
            w2_chunks.append(unpack_packed_3x3_weights_f32(w2_u16, oc_block, mid_block)[0])
        w2_slots_by_ob.append(slots)
        w2_full = np.concatenate(w2_chunks, axis=1)
        bn2w_r = bf16_u16_to_f32(f32_to_bf16_u16(bn2w))
        bn2b_r = bf16_u16_to_f32(f32_to_bf16_u16(bn2b))
        exp = np.zeros((8, 8, oc_block), dtype=np.float32)
        for oh in range(8):
            for ow in range(8):
                patch = mid_full[oh:oh + 3, ow:ow + 3, :]
                for co in range(oc_block):
                    acc = np.float32(0.0)
                    for ci in range(mid_block * 3):
                        for kh in range(3):
                            for kw in range(3):
                                acc += np.float32(patch[kh, kw, ci] * w2_full[co, ci, kh, kw])
                    exp[oh, ow, co] = silu(acc * bn2w_r[co] + bn2b_r[co])
        exp_blocks.append(exp)
    exp_full = np.concatenate(exp_blocks, axis=2)

    weight_all = np.zeros(weight_slot_size * 12, dtype=np.uint16)
    wi = 0
    for mb in range(3):
        weight_all[wi * weight_slot_size:(wi + 1) * weight_slot_size] = w1_slots[mb]
        wi += 1
    for ob in range(n_ocblocks):
        for mb in range(3):
            weight_all[wi * weight_slot_size:(wi + 1) * weight_slot_size] = w2_slots_by_ob[ob][mb]
            wi += 1

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(input_arena, dtype=np.uint16)
    WALL = iron.tensor(weight_all, dtype=np.uint16)
    C = iron.zeros(arena_size, dtype=np.uint16)
    t0 = time.perf_counter()
    DefaultNPURuntime.run(handle, [A, WALL, C])
    run_ms = (time.perf_counter() - t0) * 1000
    print(f"run_ms={run_ms:.2f}")
    raw = C.numpy().copy()
    got_blocks = []
    for ob in range(n_ocblocks):
        got_blocks.append(bf16_u16_to_f32(raw[final_base + ob * 1024:final_base + (ob + 1) * 1024]).reshape(8, 8, oc_block))
    got = np.concatenate(got_blocks, axis=2)
    max_abs = float(np.max(np.abs(got - exp_full)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp_full, rtol=7e-2, atol=7e-2)
    return got

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ic", type=int, default=48)
    p.add_argument("--mid", type=int, default=16)
    p.add_argument("--oc", type=int, default=16)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--workdir", default="conv/build_rn3_pair_vector_oneblock")
    p.add_argument("--mid-blocks", type=int, choices=(1, 3), default=1)
    p.add_argument("--oc-blocks", type=int, choices=(1, 3), default=1)
    p.add_argument("--shared-conv1", action="store_true")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    assert args.ic % 8 == 0 and args.mid % 8 == 0 and args.oc % 8 == 0
    if args.oc_blocks == 3:
        assert args.mid_blocks == 3, "--oc-blocks 3 requires --mid-blocks 3"
    if args.shared_conv1:
        assert args.mid_blocks == 3 and args.oc_blocks == 3 and args.ic == 48 and args.mid == 16 and args.oc == 16
    wd = Path(args.workdir) / f"mb{args.mid_blocks}_ob{args.oc_blocks}_shared{int(args.shared_conv1)}_ic{args.ic}_mid{args.mid}_oc{args.oc}_st{args.stack_size}"
    if args.shared_conv1:
        module, weight_slot_size, w1_size, w2_size = build_module_shared_conv1(args.stack_size)
    elif args.oc_blocks == 3:
        module, weight_slot_size, w1_size, w2_size = build_module_three_ocblocks(args.ic, args.mid, args.oc, args.oc_blocks, args.stack_size)
    elif args.mid_blocks == 1:
        module, weight_slot_size, w1_size, w2_size = build_module(args.ic, args.mid, args.oc, args.stack_size)
    else:
        module, weight_slot_size, w1_size, w2_size = build_module_three_midblocks(args.ic, args.mid, args.oc, args.stack_size)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built rn3_vector mid_blocks={args.mid_blocks} oc_blocks={args.oc_blocks} shared_conv1={args.shared_conv1} ic={args.ic} mid={args.mid} oc={args.oc} weight_slot={weight_slot_size} mlir={mlir}")
    if args.build_only:
        return 0
    if args.shared_conv1:
        got = run_kernel_shared_conv1(xclbin, insts, weight_slot_size, w1_size, w2_size)
    elif args.oc_blocks == 3:
        got = run_kernel_three_ocblocks(xclbin, insts, weight_slot_size, w1_size, w2_size, args.ic, args.mid, args.oc, args.oc_blocks)
    elif args.mid_blocks == 1:
        got = run_kernel(xclbin, insts, weight_slot_size, w1_size, w2_size, args.ic, args.mid, args.oc)
    else:
        got = run_kernel_three_midblocks(xclbin, insts, weight_slot_size, w1_size, w2_size, args.ic, args.mid, args.oc)
    print(f"PASS: rn3_vector mid_blocks={args.mid_blocks} oc_blocks={args.oc_blocks} shared_conv1={args.shared_conv1} shape={got.shape} first={got.reshape(-1)[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
