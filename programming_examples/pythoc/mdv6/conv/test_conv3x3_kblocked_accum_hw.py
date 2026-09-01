#!/usr/bin/env python3
"""Hardware smoke test for conv3x3_kblocked_accum_bf16.

This validates the new vector 3x3 k-blocked primitive before composing it into
an rn3pair fused wrapper. The first test case uses one final k block
(k_start=0, full_ic=k_block=16), so it exercises:

- compact 3x3 packed weight layout [OC/8, K/8, 9, 8ic, 8oc]
- mmul<4,8,8>-style 3x3 accumulation
- final BN+SiLU store path

The multi-k partial-accum path is the next step after this primitive passes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module

KERNELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "build"))


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
    """Pack OIHW [oc,k,3,3] to [OC/8,K/8,9,8ic,8oc] + bn_w + bn_b."""
    oc, k_block, kh, kw = w_oi3x3_f32.shape
    assert kh == 3 and kw == 3
    assert oc % 8 == 0 and k_block % 8 == 0
    w_u16 = f32_to_bf16_u16(w_oi3x3_f32).reshape(oc, k_block, 9)
    w_f = bf16_u16_to_f32(w_u16).reshape(oc, k_block, 9)
    oc_blks = oc // 8
    k_blks = k_block // 8
    blocked = w_f.reshape(oc_blks, 8, k_blks, 8, 9).transpose(0, 2, 4, 3, 1).copy()
    return np.concatenate([
        f32_to_bf16_u16(blocked.reshape(-1)),
        f32_to_bf16_u16(bn_w_f32),
        f32_to_bf16_u16(bn_b_f32),
    ]).astype(np.uint16)


def cpu_conv3x3_bnsilu(inp_f32: np.ndarray, packed_w_u16: np.ndarray, tile_h: int, tile_w: int, k_block: int, oc: int, stride: int) -> np.ndarray:
    """CPU oracle using the same rounded bf16 inputs/weights as hardware."""
    patch_w = (tile_w - 1) * stride + 3
    inp = inp_f32.reshape((tile_h - 1) * stride + 3, patch_w, k_block)
    wt_size = oc * k_block * 9
    packed = bf16_u16_to_f32(packed_w_u16[:wt_size])
    bn_w = bf16_u16_to_f32(packed_w_u16[wt_size:wt_size + oc])
    bn_b = bf16_u16_to_f32(packed_w_u16[wt_size + oc:wt_size + 2 * oc])
    oc_blks = oc // 8
    k_blks = k_block // 8
    blocked = packed.reshape(oc_blks, k_blks, 9, 8, 8)
    # back to OIHW float: [oc_blk,8oc,k_blk,8ic,9] -> [oc,k,3,3]
    w = blocked.transpose(0, 4, 1, 3, 2).reshape(oc, k_block, 3, 3)
    out = np.zeros((tile_h, tile_w, oc), dtype=np.float32)
    for oh in range(tile_h):
        for ow in range(tile_w):
            patch = inp[oh * stride:oh * stride + 3, ow * stride:ow * stride + 3, :]
            for co in range(oc):
                acc = np.float32(0.0)
                for ci in range(k_block):
                    for kh in range(3):
                        for kw in range(3):
                            acc += np.float32(patch[kh, kw, ci] * w[co, ci, kh, kw])
                v = acc * bn_w[co] + bn_b[co]
                out[oh, ow, co] = silu(v)
    return out


def build_module(tile_h=8, tile_w=8, k_block=16, oc=16, stride=1, padding=0, stack_size=4096):
    patch_h = (tile_h - 1) * stride + 3
    patch_w = (tile_w - 1) * stride + 3
    input_size = patch_h * patch_w * k_block
    weight_size = oc * k_block * 9 + 2 * oc
    output_size = tile_h * tile_w * oc

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        "conv3x3_kblocked_accum_bf16",
        os.path.join(KERNELS_DIR, "conv3x3_kblocked_accum_bf16.o"),
        [
            input_ty,
            weight_ty,
            output_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    of_in = ObjectFifo(input_ty, depth=1, name="c3k_in")
    of_wt = ObjectFifo(weight_ty, depth=1, name="c3k_wt")
    of_out = ObjectFifo(output_ty, depth=1, name="c3k_out")

    def core_fn(a, w, c, kern):
        elem_in = a.acquire(1)
        elem_wt = w.acquire(1)
        elem_out = c.acquire(1)
        kern(elem_in, elem_wt, elem_out, tile_h, tile_w, k_block, oc, 0, k_block, stride, padding)
        a.release(1)
        w.release(1)
        c.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_wt.cons(), of_out.prod(), kernel], stack_size=stack_size)
    def sequence(I, W, O, of_in_prod, of_wt_prod, of_out_cons):
        of_in_prod.fill(I, TensorAccessPattern((input_size,), offset=0, sizes=[1, input_size], strides=[0, 1]))
        of_wt_prod.fill(W, TensorAccessPattern((weight_size,), offset=0, sizes=[1, weight_size], strides=[0, 1]))
        of_out_cons.drain(O, TensorAccessPattern((output_size,), offset=0, sizes=[1, output_size], strides=[0, 1]), wait=True)

    rt = Runtime(
        sequence,
        [input_ty, weight_ty, output_ty, of_in.prod(), of_wt.prod(), of_out.cons()],
    )
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


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


def run_kernel(xclbin: Path, insts: Path, tile_h=8, tile_w=8, k_block=16, oc=16, stride=1):
    rng = np.random.default_rng(123)
    patch_h = (tile_h - 1) * stride + 3
    patch_w = (tile_w - 1) * stride + 3
    inp = rng.normal(0, 0.15, size=(patch_h, patch_w, k_block)).astype(np.float32)
    w = rng.normal(0, 0.04, size=(oc, k_block, 3, 3)).astype(np.float32)
    bn_w = rng.normal(1.0, 0.02, size=(oc,)).astype(np.float32)
    bn_b = rng.normal(0.0, 0.01, size=(oc,)).astype(np.float32)
    in_u16 = f32_to_bf16_u16(inp.reshape(-1))
    wt_u16 = pack_3x3_weights_u16(w, bn_w, bn_b)
    exp = cpu_conv3x3_bnsilu(bf16_u16_to_f32(in_u16), wt_u16, tile_h, tile_w, k_block, oc, stride)

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(in_u16, dtype=np.uint16)
    W = iron.tensor(wt_u16, dtype=np.uint16)
    C = iron.zeros(tile_h * tile_w * oc, dtype=np.uint16)
    DefaultNPURuntime.run(handle, [A, W, C])
    got = bf16_u16_to_f32(C.numpy().copy()).reshape(tile_h, tile_w, oc)
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp, rtol=3e-2, atol=3e-2)
    return got



def build_module_two_chunks(tile_h=8, tile_w=8, k_block=16, oc=16, stride=1, padding=0, stack_size=4096):
    patch_h = (tile_h - 1) * stride + 3
    patch_w = (tile_w - 1) * stride + 3
    input_size = patch_h * patch_w * k_block
    weight_size = oc * k_block * 9 + 2 * oc
    output_size = tile_h * tile_w * oc

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        "conv3x3_kblocked_accum_bf16",
        os.path.join(KERNELS_DIR, "conv3x3_kblocked_accum_bf16.o"),
        [input_ty, weight_ty, output_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )
    # Reuse one input FIFO and one weight FIFO sequentially. Four independent
    # input FIFOs exceed NPU2Col1's 2 input DMA channel limit.
    in_fifo = ObjectFifo(input_ty, depth=1, name="c3k_in_seq")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="c3k_wt_seq")
    out = ObjectFifo(output_ty, depth=1, name="c3k_out2")

    def core_fn(a, w, c, kern):
        eout = c.acquire(1)
        ea0 = a.acquire(1)
        ew0 = w.acquire(1)
        kern(ea0, ew0, eout, tile_h, tile_w, k_block, oc, 0, k_block * 2, stride, padding)
        a.release(1)
        w.release(1)
        ea1 = a.acquire(1)
        ew1 = w.acquire(1)
        kern(ea1, ew1, eout, tile_h, tile_w, k_block, oc, k_block, k_block * 2, stride, padding)
        a.release(1)
        w.release(1)
        c.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), out.prod(), kernel], stack_size=stack_size)
    def sequence(I0, I1, W0, W1, O, in_fifo_prod, wt_fifo_prod, out_cons):
        tap_i = TensorAccessPattern((input_size,), offset=0, sizes=[1, input_size], strides=[0, 1])
        tap_w = TensorAccessPattern((weight_size,), offset=0, sizes=[1, weight_size], strides=[0, 1])
        tap_o = TensorAccessPattern((output_size,), offset=0, sizes=[1, output_size], strides=[0, 1])
        in_fifo_prod.fill(I0, tap_i)
        wt_fifo_prod.fill(W0, tap_w)
        in_fifo_prod.fill(I1, tap_i)
        wt_fifo_prod.fill(W1, tap_w)
        out_cons.drain(O, tap_o, wait=True)

    rt = Runtime(
        sequence,
        [input_ty, input_ty, weight_ty, weight_ty, output_ty, in_fifo.prod(), wt_fifo.prod(), out.cons()],
    )
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


def run_kernel_two_chunks(xclbin: Path, insts: Path, tile_h=8, tile_w=8, k_block=16, oc=16, stride=1):
    rng = np.random.default_rng(456)
    patch_h = (tile_h - 1) * stride + 3
    patch_w = (tile_w - 1) * stride + 3
    inp0 = rng.normal(0, 0.15, size=(patch_h, patch_w, k_block)).astype(np.float32)
    inp1 = rng.normal(0, 0.15, size=(patch_h, patch_w, k_block)).astype(np.float32)
    w0 = rng.normal(0, 0.04, size=(oc, k_block, 3, 3)).astype(np.float32)
    w1 = rng.normal(0, 0.04, size=(oc, k_block, 3, 3)).astype(np.float32)
    bn_w = rng.normal(1.0, 0.02, size=(oc,)).astype(np.float32)
    bn_b = rng.normal(0.0, 0.01, size=(oc,)).astype(np.float32)
    in0_u16 = f32_to_bf16_u16(inp0.reshape(-1))
    in1_u16 = f32_to_bf16_u16(inp1.reshape(-1))
    wt0_u16 = pack_3x3_weights_u16(w0, np.ones_like(bn_w), np.zeros_like(bn_b))
    wt1_u16 = pack_3x3_weights_u16(w1, bn_w, bn_b)

    full_in = np.concatenate([
        bf16_u16_to_f32(in0_u16).reshape(patch_h, patch_w, k_block),
        bf16_u16_to_f32(in1_u16).reshape(patch_h, patch_w, k_block),
    ], axis=2)
    full_w = np.concatenate([
        bf16_u16_to_f32(pack_3x3_weights_u16(w0, bn_w, bn_b)[:oc*k_block*9]).reshape(oc//8, k_block//8, 9, 8, 8).transpose(0,4,1,3,2).reshape(oc, k_block, 3, 3),
        bf16_u16_to_f32(pack_3x3_weights_u16(w1, bn_w, bn_b)[:oc*k_block*9]).reshape(oc//8, k_block//8, 9, 8, 8).transpose(0,4,1,3,2).reshape(oc, k_block, 3, 3),
    ], axis=1)
    exp = np.zeros((tile_h, tile_w, oc), dtype=np.float32)
    bn_w_r = bf16_u16_to_f32(f32_to_bf16_u16(bn_w))
    bn_b_r = bf16_u16_to_f32(f32_to_bf16_u16(bn_b))
    for oh in range(tile_h):
        for ow in range(tile_w):
            patch = full_in[oh*stride:oh*stride+3, ow*stride:ow*stride+3, :]
            for co in range(oc):
                acc = np.float32(0.0)
                for ci in range(k_block*2):
                    for kh in range(3):
                        for kw in range(3):
                            acc += np.float32(patch[kh, kw, ci] * full_w[co, ci, kh, kw])
                exp[oh, ow, co] = silu(acc * bn_w_r[co] + bn_b_r[co])

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A0 = iron.tensor(in0_u16, dtype=np.uint16)
    A1 = iron.tensor(in1_u16, dtype=np.uint16)
    W0 = iron.tensor(wt0_u16, dtype=np.uint16)
    W1 = iron.tensor(wt1_u16, dtype=np.uint16)
    C = iron.zeros(tile_h * tile_w * oc, dtype=np.uint16)
    DefaultNPURuntime.run(handle, [A0, A1, W0, W1, C])
    got = bf16_u16_to_f32(C.numpy().copy()).reshape(tile_h, tile_w, oc)
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp, rtol=5e-2, atol=5e-2)
    return got

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tile-h", type=int, default=8)
    p.add_argument("--tile-w", type=int, default=8)
    p.add_argument("--k-block", type=int, default=16)
    p.add_argument("--oc", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--workdir", default="conv/build_conv3x3_kblocked_accum_micro")
    p.add_argument("--chunks", type=int, choices=(1, 2), default=1)
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    assert args.k_block % 8 == 0 and args.oc % 8 == 0
    wd = Path(args.workdir) / f"chunks{args.chunks}_th{args.tile_h}_tw{args.tile_w}_k{args.k_block}_oc{args.oc}_s{args.stride}"
    if args.chunks == 1:
        module = build_module(args.tile_h, args.tile_w, args.k_block, args.oc, args.stride, 0, args.stack_size)
    else:
        module = build_module_two_chunks(args.tile_h, args.tile_w, args.k_block, args.oc, args.stride, 0, args.stack_size)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built conv3x3_kblocked_accum chunks={args.chunks} tile={args.tile_h}x{args.tile_w} k={args.k_block} oc={args.oc} mlir={mlir}")
    if args.build_only:
        return 0
    if args.chunks == 1:
        got = run_kernel(xclbin, insts, args.tile_h, args.tile_w, args.k_block, args.oc, args.stride)
    else:
        got = run_kernel_two_chunks(xclbin, insts, args.tile_h, args.tile_w, args.k_block, args.oc, args.stride)
    print(f"PASS: conv3x3_kblocked_accum chunks={args.chunks} first={got.reshape(-1)[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
