#!/usr/bin/env python3
"""Row-band shared-intermediate rn3pair hardware microtest.

This tests a lower-pressure combined conv1+conv2 shared-intermediate design:
for each final output row, keep array[f32,32] accumulators live while
reusing only array[f32,30] for the 3x10 conv1 midplane band needed by that row.
It is slower than the desired full-shared kernel because it recomputes the
midplane per output row, but it directly checks whether reducing accumulator
live state avoids the NaN seen with array[f32,256] + array[f32,100].
"""

import argparse
import subprocess as _subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import PythocKernel, aie_kernel
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module
from pythoc import ptr, i32, bf16, f32, void
from pythoc.builtin_entities import array


_orig_subprocess_run = _subprocess.run


def _patched_subprocess_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args")
    if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and "opt" in str(cmd[0]):
        new_cmd = list(cmd)
        new_cmd[1:1] = ["-vectorize-loops=false", "-vectorize-slp=false"]
        if args:
            args = (new_cmd,) + args[1:]
        else:
            kwargs["args"] = new_cmd
    return _orig_subprocess_run(*args, **kwargs)


if getattr(_subprocess.run, "__name__", "") != "_patched_subprocess_run":
    _subprocess.run = _patched_subprocess_run


@aie_kernel
def rn3_row_band_shared_bf16(input_patch: ptr[bf16, True], weights: ptr[bf16, True], output: ptr[bf16, True], ic: i32, mid: i32) -> void:
    # Full rn3 weight layout: w1[mid,ic,3,3], bn1_w[mid], bn1_b[mid],
    # w2[ocb=4,mid,3,3], bn2_w[4], bn2_b[4].
    # For one final output row, conv2 only needs intermediate rows r..r+2
    # and columns 0..9. Store just that 3x10 band instead of the full 10x10.
    midband: array[f32, 30] = array[f32, 30]()
    accrow: array[f32, 32] = array[f32, 32]()
    patch_w: i32 = 12
    w1_size: i32 = mid * ic * 9
    bn1_w_off: i32 = w1_size
    bn1_b_off: i32 = bn1_w_off + mid
    w2_off: i32 = bn1_b_off + mid
    bn2_w_off: i32 = w2_off + 4 * mid * 9
    bn2_b_off: i32 = bn2_w_off + 4

    r: i32 = 0
    while r < 8:
        z: i32 = 0
        while z < 32:
            accrow[z] = 0.0
            z = z + 1

        o1: i32 = 0
        while o1 < mid:
            # Compute only the 3x10 conv1 intermediate band needed for final row r.
            br: i32 = 0
            while br < 3:
                mc: i32 = 0
                while mc < 10:
                    acc1: f32 = 0.0
                    kh1: i32 = 0
                    while kh1 < 3:
                        kw1: i32 = 0
                        while kw1 < 3:
                            i: i32 = 0
                            while i < ic:
                                in_idx: i32 = ((r + br + kh1) * patch_w + (mc + kw1)) * ic + i
                                w1_idx: i32 = ((o1 * ic + i) * 3 + kh1) * 3 + kw1
                                acc1 = acc1 + f32(input_patch[in_idx]) * f32(weights[w1_idx])
                                i = i + 1
                            kw1 = kw1 + 1
                        kh1 = kh1 + 1
                    x1: f32 = acc1 * f32(weights[bn1_w_off + o1]) + f32(weights[bn1_b_off + o1])
                    ax1: f32 = x1
                    if ax1 < 0.0:
                        ax1 = -x1
                    midband[br * 10 + mc] = x1 * (0.5 + x1 / (2.0 + 2.0 * ax1))
                    mc = mc + 1
                br = br + 1

            c: i32 = 0
            while c < 8:
                kh2: i32 = 0
                while kh2 < 3:
                    kw2: i32 = 0
                    while kw2 < 3:
                        y1: f32 = midband[kh2 * 10 + (c + kw2)]
                        w2_base: i32 = w2_off + ((o1 * 3 + kh2) * 3 + kw2)
                        aidx: i32 = c * 4
                        accrow[aidx] = accrow[aidx] + y1 * f32(weights[w2_base])
                        accrow[aidx + 1] = accrow[aidx + 1] + y1 * f32(weights[w2_base + mid * 9])
                        accrow[aidx + 2] = accrow[aidx + 2] + y1 * f32(weights[w2_base + 2 * mid * 9])
                        accrow[aidx + 3] = accrow[aidx + 3] + y1 * f32(weights[w2_base + 3 * mid * 9])
                        kw2 = kw2 + 1
                    kh2 = kh2 + 1
                c = c + 1
            o1 = o1 + 1

        c2: i32 = 0
        while c2 < 8:
            o2: i32 = 0
            while o2 < 4:
                idx: i32 = c2 * 4 + o2
                x2: f32 = accrow[idx] * f32(weights[bn2_w_off + o2]) + f32(weights[bn2_b_off + o2])
                ax2: f32 = x2
                if ax2 < 0.0:
                    ax2 = -x2
                output[(r * 8 + c2) * 4 + o2] = bf16(x2 * (0.5 + x2 / (2.0 + 2.0 * ax2)))
                o2 = o2 + 1
            c2 = c2 + 1
        r = r + 1

def f32_to_bf16_u16(a):
    t = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).cpu().numpy().copy()


def bf16_u16_to_f32(a):
    return torch.from_numpy(np.asarray(a, dtype=np.uint16).copy()).view(torch.bfloat16).float().numpy()


def silu_approx(x):
    ax = np.abs(x)
    return x * (0.5 + x / (2.0 + 2.0 * ax))


def cpu_oracle(input_patch, weights, ic=48, mid=48):
    w1_size = mid * ic * 9
    bn1_w_off = w1_size
    bn1_b_off = bn1_w_off + mid
    w2_off = bn1_b_off + mid
    bn2_w_off = w2_off + 4 * mid * 9
    bn2_b_off = bn2_w_off + 4
    out = np.zeros((8, 8, 4), dtype=np.float32)
    flat = input_patch.reshape(-1)
    for r in range(8):
        for c in range(8):
            for o2 in range(4):
                acc2 = np.float32(0.0)
                for kh2 in range(3):
                    for kw2 in range(3):
                        mr = r + kh2
                        mc = c + kw2
                        for o1 in range(mid):
                            acc1 = np.float32(0.0)
                            for kh1 in range(3):
                                for kw1 in range(3):
                                    for i in range(ic):
                                        in_idx = ((mr + kh1) * 12 + (mc + kw1)) * ic + i
                                        w1_idx = ((o1 * ic + i) * 3 + kh1) * 3 + kw1
                                        acc1 += flat[in_idx] * weights[w1_idx]
                            x1 = acc1 * weights[bn1_w_off + o1] + weights[bn1_b_off + o1]
                            y1 = silu_approx(x1)
                            w2_idx = w2_off + ((o2 * mid + o1) * 3 + kh2) * 3 + kw2
                            acc2 += y1 * weights[w2_idx]
                x2 = acc2 * weights[bn2_w_off + o2] + weights[bn2_b_off + o2]
                out[r, c, o2] = silu_approx(x2)
    return out


def build_module(ic=48, mid=48, stack_size=4096):
    input_ty = np.ndarray[(12 * 12 * ic,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(mid * ic * 9 + 2 * mid + 4 * mid * 9 + 8,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(256,), np.dtype[np.uint16]]
    kernel = PythocKernel(
        rn3_row_band_shared_bf16,
        [input_ty, weight_ty, output_ty, np.int32, np.int32],
        extra_globals={"array": array},
    )
    in_fifo = ObjectFifo(input_ty, depth=1, name="rn3_band_in")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3_band_wt")
    out_fifo = ObjectFifo(output_ty, depth=1, name="rn3_band_out")

    def core_fn(a, w, c, kern):
        ea = a.acquire(1)
        ew = w.acquire(1)
        ec = c.acquire(1)
        kern(ea, ew, ec, ic, mid)
        a.release(1)
        w.release(1)
        c.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), out_fifo.prod(), kernel], stack_size=stack_size)
    wt_len = mid * ic * 9 + 2 * mid + 4 * mid * 9 + 8
    def sequence(A, W, C, in_fifo_prod, wt_fifo_prod, out_fifo_cons):
        in_fifo_prod.fill(A, TensorAccessPattern((12 * 12 * ic,), offset=0, sizes=[1, 12 * 12 * ic], strides=[0, 1]))
        wt_fifo_prod.fill(W, TensorAccessPattern((wt_len,), offset=0, sizes=[1, wt_len], strides=[0, 1]))
        out_fifo_cons.drain(C, TensorAccessPattern((256,), offset=0, sizes=[1, 256], strides=[0, 1]), wait=True)

    rt = Runtime(
        sequence,
        [input_ty, weight_ty, output_ty, in_fifo.prod(), wt_fifo.prod(), out_fifo.cons()],
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


def run_kernel(xclbin: Path, insts: Path, ic=48, mid=48):
    rng = np.random.default_rng(17)
    inp = rng.normal(0, 0.15, size=(12, 12, ic)).astype(np.float32)
    w1 = rng.normal(0, 0.05, size=(mid, ic, 3, 3)).astype(np.float32)
    bn1w = rng.normal(1.0, 0.02, size=(mid,)).astype(np.float32)
    bn1b = rng.normal(0.0, 0.01, size=(mid,)).astype(np.float32)
    w2 = rng.normal(0, 0.05, size=(4, mid, 3, 3)).astype(np.float32)
    bn2w = rng.normal(1.0, 0.02, size=(4,)).astype(np.float32)
    bn2b = rng.normal(0.0, 0.01, size=(4,)).astype(np.float32)
    weights = np.concatenate([w1.reshape(-1), bn1w, bn1b, w2.reshape(-1), bn2w, bn2b]).astype(np.float32)
    in_u16 = f32_to_bf16_u16(inp.reshape(-1))
    wt_u16 = f32_to_bf16_u16(weights)
    exp = cpu_oracle(bf16_u16_to_f32(in_u16).reshape(inp.shape), bf16_u16_to_f32(wt_u16), ic=ic, mid=mid)

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(in_u16, dtype=np.uint16)
    W = iron.tensor(wt_u16, dtype=np.uint16)
    C = iron.zeros(256, dtype=np.uint16)
    DefaultNPURuntime.run(handle, [A, W, C])
    got = bf16_u16_to_f32(C.numpy().copy()).reshape(8, 8, 4)
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    np.testing.assert_allclose(got, exp, rtol=2e-2, atol=2e-2)
    return got


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ic", type=int, default=48)
    p.add_argument("--mid", type=int, default=48)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--workdir", default="conv/build_rn3_row_band_shared_micro")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    wd = Path(args.workdir) / f"ic{args.ic}_mid{args.mid}_st{args.stack_size}"
    module = build_module(args.ic, args.mid, args.stack_size)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built rn3_row_band_shared ic={args.ic} mid={args.mid} stack={args.stack_size} mlir={mlir}")
    if args.build_only:
        return 0
    got = run_kernel(xclbin, insts, args.ic, args.mid)
    print(f"PASS: rn3 row-band-shared micro ic={args.ic} mid={args.mid} first={got.reshape(-1)[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
