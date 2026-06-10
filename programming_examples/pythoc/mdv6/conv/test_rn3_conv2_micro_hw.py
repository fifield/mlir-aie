#!/usr/bin/env python3
"""Hardware microtest for rn3 conv2 accumulation from a precomputed midplane.

This reduced-pressure test consumes a synthetic 48-channel 10x10 conv1
midplane and computes the 8x8x4 conv2+BN+SiLU output using array[f32,256]
accumulators. It isolates the conv2 half of the shared-intermediate design
from the conv1 midplane-generation code.
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
def rn3_conv2_from_midplane_bf16(midplanes: ptr[bf16, True], weights: ptr[bf16, True], output: ptr[bf16, True], mid: i32) -> void:
    # midplanes layout: [mid, 10, 10]
    # weights layout: w2[ocb=4, mid, 3, 3], bn_w[4], bn_b[4]
    accs: array[f32, 256] = array[f32, 256]()
    j: i32 = 0
    while j < 256:
        accs[j] = 0.0
        j = j + 1

    w2_size: i32 = 4 * mid * 9
    bn2_w_off: i32 = w2_size
    bn2_b_off: i32 = bn2_w_off + 4

    o1: i32 = 0
    while o1 < mid:
        r: i32 = 0
        while r < 8:
            c: i32 = 0
            while c < 8:
                kh: i32 = 0
                while kh < 3:
                    kw: i32 = 0
                    while kw < 3:
                        y: f32 = f32(midplanes[(o1 * 100) + ((r + kh) * 10 + (c + kw))])
                        base_w: i32 = ((o1 * 3 + kh) * 3 + kw)
                        base_acc: i32 = (r * 8 + c) * 4
                        accs[base_acc] = accs[base_acc] + y * f32(weights[base_w])
                        accs[base_acc + 1] = accs[base_acc + 1] + y * f32(weights[base_w + mid * 9])
                        accs[base_acc + 2] = accs[base_acc + 2] + y * f32(weights[base_w + 2 * mid * 9])
                        accs[base_acc + 3] = accs[base_acc + 3] + y * f32(weights[base_w + 3 * mid * 9])
                        kw = kw + 1
                    kh = kh + 1
                c = c + 1
            r = r + 1
        o1 = o1 + 1

    r2: i32 = 0
    while r2 < 8:
        c2: i32 = 0
        while c2 < 8:
            o2: i32 = 0
            while o2 < 4:
                idx: i32 = (r2 * 8 + c2) * 4 + o2
                x: f32 = accs[idx] * f32(weights[bn2_w_off + o2]) + f32(weights[bn2_b_off + o2])
                ax: f32 = x
                if ax < 0.0:
                    ax = -ax
                output[idx] = bf16(x * (0.5 + x / (2.0 + 2.0 * ax)))
                o2 = o2 + 1
            c2 = c2 + 1
        r2 = r2 + 1


def f32_to_bf16_u16(a):
    t = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).cpu().numpy().copy()


def bf16_u16_to_f32(a):
    return torch.from_numpy(np.asarray(a, dtype=np.uint16).copy()).view(torch.bfloat16).float().numpy()


def silu_approx(x):
    ax = np.abs(x)
    return x * (0.5 + x / (2.0 + 2.0 * ax))


def cpu_conv2(midplanes, weights, mid=48):
    out = np.zeros((8, 8, 4), dtype=np.float32)
    w2_size = 4 * mid * 9
    bn2_w_off = w2_size
    bn2_b_off = bn2_w_off + 4
    for r in range(8):
        for c in range(8):
            for o2 in range(4):
                acc = np.float32(0.0)
                for o1 in range(mid):
                    for kh in range(3):
                        for kw in range(3):
                            y = midplanes[o1, r + kh, c + kw]
                            widx = ((o2 * mid + o1) * 3 + kh) * 3 + kw
                            acc += y * weights[widx]
                x = acc * weights[bn2_w_off + o2] + weights[bn2_b_off + o2]
                out[r, c, o2] = silu_approx(x)
    return out


def build_module(mid=48, stack_size=4096):
    mid_ty = np.ndarray[(mid * 100,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(4 * mid * 9 + 8,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(256,), np.dtype[np.uint16]]
    kernel = PythocKernel(
        rn3_conv2_from_midplane_bf16,
        [mid_ty, weight_ty, output_ty, np.int32],
        extra_globals={"array": array},
    )
    mid_fifo = ObjectFifo(mid_ty, depth=1, name="rn3_c2_mid")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3_c2_wt")
    out_fifo = ObjectFifo(output_ty, depth=1, name="rn3_c2_out")

    def core_fn(a, w, c, kern):
        ea = a.acquire(1)
        ew = w.acquire(1)
        ec = c.acquire(1)
        kern(ea, ew, ec, mid)
        a.release(1)
        w.release(1)
        c.release(1)

    worker = Worker(core_fn, [mid_fifo.cons(), wt_fifo.cons(), out_fifo.prod(), kernel], stack_size=stack_size)
    rt = Runtime()
    with rt.sequence(mid_ty, weight_ty, output_ty) as (A, W, C):
        rt.start(worker)
        rt.fill(mid_fifo.prod(), A, TensorAccessPattern((mid * 100,), offset=0, sizes=[1, mid * 100], strides=[0, 1]))
        rt.fill(wt_fifo.prod(), W, TensorAccessPattern((4 * mid * 9 + 8,), offset=0, sizes=[1, 4 * mid * 9 + 8], strides=[0, 1]))
        rt.drain(out_fifo.cons(), C, TensorAccessPattern((256,), offset=0, sizes=[1, 256], strides=[0, 1]), wait=True)
    return Program(NPU2Col1(), rt).resolve_program()


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


def run_kernel(xclbin: Path, insts: Path, mid=48):
    rng = np.random.default_rng(13)
    midplanes = rng.normal(0, 0.20, size=(mid, 10, 10)).astype(np.float32)
    w2 = rng.normal(0, 0.05, size=(4, mid, 3, 3)).astype(np.float32)
    bnw = rng.normal(1.0, 0.02, size=(4,)).astype(np.float32)
    bnb = rng.normal(0.0, 0.01, size=(4,)).astype(np.float32)
    weights = np.concatenate([w2.reshape(-1), bnw, bnb]).astype(np.float32)
    mid_u16 = f32_to_bf16_u16(midplanes.reshape(-1))
    wt_u16 = f32_to_bf16_u16(weights)
    exp = cpu_conv2(bf16_u16_to_f32(mid_u16).reshape(midplanes.shape), bf16_u16_to_f32(wt_u16), mid=mid)

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(mid_u16, dtype=np.uint16)
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
    p.add_argument("--mid", type=int, default=48)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--workdir", default="conv/build_rn3_conv2_micro")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    wd = Path(args.workdir) / f"mid{args.mid}_st{args.stack_size}"
    module = build_module(args.mid, args.stack_size)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built rn3_conv2 mid={args.mid} stack={args.stack_size} mlir={mlir}")
    if args.build_only:
        return 0
    got = run_kernel(xclbin, insts, args.mid)
    print(f"PASS: rn3 conv2 micro mid={args.mid} first={got.reshape(-1)[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
