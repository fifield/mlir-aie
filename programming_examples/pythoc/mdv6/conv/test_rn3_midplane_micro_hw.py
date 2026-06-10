#!/usr/bin/env python3
"""Hardware microtest for the rn3 shared-intermediate conv1 midplane.

This reduced-pressure test computes one conv1 output channel over the 10x10
intermediate extent needed by an 8x8 final rn3 tile. It uses the same bf16
input/weight conventions and SiLU approximation as test_rn3_pair_hw.py, but
only keeps array[f32,100] live and drains that midplane.
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
def rn3_conv1_midplane_bf16(input_patch: ptr[bf16, True], weights: ptr[bf16, True], output: ptr[bf16, True], ic: i32) -> void:
    # weights layout for one mid channel: w1[ic,3,3], bn_w, bn_b.
    midplane: array[f32, 100] = array[f32, 100]()
    patch_w: i32 = 12
    bn_w_off: i32 = ic * 9
    bn_b_off: i32 = bn_w_off + 1

    mr: i32 = 0
    while mr < 10:
        mc: i32 = 0
        while mc < 10:
            acc: f32 = 0.0
            kh: i32 = 0
            while kh < 3:
                kw: i32 = 0
                while kw < 3:
                    i: i32 = 0
                    while i < ic:
                        in_idx: i32 = ((mr + kh) * patch_w + (mc + kw)) * ic + i
                        w_idx: i32 = (i * 3 + kh) * 3 + kw
                        acc = acc + f32(input_patch[in_idx]) * f32(weights[w_idx])
                        i = i + 1
                    kw = kw + 1
                kh = kh + 1
            x: f32 = acc * f32(weights[bn_w_off]) + f32(weights[bn_b_off])
            ax: f32 = x
            if ax < 0.0:
                ax = -ax
            y: f32 = x * (0.5 + x / (2.0 + 2.0 * ax))
            midplane[mr * 10 + mc] = y
            mc = mc + 1
        mr = mr + 1

    j: i32 = 0
    while j < 100:
        output[j] = bf16(midplane[j])
        j = j + 1


def f32_to_bf16_u16(a):
    t = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).cpu().numpy().copy()


def bf16_u16_to_f32(a):
    return torch.from_numpy(np.asarray(a, dtype=np.uint16).copy()).view(torch.bfloat16).float().numpy()


def silu_approx(x):
    ax = np.abs(x)
    return x * (0.5 + x / (2.0 + 2.0 * ax))


def cpu_midplane(input_patch, weights, ic=48):
    out = np.zeros((10, 10), dtype=np.float32)
    flat = input_patch.reshape(-1)
    bn_w_off = ic * 9
    bn_b_off = bn_w_off + 1
    for mr in range(10):
        for mc in range(10):
            acc = np.float32(0.0)
            for kh in range(3):
                for kw in range(3):
                    for i in range(ic):
                        in_idx = ((mr + kh) * 12 + (mc + kw)) * ic + i
                        w_idx = (i * 3 + kh) * 3 + kw
                        acc += flat[in_idx] * weights[w_idx]
            x = acc * weights[bn_w_off] + weights[bn_b_off]
            out[mr, mc] = silu_approx(x)
    return out


def build_module(ic=48, stack_size=4096):
    input_ty = np.ndarray[(12 * 12 * ic,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(ic * 9 + 2,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(100,), np.dtype[np.uint16]]
    kernel = PythocKernel(
        rn3_conv1_midplane_bf16,
        [input_ty, weight_ty, output_ty, np.int32],
        extra_globals={"array": array},
    )
    in_fifo = ObjectFifo(input_ty, depth=1, name="rn3_mid_in")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3_mid_wt")
    out_fifo = ObjectFifo(output_ty, depth=1, name="rn3_mid_out")

    def core_fn(a, w, c, kern):
        ea = a.acquire(1)
        ew = w.acquire(1)
        ec = c.acquire(1)
        kern(ea, ew, ec, ic)
        a.release(1)
        w.release(1)
        c.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), out_fifo.prod(), kernel], stack_size=stack_size)
    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (A, W, C):
        rt.start(worker)
        rt.fill(in_fifo.prod(), A, TensorAccessPattern((12 * 12 * ic,), offset=0, sizes=[1, 12 * 12 * ic], strides=[0, 1]))
        rt.fill(wt_fifo.prod(), W, TensorAccessPattern((ic * 9 + 2,), offset=0, sizes=[1, ic * 9 + 2], strides=[0, 1]))
        rt.drain(out_fifo.cons(), C, TensorAccessPattern((100,), offset=0, sizes=[1, 100], strides=[0, 1]), wait=True)
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


def run_kernel(xclbin: Path, insts: Path, ic=48):
    rng = np.random.default_rng(11)
    inp = rng.normal(0, 0.15, size=(12, 12, ic)).astype(np.float32)
    w = rng.normal(0, 0.05, size=(ic, 3, 3)).astype(np.float32)
    bnw = rng.normal(1.0, 0.02, size=(1,)).astype(np.float32)
    bnb = rng.normal(0.0, 0.01, size=(1,)).astype(np.float32)
    weights = np.concatenate([w.reshape(-1), bnw, bnb]).astype(np.float32)
    in_u16 = f32_to_bf16_u16(inp.reshape(-1))
    wt_u16 = f32_to_bf16_u16(weights)
    exp = cpu_midplane(bf16_u16_to_f32(in_u16).reshape(inp.shape), bf16_u16_to_f32(wt_u16), ic=ic)

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(in_u16, dtype=np.uint16)
    W = iron.tensor(wt_u16, dtype=np.uint16)
    C = iron.zeros(100, dtype=np.uint16)
    DefaultNPURuntime.run(handle, [A, W, C])
    got = bf16_u16_to_f32(C.numpy().copy()).reshape(10, 10)
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    np.testing.assert_allclose(got, exp, rtol=2e-2, atol=2e-2)
    return got


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ic", type=int, default=48)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--workdir", default="conv/build_rn3_midplane_micro")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    wd = Path(args.workdir) / f"ic{args.ic}_st{args.stack_size}"
    module = build_module(args.ic, args.stack_size)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built rn3_midplane ic={args.ic} stack={args.stack_size} mlir={mlir}")
    if args.build_only:
        return 0
    got = run_kernel(xclbin, insts, args.ic)
    print(f"PASS: rn3 midplane micro ic={args.ic} first={got.reshape(-1)[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
