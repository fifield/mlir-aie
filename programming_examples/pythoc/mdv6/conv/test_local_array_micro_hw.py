#!/usr/bin/env python3
"""Hardware microtest for PythoC local array[...] scratch semantics.

This isolates whether PythoC stack/local arrays are safe on AIE before using
array[f32, ...] as rn3pair shared-intermediate scratch.
"""

import argparse
import subprocess as _subprocess
import sys
from pathlib import Path

import numpy as np

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
from pythoc import ptr, i32, f32, void
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
def local_array_i32(inp: ptr[i32, True], out: ptr[i32, True], n: i32) -> void:
    buf: array[i32, 256] = array[i32, 256]()
    i: i32 = 0
    while i < n:
        buf[i] = inp[i] + 7
        i = i + 1
    j: i32 = 0
    while j < n:
        out[j] = buf[j] * 3 - 5
        j = j + 1


@aie_kernel
def local_array_f32(inp: ptr[i32, True], out: ptr[i32, True], n: i32) -> void:
    buf: array[f32, 256] = array[f32, 256]()
    i: i32 = 0
    while i < n:
        buf[i] = f32(inp[i]) * 0.5 + 1.25
        i = i + 1
    j: i32 = 0
    while j < n:
        out[j] = i32(buf[j] * 4.0)
        j = j + 1


@aie_kernel
def local_array_f32_two(inp: ptr[i32, True], out: ptr[i32, True], n: i32) -> void:
    accs: array[f32, 256] = array[f32, 256]()
    midplane: array[f32, 100] = array[f32, 100]()
    i: i32 = 0
    while i < 256:
        accs[i] = f32(i) * 0.25 + 2.0
        i = i + 1
    j: i32 = 0
    while j < 100:
        midplane[j] = f32(inp[j]) * 0.5 + 1.25
        j = j + 1
    k: i32 = 0
    while k < n:
        out[k] = i32((accs[k] + midplane[k]) * 4.0)
        k = k + 1


def build_module(kind="i32", n=16, stack_size=4096):
    tensor_ty = np.ndarray[(n,), np.dtype[np.int32]]
    fn = local_array_i32 if kind == "i32" else (local_array_f32 if kind == "f32" else local_array_f32_two)
    kernel = PythocKernel(fn, [tensor_ty, tensor_ty, np.int32], extra_globals={"array": array})
    in_fifo = ObjectFifo(tensor_ty, depth=1, name=f"arr_{kind}_in")
    out_fifo = ObjectFifo(tensor_ty, depth=1, name=f"arr_{kind}_out")

    def core_fn(a, c, kern):
        ea = a.acquire(1)
        ec = c.acquire(1)
        kern(ea, ec, n)
        a.release(1)
        c.release(1)

    worker = Worker(core_fn, [in_fifo.cons(), out_fifo.prod(), kernel], stack_size=stack_size)
    def sequence(A, C, in_fifo_prod, out_fifo_cons):
        tap = TensorAccessPattern((n,), offset=0, sizes=[1, n], strides=[0, 1])
        in_fifo_prod.fill(A, tap)
        out_fifo_cons.drain(C, tap, wait=True)

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, in_fifo.prod(), out_fifo.cons()],
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


def run_kernel(xclbin: Path, insts: Path, kind="i32", n=16):
    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    inp_np = np.arange(n, dtype=np.int32)
    a = iron.tensor(inp_np, dtype=np.int32)
    c = iron.zeros(n, dtype=np.int32)
    DefaultNPURuntime.run(handle, [a, c])
    got = c.numpy().copy()
    if kind == "i32":
        exp = (inp_np + 7) * 3 - 5
    elif kind == "f32":
        exp = ((inp_np.astype(np.float32) * 0.5 + 1.25) * 4.0).astype(np.int32)
    else:
        exp = (((np.arange(n, dtype=np.float32) * 0.25 + 2.0) + (inp_np.astype(np.float32) * 0.5 + 1.25)) * 4.0).astype(np.int32)
    np.testing.assert_array_equal(got, exp)
    return got


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["i32", "f32", "f32-two"], default="i32")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--workdir", default="conv/build_local_array_micro")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    if args.n > 256:
        raise SystemExit("--n must be <= 256")
    if args.kind == "f32-two" and args.n > 100:
        raise SystemExit("--n must be <= 100 for --kind f32-two")
    wd = Path(args.workdir) / f"{args.kind}_n{args.n}_st{args.stack_size}"
    module = build_module(args.kind, args.n, args.stack_size)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built kind={args.kind} n={args.n} stack={args.stack_size} mlir={mlir}")
    if args.build_only:
        return 0
    got = run_kernel(xclbin, insts, args.kind, args.n)
    print(f"PASS: local_array kind={args.kind} n={args.n} stack={args.stack_size} first={got[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
