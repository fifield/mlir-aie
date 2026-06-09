#!/usr/bin/env python3
"""Hardware smoke for memtile-column fanout vector rn3-pair prototype."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module

from conv.aie2_rn3_pair_vector_memtile import (  # noqa: E402
    ARENA_SIZE,
    FINAL_OFFSET,
    MASK_OFFSET,
    MASK_SIZE,
    N_OC_BLOCKS,
    OC_BLOCK,
    TILE_H,
    TILE_W,
    rn3_pair_vector_memtile,
)
from conv.resident_xclbin_runner import ResidentXCLBinRunner  # noqa: E402
from conv.test_rn3_pair_vector_batch_hw import (  # noqa: E402
    bf16_u16_to_f32,
    build_weights_and_oracle_inputs,
    f32_to_bf16_u16,
)


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


def extract_output(raw_flat: np.ndarray, n_patches: int):
    raw = raw_flat.copy().reshape(n_patches, ARENA_SIZE)
    got_patches = []
    for p in range(n_patches):
        blocks = []
        for ob in range(N_OC_BLOCKS):
            start = FINAL_OFFSET + ob * TILE_H * TILE_W * OC_BLOCK
            stop = start + TILE_H * TILE_W * OC_BLOCK
            blocks.append(bf16_u16_to_f32(raw[p, start:stop]).reshape(TILE_H, TILE_W, OC_BLOCK))
        got_patches.append(np.concatenate(blocks, axis=2))
    return np.stack(got_patches, axis=0)


def check(got: np.ndarray, exp: np.ndarray):
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp, rtol=7e-2, atol=7e-2)


def _with_all_valid_masks(input_all: np.ndarray, n_patches: int) -> np.ndarray:
    old_arena = input_all.size // n_patches
    if old_arena == ARENA_SIZE:
        out = input_all.copy().reshape(n_patches, ARENA_SIZE)
    else:
        out = np.zeros((n_patches, ARENA_SIZE), dtype=np.uint16)
        out[:, :old_arena] = input_all.reshape(n_patches, old_arena)
    out[:, MASK_OFFSET:MASK_OFFSET + MASK_SIZE] = f32_to_bf16_u16(np.ones(MASK_SIZE, dtype=np.float32))
    return out.reshape(-1)


def run_hw(xclbin: Path, insts: Path, n_cores: int, *, resident: bool = False):
    input_all, weight_all, exp = build_weights_and_oracle_inputs(n_cores)
    input_all = _with_all_valid_masks(input_all, n_cores)
    if resident:
        with ResidentXCLBinRunner(xclbin, insts) as runner:
            out_arg = np.zeros(n_cores * ARENA_SIZE, dtype=np.uint16)
            res1 = runner.run(input_all, weight_all, out_arg, bo_key=f"rn3vm_{n_cores}", output_indices={2}, static_indices={1})
            s1 = runner.last_stats
            got1 = extract_output(res1[2], n_cores)
            print(f"resident_first written={s1.n_written} bytes={s1.bytes_written} write_ms={s1.write_ms:.3f} kernel_ms={s1.kernel_ms:.3f} read_ms={s1.read_ms:.3f} total_ms={s1.total_ms:.3f}")
            check(got1, exp)
            res2 = runner.run(input_all, weight_all, out_arg, bo_key=f"rn3vm_{n_cores}", output_indices={2}, static_indices={1})
            s2 = runner.last_stats
            got2 = extract_output(res2[2], n_cores)
            print(f"resident_second written={s2.n_written} bytes={s2.bytes_written} write_ms={s2.write_ms:.3f} kernel_ms={s2.kernel_ms:.3f} read_ms={s2.read_ms:.3f} total_ms={s2.total_ms:.3f}")
            check(got2, exp)
            return got2, s2.total_ms

    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(input_all, dtype=np.uint16)
    W = iron.tensor(weight_all, dtype=np.uint16)
    C = iron.zeros(n_cores * ARENA_SIZE, dtype=np.uint16)
    t0 = time.perf_counter()
    DefaultNPURuntime.run(handle, [A, W, C])
    run_ms = (time.perf_counter() - t0) * 1000
    print(f"run_ms={run_ms:.2f}")
    got = extract_output(C.numpy(), n_cores)
    check(got, exp)
    return got, run_ms


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-cores", type=int, default=16)
    p.add_argument("--workdir", default="conv/build_rn3_pair_vector_memtile")
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--resident", action="store_true")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    module = rn3_pair_vector_memtile(n_cores=args.n_cores, stack_size=args.stack_size)
    wd = Path(args.workdir) / f"cores{args.n_cores}_st{args.stack_size}"
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built rn3_pair_vector_memtile n_cores={args.n_cores} mlir={mlir}")
    if args.build_only:
        return 0
    got, run_ms = run_hw(xclbin, insts, args.n_cores, resident=args.resident)
    print(f"PASS: rn3_pair_vector_memtile n_cores={args.n_cores} shape={got.shape} run_ms={run_ms:.2f} first={got.reshape(-1)[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
