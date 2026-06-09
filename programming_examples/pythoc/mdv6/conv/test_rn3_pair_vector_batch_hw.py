#!/usr/bin/env python3
"""Hardware smoke for production-style shared-conv1 vector rn3pair generator."""
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

from conv.aie2_rn3_pair_vector import (  # noqa: E402
    ARENA_SIZE,
    FINAL_OFFSET,
    IC,
    INPUT_SIZE,
    MID_BLOCK,
    N_MID_BLOCKS,
    N_OC_BLOCKS,
    N_WEIGHT_SLOTS,
    OC_BLOCK,
    TILE_H,
    TILE_W,
    W1_SIZE,
    W2_SIZE,
    WEIGHT_SLOT_SIZE,
    rn3_pair_vector,
)
from conv.test_rn3_pair_vector_oneblock_hw import (  # noqa: E402
    bf16_u16_to_f32,
    conv3x3_bnsilu_cpu,
    f32_to_bf16_u16,
    pack_3x3_weights_u16,
    silu,
    unpack_packed_3x3_weights_f32,
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


def build_weights_and_oracle_inputs(n_patches: int, seed: int = 97531):
    rng = np.random.default_rng(seed)
    input_arenas = np.zeros((n_patches, ARENA_SIZE), dtype=np.uint16)
    inputs_f32 = []
    for p in range(n_patches):
        inp = rng.normal(0, 0.12, size=(TILE_H + 4, TILE_W + 4, IC)).astype(np.float32)
        inp_u16 = f32_to_bf16_u16(inp.reshape(-1))
        input_arenas[p, :INPUT_SIZE] = inp_u16
        inputs_f32.append(bf16_u16_to_f32(inp_u16).reshape(TILE_H + 4, TILE_W + 4, IC))

    w1_slots = []
    w1_u16s = []
    for mb in range(N_MID_BLOCKS):
        w1 = rng.normal(0, 0.035, size=(MID_BLOCK, IC, 3, 3)).astype(np.float32)
        bn1w = rng.normal(1.0, 0.02, size=(MID_BLOCK,)).astype(np.float32)
        bn1b = rng.normal(0.0, 0.01, size=(MID_BLOCK,)).astype(np.float32)
        w1_u16 = pack_3x3_weights_u16(w1, bn1w, bn1b)
        assert len(w1_u16) == W1_SIZE
        slot = np.zeros(WEIGHT_SLOT_SIZE, dtype=np.uint16)
        slot[:W1_SIZE] = w1_u16
        w1_slots.append(slot)
        w1_u16s.append(w1_u16)

    w2_slots_by_ob = []
    w2_u16_by_ob = []
    for ob in range(N_OC_BLOCKS):
        bn2w = rng.normal(1.0, 0.02, size=(OC_BLOCK,)).astype(np.float32)
        bn2b = rng.normal(0.0, 0.01, size=(OC_BLOCK,)).astype(np.float32)
        slots = []
        chunks = []
        for mb in range(N_MID_BLOCKS):
            w2 = rng.normal(0, 0.035, size=(OC_BLOCK, MID_BLOCK, 3, 3)).astype(np.float32)
            w2_u16 = pack_3x3_weights_u16(w2, bn2w, bn2b)
            assert len(w2_u16) == W2_SIZE
            slot = np.zeros(WEIGHT_SLOT_SIZE, dtype=np.uint16)
            slot[:W2_SIZE] = w2_u16
            slots.append(slot)
            chunks.append(w2_u16)
        w2_slots_by_ob.append(slots)
        w2_u16_by_ob.append(chunks)

    weight_all = np.zeros((N_WEIGHT_SLOTS, WEIGHT_SLOT_SIZE), dtype=np.uint16)
    wi = 0
    for mb in range(N_MID_BLOCKS):
        weight_all[wi] = w1_slots[mb]
        wi += 1
    for ob in range(N_OC_BLOCKS):
        for mb in range(N_MID_BLOCKS):
            weight_all[wi] = w2_slots_by_ob[ob][mb]
            wi += 1

    expected = []
    for inp_f32 in inputs_f32:
        mid_parts = []
        for mb in range(N_MID_BLOCKS):
            mid_f32 = conv3x3_bnsilu_cpu(inp_f32, w1_u16s[mb], TILE_H + 2, TILE_W + 2, IC, MID_BLOCK)
            mid_parts.append(bf16_u16_to_f32(f32_to_bf16_u16(mid_f32.reshape(-1))).reshape(TILE_H + 2, TILE_W + 2, MID_BLOCK))
        mid_full = np.concatenate(mid_parts, axis=2)
        exp_blocks = []
        for ob in range(N_OC_BLOCKS):
            w2_chunks = []
            bn2w = None
            bn2b = None
            for mb in range(N_MID_BLOCKS):
                w2, bw, bb = unpack_packed_3x3_weights_f32(w2_u16_by_ob[ob][mb], OC_BLOCK, MID_BLOCK)
                w2_chunks.append(w2)
                bn2w = bw
                bn2b = bb
            w2_full = np.concatenate(w2_chunks, axis=1)
            exp = np.zeros((TILE_H, TILE_W, OC_BLOCK), dtype=np.float32)
            for oh in range(TILE_H):
                for ow in range(TILE_W):
                    patch = mid_full[oh:oh + 3, ow:ow + 3, :]
                    for co in range(OC_BLOCK):
                        acc = np.float32(0.0)
                        for ci in range(MID_BLOCK * N_MID_BLOCKS):
                            for kh in range(3):
                                for kw in range(3):
                                    acc += np.float32(patch[kh, kw, ci] * w2_full[co, ci, kh, kw])
                        exp[oh, ow, co] = silu(acc * bn2w[co] + bn2b[co])
            exp_blocks.append(exp)
        expected.append(np.concatenate(exp_blocks, axis=2))
    return input_arenas.reshape(-1), weight_all.reshape(-1), np.stack(expected, axis=0)


def run_hw(xclbin: Path, insts: Path, n_patches: int):
    input_all, weight_all, exp = build_weights_and_oracle_inputs(n_patches)
    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    A = iron.tensor(input_all, dtype=np.uint16)
    W = iron.tensor(weight_all, dtype=np.uint16)
    C = iron.zeros(n_patches * ARENA_SIZE, dtype=np.uint16)
    t0 = time.perf_counter()
    DefaultNPURuntime.run(handle, [A, W, C])
    run_ms = (time.perf_counter() - t0) * 1000
    print(f"run_ms={run_ms:.2f}")
    raw = C.numpy().copy().reshape(n_patches, ARENA_SIZE)
    got_patches = []
    for p in range(n_patches):
        blocks = []
        for ob in range(N_OC_BLOCKS):
            start = FINAL_OFFSET + ob * TILE_H * TILE_W * OC_BLOCK
            stop = start + TILE_H * TILE_W * OC_BLOCK
            blocks.append(bf16_u16_to_f32(raw[p, start:stop]).reshape(TILE_H, TILE_W, OC_BLOCK))
        got_patches.append(np.concatenate(blocks, axis=2))
    got = np.stack(got_patches, axis=0)
    max_abs = float(np.max(np.abs(got - exp)))
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp, rtol=7e-2, atol=7e-2)
    return got


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-patches", type=int, default=1)
    p.add_argument("--workdir", default="conv/build_rn3_pair_vector")
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--no-finish-per-patch", action="store_true")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)
    module = rn3_pair_vector(
        n_patches=args.n_patches,
        stack_size=args.stack_size,
        finish_per_patch=not args.no_finish_per_patch,
    )
    wd = Path(args.workdir) / f"p{args.n_patches}_st{args.stack_size}_tg{int(not args.no_finish_per_patch)}"
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built rn3_pair_vector n_patches={args.n_patches} mlir={mlir}")
    if args.build_only:
        return 0
    got = run_hw(xclbin, insts, args.n_patches)
    print(f"PASS: rn3_pair_vector n_patches={args.n_patches} shape={got.shape} first={got.reshape(-1)[:8].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
