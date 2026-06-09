#!/usr/bin/env python3
"""Spatial-lane vector rn3-pair expanded-runtime prototype.

This is the first recoverable-regimes join point:

* keep the high-performance specialized vector rn3-pair body
  (`rn3_pair_vector_stage_bf16`);
* expand one runtime sequence over multiple spatial patch lanes;
* avoid generic regime kernels and avoid repeated output drains.

Each worker/lane computes one or more 8x8 rn3-pair output patches. All lanes use
identical 12-slot weights: 3 conv1 mid-block slots followed by 3x3 conv2 slots.
The runtime sequence feeds each lane with its assigned input patch, a repeated
input-side weight TAP, and a distinct output arena drain.

Host sequence args:
  I: [n_lanes * patches_per_lane, ARENA_SIZE] uint16
  W: [12, WEIGHT_SLOT_SIZE] uint16
  O: [n_lanes * patches_per_lane, ARENA_SIZE] uint16

This is intentionally a prototype before full model integration. The key build
question is whether lane-level spatial expansion routes and whether hardware
latency improves versus the one-worker pN vector scaffold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.rep_elan_bf16_pythoc import (  # noqa: E402
    KERNEL_EXTRA_GLOBALS,
    _MMUL_HELPERS,
    rn3_pair_vector_stage_bf16,
)

# Reuse the validated re6 rn3-pair vector geometry.
TILE_H = 8
TILE_W = 8
IC = 48
MID_BLOCK = 16
OC_BLOCK = 16
N_MID_BLOCKS = 3
N_OC_BLOCKS = 3
INPUT_SIZE = (TILE_H + 4) * (TILE_W + 4) * IC
SCRATCH_SIZE = N_MID_BLOCKS * (TILE_H + 2) * (TILE_W + 2) * MID_BLOCK
FINAL_OFFSET = SCRATCH_SIZE
FINAL_SIZE = N_OC_BLOCKS * TILE_H * TILE_W * OC_BLOCK
MASK_OFFSET = FINAL_OFFSET + FINAL_SIZE
# Per-patch 10x10 conv1-intermediate validity mask. Full-image boundary
# patches must zero conv1 scratch outside the global conv1 output domain before
# conv2 consumes it, matching the baseline two-conv path's conv2 padding.
MASK_SIZE = (TILE_H + 2) * (TILE_W + 2)
ARENA_SIZE = MASK_OFFSET + MASK_SIZE
W1_SIZE = MID_BLOCK * IC * 9 + 2 * MID_BLOCK
# Conv2 must accumulate all 48 intermediate channels in f32 before BN/SiLU to
# match the baseline mc_re6_rn3 path. One full conv2 slot per output block.
W2_SIZE = OC_BLOCK * (N_MID_BLOCKS * MID_BLOCK) * 9 + 2 * OC_BLOCK
WEIGHT_SLOT_SIZE = max(W1_SIZE, W2_SIZE)
# 3 conv1 mid-block slots + 3 full conv2 output-block slots.
N_WEIGHT_SLOTS = N_MID_BLOCKS + N_OC_BLOCKS


def rn3_pair_vector_ocb(
    dev=None,
    n_lanes: int = 2,
    patches_per_lane: int = 1,
    stack_size: int = 4096,
    finish_per_patch: bool = True,
):
    """Return resolved IRON program for spatial-lane rn3-pair vector prototype."""
    if n_lanes <= 0:
        raise ValueError("n_lanes must be positive")
    if patches_per_lane <= 0:
        raise ValueError("patches_per_lane must be positive")
    if n_lanes > 32:
        raise ValueError("keep n_lanes <= 32 for this routing prototype")
    dev = NPU2() if dev is None else dev
    total_patches = n_lanes * patches_per_lane

    arena_ty = np.ndarray[(ARENA_SIZE,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(WEIGHT_SLOT_SIZE,), np.dtype[np.uint16]]
    input_batch_ty = np.ndarray[(total_patches * ARENA_SIZE,), np.dtype[np.uint16]]
    weight_all_ty = np.ndarray[(N_WEIGHT_SLOTS * WEIGHT_SLOT_SIZE,), np.dtype[np.uint16]]
    output_batch_ty = np.ndarray[(total_patches * ARENA_SIZE,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        rn3_pair_vector_stage_bf16,
        [arena_ty, weight_ty, arena_ty, np.int32, np.int32, np.int32],
        extra_globals=KERNEL_EXTRA_GLOBALS,
        helpers=_MMUL_HELPERS,
    )

    in_fifos = []
    wt_fifos = []
    out_fifos = []
    workers = []

    def core_fn(a, w, c, kern):
        patch_i = 0
        while patch_i < patches_per_lane:
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
                ew = w.acquire(1)
                kern(eout, ew, eout, 1, ob, 0)
                w.release(1)
                ob = ob + 1
            c.release(1)
            patch_i = patch_i + 1

    for lane in range(n_lanes):
        inf = ObjectFifo(arena_ty, depth=1, name=f"rn3vo_in_{lane}")
        wf = ObjectFifo(weight_ty, depth=1, name=f"rn3vo_wt_{lane}")
        outf = ObjectFifo(arena_ty, depth=1, name=f"rn3vo_out_{lane}")
        in_fifos.append(inf)
        wt_fifos.append(wf)
        out_fifos.append(outf)
        workers.append(
            Worker(core_fn, [inf.cons(), wf.cons(), outf.prod(), kernel], stack_size=stack_size)
        )

    rt = Runtime()
    with rt.sequence(input_batch_ty, weight_all_ty, output_batch_ty) as (I, W, O):
        rt.start(*workers)
        for k in range(patches_per_lane):
            # One task group spans all spatial lanes for this patch step. The
            # first draft finished per lane and effectively serialized lanes;
            # this keeps lane fills/drains in the same schedulable group.
            tg = rt.task_group() if finish_per_patch else None
            for lane in range(n_lanes):
                patch_idx = lane * patches_per_lane + k
                rt.fill(
                    in_fifos[lane].prod(),
                    I,
                    TensorAccessPattern(
                        (total_patches * ARENA_SIZE,),
                        offset=patch_idx * ARENA_SIZE,
                        sizes=[1, ARENA_SIZE],
                        strides=[0, 1],
                    ),
                    task_group=tg,
                )
                # Keep the validated repeated input-side weight TAP. It is safe
                # for the vector rn3-pair path; repeated output drains are not.
                rt.fill(
                    wt_fifos[lane].prod(),
                    W,
                    TensorAccessPattern(
                        (N_WEIGHT_SLOTS * WEIGHT_SLOT_SIZE,),
                        offset=0,
                        sizes=[N_WEIGHT_SLOTS, WEIGHT_SLOT_SIZE],
                        strides=[WEIGHT_SLOT_SIZE, 1],
                    ),
                    task_group=tg,
                )
                rt.drain(
                    out_fifos[lane].cons(),
                    O,
                    TensorAccessPattern(
                        (total_patches * ARENA_SIZE,),
                        offset=patch_idx * ARENA_SIZE,
                        sizes=[1, ARENA_SIZE],
                        strides=[0, 1],
                    ),
                    task_group=tg,
                    wait=True,
                )
            if tg is not None:
                rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-lanes", type=int, default=2)
    p.add_argument("--patches-per-lane", type=int, default=1)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--no-finish-per-patch", action="store_true")
    args = p.parse_args(argv)
    print(
        rn3_pair_vector_ocb(
            n_lanes=args.n_lanes,
            patches_per_lane=args.patches_per_lane,
            stack_size=args.stack_size,
            finish_per_patch=not args.no_finish_per_patch,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
