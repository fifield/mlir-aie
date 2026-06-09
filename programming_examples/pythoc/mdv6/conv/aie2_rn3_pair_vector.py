#!/usr/bin/env python3
"""Production-style vector rn3 pair-fusion generator.

This wraps the validated `rn3_pair_vector_stage_bf16` staged PythoC kernel in an
IRON runtime shape that can process N row-major 8x8 output tiles in one host
dispatch. It intentionally avoids repeated output drains: each patch gets its
own input fill, one repeated input-side weight TAP for the 12 sequential weight
slots, and one full-arena output drain, optionally grouped with
`finish_task_group` per patch to encourage BD reuse.

Arena layout per patch, in uint16/bf16 elements:
  [0:4800)       shared conv1 scratch: 3 * 10*10*16
  [4800:7872)    final output:        3 * 8*8*16

Input arena uses the same 7872-element object type with the 12*12*48 input patch
stored at the front and the rest zero-padded.
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
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.rep_elan_bf16_pythoc import (  # noqa: E402
    KERNEL_EXTRA_GLOBALS,
    _MMUL_HELPERS,
    rn3_pair_vector_stage_bf16,
)


TILE_H = 8
TILE_W = 8
IC = 48
MID_BLOCK = 16
OC_BLOCK = 16
N_MID_BLOCKS = 3
N_OC_BLOCKS = 3
INPUT_SIZE = (TILE_H + 4) * (TILE_W + 4) * IC          # 6912
SCRATCH_SIZE = N_MID_BLOCKS * (TILE_H + 2) * (TILE_W + 2) * MID_BLOCK  # 4800
FINAL_OFFSET = SCRATCH_SIZE
FINAL_SIZE = N_OC_BLOCKS * TILE_H * TILE_W * OC_BLOCK  # 3072
ARENA_SIZE = SCRATCH_SIZE + FINAL_SIZE                 # 7872
W1_SIZE = MID_BLOCK * IC * 9 + 2 * MID_BLOCK            # 6944
W2_SIZE = OC_BLOCK * MID_BLOCK * 9 + 2 * OC_BLOCK       # 2336
WEIGHT_SLOT_SIZE = max(W1_SIZE, W2_SIZE)                # 6944
N_WEIGHT_SLOTS = N_MID_BLOCKS + N_OC_BLOCKS * N_MID_BLOCKS  # 12


def rn3_pair_vector(dev=None, n_patches: int = 1, stack_size: int = 4096, finish_per_patch: bool = True):
    """Return resolved IRON program for shared-conv1 vector rn3pair.

    Host sequence args:
      I: [n_patches, ARENA_SIZE] uint16, input patch at prefix.
      W: [12, WEIGHT_SLOT_SIZE] uint16, 3 conv1 slots then 9 conv2 slots.
      O: [n_patches, ARENA_SIZE] uint16, final output starts at FINAL_OFFSET.
    """
    if n_patches <= 0:
        raise ValueError("n_patches must be positive")
    dev = NPU2Col1() if dev is None else dev

    arena_ty = np.ndarray[(ARENA_SIZE,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(WEIGHT_SLOT_SIZE,), np.dtype[np.uint16]]
    input_batch_ty = np.ndarray[(n_patches * ARENA_SIZE,), np.dtype[np.uint16]]
    weight_all_ty = np.ndarray[(N_WEIGHT_SLOTS * WEIGHT_SLOT_SIZE,), np.dtype[np.uint16]]
    output_batch_ty = np.ndarray[(n_patches * ARENA_SIZE,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        rn3_pair_vector_stage_bf16,
        [arena_ty, weight_ty, arena_ty, np.int32, np.int32, np.int32],
        extra_globals=KERNEL_EXTRA_GLOBALS,
        helpers=_MMUL_HELPERS,
    )

    in_fifo = ObjectFifo(arena_ty, depth=1, name="rn3vv_in")
    wt_fifo = ObjectFifo(weight_ty, depth=1, name="rn3vv_wt_seq")
    out_fifo = ObjectFifo(arena_ty, depth=1, name="rn3vv_arena")

    def core_fn(a, w, c, kern):
        patch_i = 0
        while patch_i < n_patches:
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
            patch_i = patch_i + 1

    worker = Worker(core_fn, [in_fifo.cons(), wt_fifo.cons(), out_fifo.prod(), kernel], stack_size=stack_size)

    rt = Runtime()
    with rt.sequence(input_batch_ty, weight_all_ty, output_batch_ty) as (I, W, O):
        rt.start(worker)
        for p in range(n_patches):
            tg = rt.task_group() if finish_per_patch else None
            rt.fill(
                in_fifo.prod(),
                I,
                TensorAccessPattern(
                    (n_patches * ARENA_SIZE,),
                    offset=p * ARENA_SIZE,
                    sizes=[1, ARENA_SIZE],
                    strides=[0, 1],
                ),
                task_group=tg,
            )
            # Feed the fixed 12-slot weight pack as one repeated input-side TAP.
            # This matches the validated one-tile smoke path. Do not replace it
            # with 12 separate fills: that built but timed out on hardware.
            rt.fill(
                wt_fifo.prod(),
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
                out_fifo.cons(),
                O,
                TensorAccessPattern(
                    (n_patches * ARENA_SIZE,),
                    offset=p * ARENA_SIZE,
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
    p.add_argument("--n-patches", type=int, default=1)
    p.add_argument("--stack-size", type=int, default=4096)
    p.add_argument("--no-finish-per-patch", action="store_true")
    args = p.parse_args(argv)
    print(rn3_pair_vector(n_patches=args.n_patches, stack_size=args.stack_size, finish_per_patch=not args.no_finish_per_patch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
