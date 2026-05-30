#!/usr/bin/env python3
# dynamic_dma_use_lock.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --work-dir ./dynamic_dma_use_lock_build | FileCheck %s
# CHECK: PASS!

"""Demonstrate PythoC use_lock / LockAction on MLIR-allocated locks.

Same add-one flow as dynamic_dma_add_one.py, but instead of polling the lock
*value* register, the core uses the new PythoC `use_lock` op with a `LockAction`
enum — which lowers to the `llvm.aie2p.acquire` / `.release` intrinsics, exactly
like mlir-aie's `aie.use_lock`.

The interesting part: the locks are the ordinary MLIR-allocated `aie.lock`
objects. We derive the *localized* lock index (what the acquire/release
intrinsic actually takes) from the allocated lock + the core tile, using the
same rule mlir-aie's AIELocalizeLocks pass uses, and pass it to the kernel as a
plain scalar. So the kernel synchronizes against the very locks the DMA BDs
release — no hand-picked magic constant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aie.dialects.aie import (
    AIEDevice,
    buffer,
    core,
    device,
    DMAChannelDir,
    flow,
    get_target_model,
    lock,
    tile,
    WireBundle,
)
from aie.dialects.aiex import (
    dma_await_task,
    dma_free_task,
    dma_start_task,
    npu_maskwrite32,
    runtime_sequence,
    shim_dma_single_bd_task,
)
from aie.extras.context import mlir_mod_ctx
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils.compile import compile_mlir_module
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.regdb import AIEAddressDecoder
import aie.iron as iron

from pythoc import ptr, i32
from pythoc.aie.operations import read_tm, write_tm, use_lock, LockAction

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "dynamic_dma_use_lock_build"

N = 256

# ── Register addresses ───────────────────────────────────────────────────────
_decoder = AIEAddressDecoder()
_reg = _decoder.get_register_offset

DMA_BD0_0 = _reg("DMA_BD0_0", "memory")
DMA_BD1_0 = _reg("DMA_BD1_0", "memory")
DMA_S2MM_0_START_QUEUE = _reg("DMA_S2MM_0_Start_Queue", "memory")
DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory")
CORE_PROCESSOR_BUS = _reg("Core_Processor_Bus", "core")

_REGDB_GLOBALS = {
    "DMA_BD0_0": DMA_BD0_0,
    "DMA_BD1_0": DMA_BD1_0,
    "DMA_S2MM_0_START_QUEUE": DMA_S2MM_0_START_QUEUE,
    "DMA_MM2S_0_START_QUEUE": DMA_MM2S_0_START_QUEUE,
}


# ── Localized lock index, derived from the MLIR-allocated lock ────────────────


def localized_lock_index(lock_val, core_tile, target_model):
    """Localized lock index for `core_tile` to use the allocated `lock_val`.

    Mirrors mlir-aie's getLockLocalBaseIndex(core, lockTile) + lockID
    (AIELocalizeLocks.cpp / AIETargetModel.cpp) for AIE2/AIE2P core tiles:

        own tile (self) -> 3*num_locks   (isMemEast == isInternal)
        South           -> 0
        West            -> 1*num_locks
        North           -> 2*num_locks
    """
    lop = lock_val.owner.opview
    lock_id = lop.lockID.value
    ltile = lop.tile.owner.opview
    lc, lr = ltile.col.value, ltile.row.value
    cc, cr = core_tile.col.value, core_tile.row.value
    n = target_model.get_num_locks(cc, cr)

    if (lc, lr) == (cc, cr):           # self  (MemEast == isInternal)
        base = 3 * n
    elif lc == cc and lr == cr - 1:    # South
        base = 0
    elif lc == cc - 1 and lr == cr:    # West
        base = n
    elif lc == cc and lr == cr + 1:    # North
        base = 2 * n
    else:
        raise ValueError(
            f"lock tile ({lc},{lr}) is not local to core tile ({cc},{cr})"
        )
    return base + lock_id


# ── PythoC kernel ────────────────────────────────────────────────────────────


@aie_kernel
def dynamic_dma_use_lock(
    in_buf: ptr[i32, True],
    out_buf: ptr[i32, True],
    in_addr_words: i32,
    out_addr_words: i32,
    num_words: i32,
    s2mm_lock: i32,
    mm2s_lock: i32,
):
    """Receive, add one, send — waiting on locks via use_lock (acquire-GE 1).

    s2mm_lock / mm2s_lock are *localized* lock indices (see
    localized_lock_index); the S2MM/MM2S BDs release these locks (+1) on
    completion, and the core blocks on them with an acquire-greater-equal-1.
    """
    # Program + start S2MM BD0 (receive), releasing lock 0 (+1) on completion.
    write_tm((in_addr_words << 14) | num_words, DMA_BD0_0)
    write_tm(0, DMA_BD0_0 + 4)
    write_tm(0, DMA_BD0_0 + 8)
    write_tm(0, DMA_BD0_0 + 12)
    write_tm(0, DMA_BD0_0 + 16)
    write_tm(0x02040000, DMA_BD0_0 + 20)  # Valid | Lock_Rel +1 | Rel_ID 0
    write_tm(0, DMA_S2MM_0_START_QUEUE)

    # Wait for the receive: acquire-greater-equal 1 on the S2MM lock.
    use_lock(s2mm_lock, LockAction.AcquireGreaterEqual, 1)

    # Add one.
    i: i32 = 0
    while i < num_words:
        out_buf[i] = in_buf[i] + 1
        i = i + 1

    # Program + start MM2S BD1 (send), releasing lock 1 (+1) on completion.
    write_tm((out_addr_words << 14) | num_words, DMA_BD1_0)
    write_tm(0, DMA_BD1_0 + 4)
    write_tm(0, DMA_BD1_0 + 8)
    write_tm(0, DMA_BD1_0 + 12)
    write_tm(0, DMA_BD1_0 + 16)
    write_tm(0x02042000, DMA_BD1_0 + 20)  # Valid | Lock_Rel +1 | Rel_ID 1
    write_tm(1, DMA_MM2S_0_START_QUEUE)

    # Wait for the send to complete.
    use_lock(mm2s_lock, LockAction.AcquireGreaterEqual, 1)


# ── Design construction ──────────────────────────────────────────────────────


def build_mlir_module(dev, kernel):
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tmodel = get_target_model(dev)

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            kernel.resolve()

            t00 = tile(0, 0)
            t02 = tile(0, 2)

            in_buf = buffer(t02, datatype=tensor_ty, name="in_buf", address=4096)
            out_buf = buffer(
                t02, datatype=tensor_ty, name="out_buf", address=4096 + N * 4
            )

            # MLIR-allocated locks — the DMA BDs release these, the core
            # acquires them. We hand their *derived* localized index to the
            # kernel below.
            s2mm_done = lock(t02, lock_id=0, init=0, sym_name="s2mm_done")
            mm2s_done = lock(t02, lock_id=1, init=0, sym_name="mm2s_done")

            flow(t00, WireBundle.DMA, 0, t02, WireBundle.DMA, 0)
            flow(t02, WireBundle.DMA, 0, t00, WireBundle.DMA, 0)

            from aie.dialects.aie import shim_dma_allocation

            shim_dma_allocation("in_alloc", t00, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("out_alloc", t00, DMAChannelDir.S2MM, 0)

            s2mm_idx = localized_lock_index(s2mm_done, t02, tmodel)
            mm2s_idx = localized_lock_index(mm2s_done, t02, tmodel)
            print(f"      localized lock indices: s2mm={s2mm_idx} mm2s={mm2s_idx}")

            @core(t02)
            def core_body():
                kernel(
                    in_buf, out_buf, 4096 // 4, (4096 + N * 4) // 4, N,
                    s2mm_idx, mm2s_idx,
                )

            @runtime_sequence(tensor_ty, tensor_ty)
            def sequence(A, C):
                npu_maskwrite32(
                    address=CORE_PROCESSOR_BUS, value=0x1, mask=0x1,
                    column=0, row=2,
                )
                in_task = shim_dma_single_bd_task("in_alloc", A, sizes=[1, 1, 1, N])
                out_task = shim_dma_single_bd_task(
                    "out_alloc", C, sizes=[1, 1, 1, N], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)

        if not ctx.module.operation.verify():
            raise RuntimeError("Generated MLIR failed verification")
        return ctx.module


# ── Compile & Run ─────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="PythoC use_lock on MLIR locks")
    p.add_argument("--device", choices=("npu", "npu1", "npu2"), default="npu2")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def pick_device(name):
    if name.lower() == "npu2":
        return AIEDevice.npu2, "aie2p"
    return AIEDevice.npu1_1col, "aie2"


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dev, target_arch = pick_device(args.device)

    try:
        tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
        print(f"[1/4] Compiling PythoC kernel ({target_arch}) with use_lock")
        kernel = PythocKernel(
            dynamic_dma_use_lock,
            [tensor_ty, tensor_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
            target_arch=target_arch,
            extra_globals=_REGDB_GLOBALS,
        )
        print(f"      -> {kernel.object_file_name}")

        print("[2/4] Building MLIR module")
        module = build_mlir_module(dev, kernel)
        mlir_path = work_dir / "design.mlir"
        with open(mlir_path, "w") as f:
            print(module, file=f)
        print(f"      -> {mlir_path}")

        print("[3/4] Compiling design with aiecc")
        insts_path = work_dir / "insts.bin"
        xclbin_path = work_dir / "final.xclbin"
        compile_mlir_module(
            mlir_module=module, insts_path=str(insts_path),
            xclbin_path=str(xclbin_path), work_dir=str(work_dir),
            verbose=args.verbose,
        )
        print(f"      -> {xclbin_path}")

        print("[4/4] Running with pyxrt and validating")
        npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
        handle = DefaultNPURuntime.load(npu_kernel)
        in_data = np.arange(1, N + 1, dtype=np.int32)
        in_t = iron.tensor(in_data, dtype=np.int32)
        out_t = iron.zeros(N, dtype=np.int32)
        DefaultNPURuntime.run(handle, [in_t, out_t])
        output = np.array(out_t.numpy())

        expected = np.arange(2, N + 2, dtype=np.int32)
        if np.array_equal(output, expected):
            print(f"      First elements: {output[:8]}")
            print("PASS!")
            return 0
        n_wrong = int(np.sum(output != expected))
        print(f"\nFAILED: {n_wrong}/{N} elements incorrect")
        print(f"      got:      {output[:8]}")
        print(f"      expected: {expected[:8]}")
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
