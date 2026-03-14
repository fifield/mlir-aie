#!/usr/bin/env python3
# dynamic_dma_struct.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --work-dir ./dynamic_dma_struct_build | FileCheck %s
# CHECK: PASS!

"""Dynamic DMA programming with struct-based helper functions.

Same functionality as dynamic_dma_add_one.py, but demonstrates PythoC struct
support by passing DMA channel configuration (BD base address, start queue
address, lock address) as a struct to a helper function that programs a BD
and starts a DMA transfer.

This tests:
  - PythoC struct types in @aie_kernel cross-compilation
  - Helper function calls from the main kernel
  - Multi-function compilation in compile_pythoc_source
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

# Import PythoC types and operations for inline kernel definition
from pythoc import ptr, i32, struct
from pythoc.aie.operations import read_tm, write_tm

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "dynamic_dma_struct_build"

# Number of int32 elements to process
N = 256

# ── Register addresses from register database ────────────────────────────────

_decoder = AIEAddressDecoder()
_reg = _decoder.get_register_offset

# DMA buffer descriptor base address (BD0 start; stride 0x20 between BDs)
DMA_BD_BASE = _reg("DMA_BD0_0", "memory")      # 0x1d000

# DMA channel start queue registers
DMA_S2MM_0_START_QUEUE = _reg("DMA_S2MM_0_Start_Queue", "memory")
DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory")

# Lock value registers
LOCK0_VALUE = _reg("Lock0_value", "memory")
LOCK1_VALUE = _reg("Lock1_value", "memory")

# Core processor bus enable register
CORE_PROCESSOR_BUS = _reg("Core_Processor_Bus", "core")

# Constants passed to the PythoC kernel via extra_globals
_REGDB_GLOBALS = {
    "DMA_BD_BASE": DMA_BD_BASE,
    "DMA_S2MM_0_START_QUEUE": DMA_S2MM_0_START_QUEUE,
    "DMA_MM2S_0_START_QUEUE": DMA_MM2S_0_START_QUEUE,
    "LOCK0_VALUE": LOCK0_VALUE,
    "LOCK1_VALUE": LOCK1_VALUE,
}


# ── PythoC kernel and helpers ─────────────────────────────────────────────────
#
# Helper functions are decorated with @aie_kernel just like the main kernel.
# PythocKernel's helpers= parameter compiles them all into the same LLVM
# module so the main kernel can call them.


@aie_kernel
def program_bd_and_start(
    ch: struct[bd_base: i32, start_queue: i32, lock_addr: i32],
    bd_id: i32,
    base_addr_words: i32,
    num_words: i32,
    lock_rel_id: i32,
):
    """Program a DMA buffer descriptor and start the channel.

    Args:
        ch: DMA channel config (BD base address, start queue addr, lock addr)
        bd_id: Buffer descriptor ID (used in start queue)
        base_addr_words: Buffer base address in 32-bit word units
        num_words: Number of 32-bit words to transfer
        lock_rel_id: Lock ID to release on completion
    """
    bd: i32 = ch.bd_base + (bd_id * 32)  # 0x20 stride between BDs

    # BD word 0: [27:14] BASE_ADDRESS, [13:0] BUFFER_LENGTH
    write_tm((base_addr_words << 14) | num_words, bd)

    # BD words 1-4: defaults (contiguous 1D, no packet, no iteration)
    write_tm(0, bd + 4)
    write_tm(0, bd + 8)
    write_tm(0, bd + 12)
    write_tm(0, bd + 16)

    # BD word 5: VALID_BD=1, LOCK_REL_VALUE=+1, LOCK_REL_ID
    valid_bd: i32 = 1 << 25
    lock_rel_val: i32 = 1 << 18   # +1
    lock_id_bits: i32 = lock_rel_id << 13
    write_tm(valid_bd | lock_rel_val | lock_id_bits, bd + 20)

    # Start the channel: bits [3:0] = BD_ID
    write_tm(bd_id, ch.start_queue)


@aie_kernel
def wait_for_lock(
    ch: struct[bd_base: i32, start_queue: i32, lock_addr: i32],
):
    """Poll until the lock associated with a DMA channel is released."""
    done: i32 = 0
    while done == 0:
        done = read_tm(ch.lock_addr)


@aie_kernel
def dynamic_dma_struct(
    in_buf: ptr[i32, True],
    out_buf: ptr[i32, True],
    in_addr_words: i32,
    out_addr_words: i32,
    num_words: i32,
):
    """Dynamically program tile DMA using struct-based helpers.

    Uses a DMAChannel struct to pass BD base, start queue, and lock
    addresses to helper functions, reducing repetition.
    """
    # Build channel config structs (both share the same BD base;
    # bd_id selects which BD within that base: BD0=0, BD1=1, etc.)
    s2mm: struct[bd_base: i32, start_queue: i32, lock_addr: i32] = (
        DMA_BD_BASE, DMA_S2MM_0_START_QUEUE, LOCK0_VALUE
    )
    mm2s: struct[bd_base: i32, start_queue: i32, lock_addr: i32] = (
        DMA_BD_BASE, DMA_MM2S_0_START_QUEUE, LOCK1_VALUE
    )

    # Receive data from stream via S2MM channel
    program_bd_and_start(s2mm, 0, in_addr_words, num_words, 0)
    wait_for_lock(s2mm)

    # Process data: add 1 to each element
    i: i32 = 0
    while i < num_words:
        out_buf[i] = in_buf[i] + 1
        i = i + 1

    # Send data to stream via MM2S channel
    program_bd_and_start(mm2s, 1, out_addr_words, num_words, 1)
    wait_for_lock(mm2s)


# ── Design construction ──────────────────────────────────────────────────────


def build_mlir_module(dev, kernel):
    """Build MLIR module using lower-level dialect API with raw flows."""
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            kernel.resolve()

            t00 = tile(0, 0)
            t02 = tile(0, 2)

            in_buf = buffer(t02, datatype=tensor_ty, name="in_buf", address=0)
            out_buf = buffer(
                t02, datatype=tensor_ty, name="out_buf", address=N * 4
            )

            lock(t02, lock_id=0, init=0, sym_name="s2mm_done")
            lock(t02, lock_id=1, init=0, sym_name="mm2s_done")

            flow(t00, WireBundle.DMA, 0, t02, WireBundle.DMA, 0)
            flow(t02, WireBundle.DMA, 0, t00, WireBundle.DMA, 0)

            from aie.dialects.aie import shim_dma_allocation

            shim_dma_allocation("in_alloc", t00, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("out_alloc", t00, DMAChannelDir.S2MM, 0)

            @core(t02, kernel.bin_name)
            def core_body():
                kernel(in_buf, out_buf, 0, N, N)

            @runtime_sequence(tensor_ty, tensor_ty)
            def sequence(A, C):
                npu_maskwrite32(
                    address=CORE_PROCESSOR_BUS,
                    value=0x1,
                    mask=0x1,
                    column=0,
                    row=2,
                )

                in_task = shim_dma_single_bd_task(
                    "in_alloc", A, sizes=[1, 1, 1, N]
                )
                out_task = shim_dma_single_bd_task(
                    "out_alloc", C, sizes=[1, 1, 1, N], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)

        res = ctx.module.operation.verify()
        if not res:
            raise RuntimeError("Generated MLIR failed verification")
        return ctx.module


# ── Compile & Run ─────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic DMA with struct-based helpers",
    )
    parser.add_argument(
        "--device",
        choices=("npu", "npu1", "npu2"),
        default="npu2",
        help="Target device",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Directory for generated artifacts",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def pick_device(name: str):
    normalized = name.lower()
    if normalized == "npu2":
        return AIEDevice.npu2, "aie2p"
    return AIEDevice.npu1_1col, "aie2"


def compile_design(module, insts_path, xclbin_path, work_dir, verbose):
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(insts_path),
        xclbin_path=str(xclbin_path),
        work_dir=str(work_dir),
        verbose=verbose,
    )


def run_with_xrt(xclbin_path, insts_path, verbose):
    """Execute design on NPU and return output array."""
    npu_kernel = NPUKernel(
        str(xclbin_path),
        str(insts_path),
        kernel_name="MLIR_AIE",
    )
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    in_data = np.arange(1, N + 1, dtype=np.int32)
    in_buf = iron.tensor(in_data, dtype=np.int32)
    out_buf = iron.zeros(N, dtype=np.int32)

    DefaultNPURuntime.run(kernel_handle, [in_buf, out_buf])

    output = out_buf.numpy()
    return np.array(output)


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    dev, target_arch = pick_device(args.device)

    try:
        tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

        print(f"[1/4] Compiling PythoC kernel ({target_arch}) with struct helpers")

        kernel = PythocKernel(
            dynamic_dma_struct,
            [tensor_ty, tensor_ty, np.int32, np.int32, np.int32],
            target_arch=target_arch,
            extra_globals=_REGDB_GLOBALS,
            helpers=[program_bd_and_start, wait_for_lock],
        )
        print(f"      -> {kernel.bin_name}")

        print("[2/4] Building MLIR module with raw flows + dynamic DMA")
        module = build_mlir_module(dev, kernel)
        mlir_path = work_dir / "design.mlir"
        with open(mlir_path, "w") as f:
            print(module, file=f)
        print(f"      -> {mlir_path}")

        print("[3/4] Compiling design with aiecc")
        insts_path = work_dir / "insts.bin"
        xclbin_path = work_dir / "final.xclbin"
        compile_design(module, insts_path, xclbin_path, work_dir, args.verbose)
        print(f"      -> {xclbin_path}\n      -> {insts_path}")

        print("[4/4] Running with pyxrt and validating results")
        output = run_with_xrt(xclbin_path, insts_path, args.verbose)

        expected = np.arange(2, N + 2, dtype=np.int32)
        errors = 0
        for i in range(N):
            if output[i] != expected[i]:
                print(f"Error at [{i}]: got {output[i]}, expected {expected[i]}")
                errors += 1
                if errors >= 10:
                    print("  ... (more errors suppressed)")
                    break

        if errors == 0:
            print(f"      First elements: {output[:8]}")
            print(f"      Expected:       {expected[:8]}")
            print("PASS!")
            return 0
        else:
            total_wrong = np.sum(output != expected)
            print(f"\nFAILED: {total_wrong}/{N} elements incorrect")
            return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
