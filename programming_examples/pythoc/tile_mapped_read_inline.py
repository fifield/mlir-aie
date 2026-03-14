#!/usr/bin/env python3
# tile_mapped_read_inline.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --work-dir ./tile_mapped_read_inline_build | FileCheck %s
# CHECK: PASS!

"""Single-file end-to-end example: PythoC inline kernel reading tile-mapped memory.

This demonstrates how to read hardware registers (lock values) from the
processor bus using PythoC's read_tm() operation, which maps to the
llvm.aie2.read.tm / llvm.aie2p.read.tm intrinsic for the Peano compiler.

The design:
  - Creates 8 locks on tile (0,2) with init values 42..49
  - Enables the processor bus via npu_maskwrite32
  - The kernel reads lock values via read_tm() and writes them to output
  - Host verifies output matches expected lock init values
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aie.dialects.aie import (
    AIEDevice,
    core,
    device,
    lock,
    object_fifo,
    object_fifo_link,
    tile,
    ObjectFifoPort,
)
from aie.dialects.aiex import (
    npu_dma_memcpy_nd,
    npu_maskwrite32,
    npu_sync,
    runtime_sequence,
)
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils.compile import compile_mlir_module
from aie.utils import DefaultNPURuntime, NPUKernel
import aie.iron as iron

# Import PythoC types and operations for inline kernel definition
from pythoc import ptr, i32, u32
from pythoc.aie.operations import read_tm

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "tile_mapped_read_inline_build"

# Constants matching the original test
IN_SIZE = 64
OUT_SIZE = 64
NUM_LOCKS = 8
LOCK_BASE_ID = 8
LOCK_INIT_BASE = 42  # Lock 8 = 42, lock 9 = 43, ..., lock 15 = 49

# Lock register address offset and stride (within the tile's register space)
# Lock value registers start at 0x1F080 with stride 0x10 between them
LOCK_REG_ADDR = 0x0001F080
LOCK_REG_STRIDE = 0x10

# Core Processor Bus enable register address for tile (0,2)
# For AIE2/AIE2P: address 0x32038, enable bit = 0x1
PROC_BUS_REG_ADDR = 0x32038


# ── PythoC kernel ────────────────────────────────────────────────────────────

@aie_kernel
def read_processor_bus(data: ptr[i32, True], addr: i32, size: i32, stride: i32):
    """Read values from the processor bus (tile-mapped memory).

    Reads 'size' 32-bit values starting at register address 'addr',
    with 'stride' bytes between consecutive registers.

    This is the PythoC equivalent of the Chess kernel that uses
    chess_storage(TM : 0x80000) for tile-mapped memory access.
    In Peano/llvm-aie, we use the read_tm() intrinsic instead.

    Args:
        data: Output buffer to store read values
        addr: Starting register address within the tile
        size: Number of values to read
        stride: Byte stride between consecutive register addresses
    """
    i: i32 = 0
    while i < size:
        offset: i32 = addr + (i * stride)
        val: i32 = read_tm(offset)
        data[i] = val
        i = i + 1


# ── Design construction ─────────────────────────────────────────────────────

def build_mlir_module(dev):
    """Build the MLIR module using lower-level Python API + PythoC kernel."""

    target_arch = "aie2p" if dev == AIEDevice.npu2 else "aie2"

    kernel = PythocKernel(
        read_processor_bus,
        [np.ndarray[(8,), np.dtype[np.int32]], np.int32, np.int32, np.int32],
        target_arch=target_arch,
    )

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            # Types
            memref_8xi32 = np.ndarray[(8,), np.dtype[np.int32]]
            memref_16xi32 = np.ndarray[(16,), np.dtype[np.int32]]
            memref_64xi32 = np.ndarray[(64,), np.dtype[np.int32]]

            # Emit the external_func declaration for the kernel.
            # (IRON's Worker/Program calls this automatically, but the
            #  low-level @device/@core path requires an explicit call.)
            kernel.resolve()

            # Tile declarations
            t00 = tile(0, 0)
            t01 = tile(0, 1)
            t02 = tile(0, 2)

            # ObjectFIFOs for data movement
            of_in0 = object_fifo("objFifo_in0", t00, t01, 2, memref_16xi32)
            of_in1 = object_fifo("objFifo_in1", t01, t02, 2, memref_8xi32)
            object_fifo_link(of_in0, of_in1)

            of_out1 = object_fifo("objFifo_out1", t02, t01, 2, memref_8xi32)
            of_out0 = object_fifo("objFifo_out0", t01, t00, 2, memref_16xi32)
            object_fifo_link(of_out1, of_out0)

            # Create 8 locks on tile (0,2) with init values 42 to 49
            # These lock values are what the core will read via the processor bus
            for i in range(NUM_LOCKS):
                lock(
                    t02,
                    lock_id=LOCK_BASE_ID + i,
                    init=LOCK_INIT_BASE + i,
                    sym_name=f"lock{LOCK_BASE_ID + i}",
                )

            # Core program: read lock values via processor bus and write to output
            @core(t02, kernel.bin_name)
            def core_body():
                for _ in range_(8):
                    of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    kernel(
                        elem_out, LOCK_REG_ADDR, NUM_LOCKS, LOCK_REG_STRIDE
                    )
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

            # Runtime sequence
            @runtime_sequence(memref_64xi32, memref_64xi32)
            def sequence(buf_in, buf_out):
                # Enable the Core Processor Bus register
                # Without this the core will hang on access to the processor bus
                npu_maskwrite32(
                    address=PROC_BUS_REG_ADDR,
                    value=0x1,
                    mask=0x1,
                    column=0,
                    row=2,
                )

                # Transfer data to/from the AIE array
                npu_dma_memcpy_nd(
                    metadata=of_in0,
                    bd_id=0,
                    mem=buf_in,
                    sizes=[1, 1, 1, IN_SIZE],
                )
                npu_dma_memcpy_nd(
                    metadata=of_out0,
                    bd_id=1,
                    mem=buf_out,
                    sizes=[1, 1, 1, OUT_SIZE],
                    issue_token=True,
                )
                npu_sync(
                    column=0, row=0, direction=0, channel=0
                )

        res = ctx.module.operation.verify()
        if not res:
            raise RuntimeError("Generated MLIR failed verification")
        return ctx.module


# ── Compile & Run ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PythoC inline kernel: tile-mapped read (processor bus)",
    )
    parser.add_argument(
        "--device",
        choices=("npu", "npu1", "npu2"),
        default="npu2",
        help="Target device: npu/npu1 (AIE2) or npu2 (AIE2P/Strix)",
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
    """Execute design on NPU using XRT and verify results."""
    npu_kernel = NPUKernel(
        str(xclbin_path),
        str(insts_path),
        kernel_name="MLIR_AIE",
    )
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    # Input buffer (used only for synchronization, values don't matter)
    in_data = np.full(IN_SIZE, 0xFABCDEF, dtype=np.int32)
    in_buf = iron.tensor(in_data, dtype=np.int32)
    out_buf = iron.zeros(OUT_SIZE, dtype=np.int32)

    DefaultNPURuntime.run(kernel_handle, [in_buf, out_buf])

    output = out_buf.numpy()
    return np.array(output)


def verify_results(output):
    """Verify that output contains the expected lock init values."""
    errors = 0
    for i in range(OUT_SIZE):
        expected = (i % NUM_LOCKS) + LOCK_INIT_BASE
        if output[i] != expected:
            print(f"Error at [{i}]: got {output[i]}, expected {expected}")
            errors += 1
    return errors


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    dev, target_arch = pick_device(args.device)

    try:
        print(
            f"[1/3] Building MLIR module with inline PythoC kernel ({target_arch})"
        )
        module = build_mlir_module(dev)
        mlir_path = work_dir / "design.mlir"
        with open(mlir_path, "w") as f:
            print(module, file=f)
        print(f"      -> {mlir_path}")

        print("[2/3] Compiling design with aiecc")
        insts_path = work_dir / "insts.bin"
        xclbin_path = work_dir / "final.xclbin"
        compile_design(module, insts_path, xclbin_path, work_dir, args.verbose)
        print(f"      -> {xclbin_path}\n      -> {insts_path}")

        print("[3/3] Running with pyxrt and validating results")
        output = run_with_xrt(xclbin_path, insts_path, args.verbose)
        errors = verify_results(output)

        if errors == 0:
            print(f"      Lock values read correctly: {output[:8]}")
            print("PASS!")
            return 0
        else:
            print(f"\nFAILED: {errors} errors")
            return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
