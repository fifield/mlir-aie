#!/usr/bin/env python3
# dynamic_dma_add_one.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --work-dir ./dynamic_dma_add_one_build | FileCheck %s
# CHECK: PASS!

"""Dynamic DMA programming test: core programs its own DMA via write_tm.

The PythoC kernel dynamically programs the compute tile's DMA buffer
descriptors and starts DMA channels using write_tm (processor bus writes).
This demonstrates that the core can control its own DMA at runtime rather
than relying on static MLIR-generated DMA configuration.

Flow:
  1. Host sends N int32 values to the compute tile via shim DMA.
  2. The kernel programs BD0 for S2MM ch0 (receive), starts the channel,
     and waits for the transfer to complete via lock-based signaling.
  3. The kernel processes data: out[i] = in[i] + 1.
  4. The kernel programs BD1 for MM2S ch0 (send), starts the channel,
     and waits for completion.
  5. Host receives the processed data and verifies correctness.

The compute tile's DMA is NOT statically configured by MLIR — only the
stream routing and shim DMA are set up at compile time.

AIE2P Compute Tile DMA Register Map (looked up from regdb):
  BD base:            DMA_BD0_0 (stride 0x20 per BD, 6 words each)
  S2MM_0_START_QUEUE: DMA_S2MM_0_Start_Queue (bits [3:0]=BD_ID, [23:16]=repeat_count)
  MM2S_0_START_QUEUE: DMA_MM2S_0_Start_Queue
  Lock N value:       Lock{N}_value
  Proc bus enable:    Core_Processor_Bus (bit 0)

BD word layout (compute tile):
  Word 0 [27:14]=BASE_ADDRESS (word addr), [13:0]=BUFFER_LENGTH (words)
  Word 1 [31]=COMPRESSION, [30]=PACKET, ...
  Word 2 [25:13]=D1_STEPSIZE, [12:0]=D0_STEPSIZE (stored as value-1)
  Word 3 [28:21]=D1_WRAP, [20:13]=D0_WRAP, [12:0]=D2_STEPSIZE
  Word 4 [24:19]=ITER_CUR, [18:13]=ITER_WRAP, [12:0]=ITER_STEP
  Word 5 [31]=TLAST_SUP, [30:27]=NEXT_BD, [26]=USE_NEXT_BD, [25]=VALID_BD,
         [24:18]=LOCK_REL_VALUE, [17:13]=LOCK_REL_ID, [12]=LOCK_ACQ_EN,
         [11:5]=LOCK_ACQ_VALUE, [3:0]=LOCK_ACQ_ID
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
from pythoc import ptr, i32
from pythoc.aie.operations import read_tm, write_tm

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "dynamic_dma_add_one_build"

# Number of int32 elements to process
N = 256

# ── Register addresses from register database ────────────────────────────────

_decoder = AIEAddressDecoder()
_reg = _decoder.get_register_offset

# DMA buffer descriptor registers (memory module)
DMA_BD0_0 = _reg("DMA_BD0_0", "memory")
DMA_BD0_1 = _reg("DMA_BD0_1", "memory")
DMA_BD0_2 = _reg("DMA_BD0_2", "memory")
DMA_BD0_3 = _reg("DMA_BD0_3", "memory")
DMA_BD0_4 = _reg("DMA_BD0_4", "memory")
DMA_BD0_5 = _reg("DMA_BD0_5", "memory")
DMA_BD1_0 = _reg("DMA_BD1_0", "memory")
DMA_BD1_1 = _reg("DMA_BD1_1", "memory")
DMA_BD1_2 = _reg("DMA_BD1_2", "memory")
DMA_BD1_3 = _reg("DMA_BD1_3", "memory")
DMA_BD1_4 = _reg("DMA_BD1_4", "memory")
DMA_BD1_5 = _reg("DMA_BD1_5", "memory")

# DMA channel start queue registers (memory module)
DMA_S2MM_0_START_QUEUE = _reg("DMA_S2MM_0_Start_Queue", "memory")
DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory")

# Lock value registers (memory module)
LOCK0_VALUE = _reg("Lock0_value", "memory")
LOCK1_VALUE = _reg("Lock1_value", "memory")

# Core processor bus enable register (core module)
CORE_PROCESSOR_BUS = _reg("Core_Processor_Bus", "core")

# Collect all register constants for PythoC kernel compilation
_REGDB_GLOBALS = {
    "DMA_BD0_0": DMA_BD0_0,
    "DMA_BD0_1": DMA_BD0_1,
    "DMA_BD0_2": DMA_BD0_2,
    "DMA_BD0_3": DMA_BD0_3,
    "DMA_BD0_4": DMA_BD0_4,
    "DMA_BD0_5": DMA_BD0_5,
    "DMA_BD1_0": DMA_BD1_0,
    "DMA_BD1_1": DMA_BD1_1,
    "DMA_BD1_2": DMA_BD1_2,
    "DMA_BD1_3": DMA_BD1_3,
    "DMA_BD1_4": DMA_BD1_4,
    "DMA_BD1_5": DMA_BD1_5,
    "DMA_S2MM_0_START_QUEUE": DMA_S2MM_0_START_QUEUE,
    "DMA_MM2S_0_START_QUEUE": DMA_MM2S_0_START_QUEUE,
    "LOCK0_VALUE": LOCK0_VALUE,
    "LOCK1_VALUE": LOCK1_VALUE,
}


# ── PythoC kernel ────────────────────────────────────────────────────────────


@aie_kernel
def dynamic_dma_add_one(
    in_buf: ptr[i32, True],
    out_buf: ptr[i32, True],
    in_addr_words: i32,
    out_addr_words: i32,
    num_words: i32,
):
    """Dynamically program tile DMA to receive/send data, with add-one processing.

    The kernel programs DMA buffer descriptors and starts DMA channels using
    write_tm (processor bus writes). Lock-based signaling is used to detect
    DMA completion.

    Args:
        in_buf:  Pointer to input buffer in tile local memory
        out_buf: Pointer to output buffer in tile local memory
        in_addr_words:  Input buffer base address in 32-bit word units
        out_addr_words: Output buffer base address in 32-bit word units
        num_words:      Number of 32-bit words to transfer/process
    """

    # ── Program BD0 for S2MM channel 0 (receive from stream) ──────────────

    # BD0 word 0: [27:14] BASE_ADDRESS, [13:0] BUFFER_LENGTH
    bd0_w0: i32 = (in_addr_words << 14) | num_words
    write_tm(bd0_w0, DMA_BD0_0)

    # BD0 words 1-4: defaults (no packet, contiguous 1D, no iteration)
    write_tm(0, DMA_BD0_1)
    write_tm(0, DMA_BD0_2)
    write_tm(0, DMA_BD0_3)
    write_tm(0, DMA_BD0_4)

    # BD0 word 5: VALID_BD=1, LOCK_REL_VALUE=+1, LOCK_REL_ID=0
    #   bit 25 = VALID_BD
    #   bits [24:18] = LOCK_REL_VALUE = 1
    #   bits [17:13] = LOCK_REL_ID = 0
    #   -> 0x02000000 | 0x00040000 = 0x02040000
    write_tm(0x02040000, DMA_BD0_5)

    # Start S2MM channel 0: Start_BD_ID=0 (bits [3:0]), Repeat_Count=0
    write_tm(0, DMA_S2MM_0_START_QUEUE)

    # Wait for S2MM completion: poll lock 0 value register
    # Lock 0 starts at 0; BD0 adds +1 when transfer finishes
    done: i32 = 0
    while done == 0:
        done = read_tm(LOCK0_VALUE)

    # ── Process data: add 1 to each element ───────────────────────────────

    i: i32 = 0
    while i < num_words:
        out_buf[i] = in_buf[i] + 1
        i = i + 1

    # ── Program BD1 for MM2S channel 0 (send to stream) ───────────────────

    # BD1 word 0: [27:14] BASE_ADDRESS, [13:0] BUFFER_LENGTH
    bd1_w0: i32 = (out_addr_words << 14) | num_words
    write_tm(bd1_w0, DMA_BD1_0)

    # BD1 words 1-4: defaults
    write_tm(0, DMA_BD1_1)
    write_tm(0, DMA_BD1_2)
    write_tm(0, DMA_BD1_3)
    write_tm(0, DMA_BD1_4)

    # BD1 word 5: VALID_BD=1, LOCK_REL_VALUE=+1, LOCK_REL_ID=1
    #   bit 25 = VALID_BD
    #   bits [24:18] = LOCK_REL_VALUE = 1
    #   bits [17:13] = LOCK_REL_ID = 1
    #   -> 0x02000000 | 0x00040000 | 0x00002000 = 0x02042000
    write_tm(0x02042000, DMA_BD1_5)

    # Start MM2S channel 0: Start_BD_ID=1 (bits [3:0])
    write_tm(1, DMA_MM2S_0_START_QUEUE)

    # Wait for MM2S completion: poll lock 1 value register
    done = 0
    while done == 0:
        done = read_tm(LOCK1_VALUE)


# ── Design construction ──────────────────────────────────────────────────────


def build_mlir_module(dev, kernel):
    """Build MLIR module using lower-level dialect API with raw flows."""
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            # Emit the external_func declaration for the kernel.
            # (IRON's Worker/Program calls this automatically, but the
            #  low-level @device/@core path requires an explicit call.)
            kernel.resolve()

            # Tile declarations
            t00 = tile(0, 0)  # Shim tile
            t02 = tile(0, 2)  # Compute tile

            # Local memory buffers on compute tile at known addresses
            # in_buf:  N words at byte address 0
            # out_buf: N words at byte address N*4
            in_buf = buffer(t02, datatype=tensor_ty, name="in_buf", address=0)
            out_buf = buffer(
                t02, datatype=tensor_ty, name="out_buf", address=N * 4
            )

            # Locks for DMA completion signaling (DMA BDs release these)
            lock(t02, lock_id=0, init=0, sym_name="s2mm_done")
            lock(t02, lock_id=1, init=0, sym_name="mm2s_done")

            # Circuit-switched stream routing
            flow(t00, WireBundle.DMA, 0, t02, WireBundle.DMA, 0)  # input
            flow(t02, WireBundle.DMA, 0, t00, WireBundle.DMA, 0)  # output

            # Shim DMA channel allocation (maps names to shim DMA channels)
            from aie.dialects.aie import shim_dma_allocation

            shim_dma_allocation("in_alloc", t00, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("out_alloc", t00, DMAChannelDir.S2MM, 0)

            # Core: call the PythoC kernel
            # in_addr_words = 0 (byte addr 0 / 4)
            # out_addr_words = N (byte addr N*4 / 4)
            @core(t02)
            def core_body():
                kernel(in_buf, out_buf, 0, N, N)

            # Runtime sequence (host-side DMA programming)
            @runtime_sequence(tensor_ty, tensor_ty)
            def sequence(A, C):
                # Enable processor bus on compute tile (0,2)
                npu_maskwrite32(
                    address=CORE_PROCESSOR_BUS,
                    value=0x1,
                    mask=0x1,
                    column=0,
                    row=2,
                )

                # Program shim DMA to send input and receive output
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
        description="Dynamic DMA programming via PythoC write_tm",
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

    # Input data: 1, 2, 3, ..., N
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

        print(f"[1/4] Compiling PythoC kernel ({target_arch})")
        kernel = PythocKernel(
            dynamic_dma_add_one,
            [tensor_ty, tensor_ty, np.int32, np.int32, np.int32],
            target_arch=target_arch,
            extra_globals=_REGDB_GLOBALS,
        )
        print(f"      -> {kernel.object_file_name}")

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

        # Verify: output should be input + 1
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
