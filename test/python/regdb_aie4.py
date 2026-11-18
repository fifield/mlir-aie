#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

"""
Tests for the AIE4 (npu3) register database in aie.utils.regdb.

On AIE4 a memtile spans five 1MB address rows (registers, then 4MB of data
memory), so core tile row r sits at address row r + 4. Register offsets are
those of aie-codegen's xaie4gbl_params.h.
"""

from aie.utils.regdb import AIEAddressDecoder, MLIRModuleAnnotator

decoder = AIEAddressDecoder(arch="aie4")


def show(result, *keys):
    for key in keys:
        print(f"  {key}: {result.get(key)}")


# Core tile (0, 2) is address row 6: 0x600000 + 0xA0110
# CHECK-LABEL: Test 1: core Timer_Control
# CHECK: col: 0
# CHECK: row: 2
# CHECK: tile_type: core_tile
# CHECK: module: core
# CHECK: register: TIMER_CONTROL
print("\nTest 1: core Timer_Control")
show(decoder.parse_address(0x6A0110), "col", "row", "tile_type", "module", "register")

# CHECK-LABEL: Test 2: memory DMA_S2MM_0_Ctrl
# CHECK: col: 1
# CHECK: row: 3
# CHECK: module: memory
# CHECK: register: DMA_S2MM_0_CTRL
print("\nTest 2: memory DMA_S2MM_0_Ctrl")
show(
    decoder.parse_address((1 << 25) | (7 << 20) | 0xACE00),
    "col",
    "row",
    "module",
    "register",
)

# Memtile registers are in address row 1
# CHECK-LABEL: Test 3: memtile DMA_S2MM_0_BD0_0
# CHECK: row: 1
# CHECK: tile_type: memory_tile
# CHECK: register: DMA_S2MM_0_BD0_0
print("\nTest 3: memtile DMA_S2MM_0_BD0_0")
show(decoder.parse_address(0x108000), "row", "tile_type", "register")

# Memtile data memory starts in address row 2
# CHECK-LABEL: Test 4: memtile DataMemory
# CHECK: row: 1
# CHECK: register: DataMemory+0x100
# CHECK: is_memory_region: True
print("\nTest 4: memtile DataMemory")
show(decoder.parse_address(0x200100), "row", "register", "is_memory_region")

# CHECK-LABEL: Test 5: shim task queue
# CHECK: row: 0
# CHECK: tile_type: shim_tile
# CHECK: register: DMA_S2MM_0_TASK_QUEUE
print("\nTest 5: shim task queue")
show(decoder.parse_address(0x18544), "row", "tile_type", "register")

# Reverse lookup is case-insensitive and applies the address-row shift
# CHECK-LABEL: Test 6: encode_address
# CHECK: Timer_Control (0, 2): 0x6a0110
# CHECK: DMA_S2MM_0_Ctrl (0, 1): 0x109e00
# CHECK: Core_Control (2, 5): 0x49a3000
print("\nTest 6: encode_address")
print(f"  Timer_Control (0, 2): {hex(decoder.encode_address(0, 2, 'Timer_Control'))}")
print(
    f"  DMA_S2MM_0_Ctrl (0, 1): {hex(decoder.encode_address(0, 1, 'DMA_S2MM_0_Ctrl'))}"
)
print(f"  Core_Control (2, 5): {hex(decoder.encode_address(2, 5, 'Core_Control'))}")

# CHECK-LABEL: Test 7: bit fields
# CHECK: Bit Fields:
# CHECK-NEXT: [31] Enable_Token_Issue reset=0x0
# CHECK-NEXT: [27:16] Repeat_Count reset=0x0
# CHECK-NEXT: [3:0] Start_BD_ID reset=0x0
print("\nTest 7: bit fields")
print(decoder.format_result(decoder.parse_address(0x18544), show_bit_fields=True))

# CHECK-LABEL: Test 8: annotate npu3 module
# CHECK: annotated_count: 2
# CHECK: Tile(0, 0), Offset 0x18544, Name: DMA_S2MM_0_TASK_QUEUE | Value 0x10003: Enable_Token_Issue=0, Repeat_Count=1, Start_BD_ID=3
# CHECK: Tile(0, 2), Offset 0xA0110, Name: TIMER_CONTROL | Value 0x2500: Reset=0, Reset_Event=37
print("\nTest 8: annotate npu3 module")

from aie.ir import Context, Module, Location
from aie._mlir_libs import get_dialect_registry
from aie.dialects import aie as aiedialect
from aie.dialects import aiex as aiexdialect

mlir_source = """
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence() {
      %addr0 = arith.constant 0x18544 : i32
      %val0 = arith.constant 0x10003 : i32
      aiex.npu.write32(%addr0, %val0) {column = 0 : i32, row = 0 : i32} : i32, i32
      %addr1 = arith.constant 0x6A0110 : i32
      %val1 = arith.constant 0x2500 : i32
      aiex.npu.write32(%addr1, %val1) : i32, i32
    }
  }
}
"""

annotator = MLIRModuleAnnotator()
with Context() as ctx, Location.unknown():
    ctx.append_dialect_registry(get_dialect_registry())
    ctx.load_all_available_dialects()
    module = Module.parse(mlir_source)
    print(f"  annotated_count: {annotator.annotate_module(module)}")
    print(str(module))

# CHECK: All tests completed
print("\nAll tests completed")
