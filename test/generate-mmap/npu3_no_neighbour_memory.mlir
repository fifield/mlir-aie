//===- npu3_no_neighbour_memory.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An AIE4 core cannot access neighbour data memory: only the east window
// (0xE0000) maps its own tile; south/west/north are reserved.

// RUN: aie-translate --aie-generate-bcf --tilecol=1 --tilerow=3 %s | FileCheck %s --check-prefix=BCF
// RUN: aie-translate --aie-generate-ldscript --tilecol=1 --tilerow=3 %s | FileCheck %s --check-prefix=LD
// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s --check-prefix=MMAP

// BCF-LABEL: // mapping neighbors tile memory
// BCF-NEXT: _reserved DMb 0x80000 0x20000  // No tile with memory exists to the south.
// BCF-NEXT: _reserved DMb 0xA0000 0x20000  // No tile with memory exists to the west.
// BCF-NEXT: _reserved DMb 0xC0000 0x20000  // No tile with memory exists to the north.
// BCF-NEXT: // east -------------------------------------------------
// BCF-NEXT: _symbol own 0xE0400 64
// BCF-NEXT: _extern own
// BCF-NEXT: _reserved DMb 0xE0400 64
// BCF-EMPTY:
// BCF-NEXT: // end mapping neighbors tile memory
// BCF-NOT: south
// BCF-NOT: west
// BCF-NOT: north

// LD-LABEL: _sp_start_value_DM_stack = .;
// LD-NEXT: . += 0x400; /* stack */
// LD-NEXT: /* No tile with memory exists to the south. */
// LD-NEXT: . = 0x80000;
// LD-NEXT: . += 0x20000;
// LD-NEXT: /* No tile with memory exists to the west. */
// LD-NEXT: . = 0xA0000;
// LD-NEXT: . += 0x20000;
// LD-NEXT: /* No tile with memory exists to the north. */
// LD-NEXT: . = 0xC0000;
// LD-NEXT: . += 0x20000;
// LD-NEXT: . = 0xE0400;
// LD-NEXT: own = .;
// LD-NEXT: . += 0x40;
// LD-NEXT: .bss

// MMAP-LABEL: // Tile(0, 3)
// MMAP-NEXT: // Memory map: name base_address num_bytes
// MMAP-NEXT: _symbol west 0xE0000 64
// MMAP-NEXT: // Tile(1, 1)
// MMAP-NEXT: // Memory map: name base_address num_bytes
// MMAP-NEXT: // Tile(1, 2)
// MMAP-NEXT: // Memory map: name base_address num_bytes
// MMAP-NEXT: _symbol south 0xE0000 64
// MMAP-NEXT: // Tile(1, 3)
// MMAP-NEXT: // Memory map: name base_address num_bytes
// MMAP-NEXT: _symbol own 0xE0400 64
// MMAP-NEXT: // Tile(1, 4)
// MMAP-NEXT: // Memory map: name base_address num_bytes
// MMAP-NEXT: _symbol north 0xE0000 64
// MMAP-NEXT: // Tile(2, 3)
// MMAP-NEXT: // Memory map: name base_address num_bytes
// MMAP-NEXT: _symbol east 0xE0000 64
// MMAP-NOT: _symbol

module {
  aie.device(npu3) {
    %t03 = aie.tile(0, 3)
    %t11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    %t23 = aie.tile(2, 3)
    %west = aie.buffer(%t03) {sym_name = "west", address = 0 : i32} : memref<16xi32>
    %south = aie.buffer(%t12) {sym_name = "south", address = 0 : i32} : memref<16xi32>
    %own = aie.buffer(%t13) {sym_name = "own", address = 0x400 : i32} : memref<16xi32>
    %north = aie.buffer(%t14) {sym_name = "north", address = 0 : i32} : memref<16xi32>
    %east = aie.buffer(%t23) {sym_name = "east", address = 0 : i32} : memref<16xi32>
    %core = aie.core(%t13) {
      aie.end
    } {stack_size = 0x400 : i32}
  }
}
