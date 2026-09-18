// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s --implicit-check-not='aie.connect<{{.*}}Switch' --implicit-check-not='aie.flow('

// All eight restricted DMA loopbacks can route on the wide network, but need
// several rounds of negotiation to avoid each other's limited exits. Scaling
// historical overuse one-for-one with the width penalty previously accepted a
// 512 -> 32 -> 512 detour before that negotiation settled. A larger width
// penalty alone did not help because history grew proportionally faster too.
// No generated switchbox connection may use either side of the bridge.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.switchbox(%mem_tile_0_1) {
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 0>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 1>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 2>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 3>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 4>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 5>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 6>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 7>

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.flow(%mem, DMA : 0, %mem, DMA : 2)
    aie.flow(%mem, DMA : 1, %mem, DMA : 3)
    aie.flow(%mem, DMA : 2, %mem, DMA : 0)
    aie.flow(%mem, DMA : 3, %mem, DMA : 1)
    aie.flow(%mem, DMA : 4, %mem, DMA : 5)
    aie.flow(%mem, DMA : 5, %mem, DMA : 4)
    aie.flow(%mem, DMA : 6, %mem, DMA : 7)
    aie.flow(%mem, DMA : 7, %mem, DMA : 6)
  }
}

// -----

// Ten sources share the restricted memtile exits and return lanes, with two
// DMA destinations on the core. All routes can stay wide. Reducing the history
// scale by only a factor of two still took an avoidable bridge round trip for
// DMA:9 -> DMA:7 with the fixed penalty; a factor of four lets it settle.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.switchbox(%mem_tile_0_1) {
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 0>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 1>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 2>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 3>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 4>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 5>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 6>
// CHECK-DAG: aie.connect<{{[a-zA-Z]+}} : {{[0-9]+}}, DMA : 7>
// CHECK: aie.switchbox(%tile_0_2) {
// CHECK-DAG: aie.connect<South : {{[0-9]+}}, DMA : 0>
// CHECK-DAG: aie.connect<South : {{[0-9]+}}, DMA : 1>

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.flow(%mem, DMA : 0, %mem, DMA : 3)
    aie.flow(%mem, DMA : 1, %mem, DMA : 6)
    aie.flow(%mem, DMA : 2, %mem, DMA : 0)
    aie.flow(%mem, DMA : 3, %mem, DMA : 1)
    aie.flow(%mem, DMA : 4, %mem, DMA : 2)
    aie.flow(%mem, DMA : 5, %core, DMA : 0)
    aie.flow(%mem, DMA : 6, %mem, DMA : 5)
    aie.flow(%mem, DMA : 7, %mem, DMA : 4)
    aie.flow(%mem, DMA : 8, %core, DMA : 1)
    aie.flow(%mem, DMA : 9, %mem, DMA : 7)
  }
}
