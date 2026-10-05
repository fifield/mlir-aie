// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUTE
// RUN: aie-opt %s --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows | aie-translate --aie-generate-xaie | FileCheck %s --check-prefix=XAIE
// RUN: aie-opt %s --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows --convert-aie-to-transaction --canonicalize | FileCheck %s --check-prefix=TXN

// Both control DMA channels inject through their own circuit-switched ports.
// ROUTE-DAG: aie.connect<DMAControl : 0, {{.*}}>
// ROUTE-DAG: aie.connect<DMAControl : 1, {{.*}}>
// ROUTE-DAG: aie.masterset(TileControl : 0, {{.*}})
// XAIE-DAG: XAie_StrmConnCctEnable({{.*}}, DMA_CTRL, 0,
// XAIE-DAG: XAie_StrmConnCctEnable({{.*}}, DMA_CTRL, 1,
// The wide-switch source selectors for control DMA are 26 and 27.
// TXN-DAG: %[[CTRL0:.*]] = arith.constant -2147483622 : i32
// TXN-DAG: %[[CTRL1:.*]] = arith.constant -2147483621 : i32
// TXN-DAG: aiex.npu.write32(%{{.*}}, %[[CTRL0]])
// TXN-DAG: aiex.npu.write32(%{{.*}}, %[[CTRL1]])

aie.device(npu3_1col) {
  %shim = aie.tile(0, 0)
  %core = aie.tile(0, 3)
}
