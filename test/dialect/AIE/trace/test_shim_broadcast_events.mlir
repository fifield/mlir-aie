// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-trace-to-config -split-input-file | FileCheck %s

// AIE4 shim broadcast channels use BROADCAST_n event names.
// CHECK-LABEL: aie.device(npu3)
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.trace @shim_trace(%shim) {
    aie.trace.mode "Event-Time"
    aie.trace.packet id=1 type="shimtile"
    aie.trace.event<"USER_EVENT_0">
    aie.trace.start broadcast=3
    aie.trace.stop broadcast=4
  }
  // CHECK-DAG: aie.trace.reg register = "Trace_A_Control0" field = "Trace_Start_Event" value = 235
  // CHECK-DAG: aie.trace.reg register = "Trace_A_Control0" field = "Trace_Stop_Event" value = 236
}

// -----

// AIE2 shim broadcast channels retain BROADCAST_A_n event names.
// CHECK-LABEL: aie.device(npu1_1col)
aie.device(npu1_1col) {
  %shim = aie.tile(0, 0)
  aie.trace @shim_trace(%shim) {
    aie.trace.mode "Event-Time"
    aie.trace.packet id=1 type="shimtile"
    aie.trace.event<"USER_EVENT_0">
    aie.trace.start broadcast=3
    aie.trace.stop broadcast=4
  }
  // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Start_Event" value = 113
  // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Stop_Event" value = 114
}
