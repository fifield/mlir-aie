// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b" --convert-func-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts | aie-translate --mlir-to-llvmir > %t.ll
// RUN: opt -S -passes='default<O1>' %t.ll | FileCheck %s --check-prefix=OPT
// RUN: opt -S -passes='default<O2>' %t.ll | FileCheck %s --check-prefix=OPT
// RUN: opt -S -passes='default<O3>' %t.ll | FileCheck %s --check-prefix=OPT
// RUN: llc -mtriple=aie4 -mcpu=aie4b -O0 %t.ll -o /dev/null
// RUN: opt -S -passes='default<O1>' %t.ll | llc -mtriple=aie4 -mcpu=aie4b -O2 -o - | FileCheck %s --check-prefix=ASM

// Keep native operations on each side of a mode-changing call at O1/O2/O3.
// Check O0 code generation and scheduled O2 helper calls for all six shapes;
// the helpers must emit the requested instruction without changing crFPOrder.
func.func private @change_modes()

// OPT-LABEL: define {{.*}} @native_bf16_32(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.512_20(
// OPT: call void @change_modes()
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.512_20(
// OPT: ret
// ASM-LABEL: native_bf16_32:
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.512_20
// ASM: jl{{.*}}#change_modes
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.512_20
func.func @native_bf16_32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> (vector<32xbf16>, vector<32xbf16>) {
  %r0 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<32xbf16>
  func.call @change_modes() : () -> ()
  %r1 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<32xbf16>
  return %r0, %r1 : vector<32xbf16>, vector<32xbf16>
}

// OPT-LABEL: define {{.*}} @native_bf16_64(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.1024_24(
// OPT: call void @change_modes()
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.1024_24(
// OPT: ret
// ASM-LABEL: native_bf16_64:
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.1024_24
// ASM: jl{{.*}}#change_modes
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddbfloat16.1024_24
func.func @native_bf16_64(%a: vector<64xbf16>, %b: vector<64xbf16>) -> (vector<64xbf16>, vector<64xbf16>) {
  %r0 = aievec.aie4.fminmax %a, %b[kind = <max>] : vector<64xbf16>
  func.call @change_modes() : () -> ()
  %r1 = aievec.aie4.fminmax %a, %b[kind = <max>] : vector<64xbf16>
  return %r0, %r1 : vector<64xbf16>, vector<64xbf16>
}

// OPT-LABEL: define {{.*}} @native_f16_32(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat16.512_41(
// OPT: call void @change_modes()
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat16.512_41(
// OPT: ret
// ASM-LABEL: native_f16_32:
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat16.512_41
// ASM: jl{{.*}}#change_modes
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat16.512_41
func.func @native_f16_32(%a: vector<32xf16>, %b: vector<32xf16>) -> (ui32, ui32) {
  %r0 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<32xf16>, ui32
  func.call @change_modes() : () -> ()
  %r1 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<32xf16>, ui32
  return %r0, %r1 : ui32, ui32
}

// OPT-LABEL: define {{.*}} @native_f16_64(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat16.1024_45(
// OPT: call void @change_modes()
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat16.1024_45(
// OPT: ret
// ASM-LABEL: native_f16_64:
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat16.1024_45
// ASM: jl{{.*}}#change_modes
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat16.1024_45
func.func @native_f16_64(%a: vector<64xf16>, %b: vector<64xf16>) -> (ui64, ui64) {
  %r0 = aievec.aie4.fcmp %a, %b[pred = <ge>] : vector<64xf16>, ui64
  func.call @change_modes() : () -> ()
  %r1 = aievec.aie4.fcmp %a, %b[pred = <ge>] : vector<64xf16>, ui64
  return %r0, %r1 : ui64, ui64
}

// OPT-LABEL: define {{.*}} @native_f32_16(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36(
// OPT: call void @change_modes()
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36(
// OPT: ret
// ASM-LABEL: native_f32_16:
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36
// ASM: jl{{.*}}#change_modes
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36
func.func @native_f32_16(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>, vector<16xf32>) {
  %r0 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<16xf32>
  func.call @change_modes() : () -> ()
  %r1 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<16xf32>
  return %r0, %r1 : vector<16xf32>, vector<16xf32>
}

// OPT-LABEL: define {{.*}} @native_f32_32(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.1024_49(
// OPT: call void @change_modes()
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.1024_49(
// OPT: ret
// ASM-LABEL: native_f32_32:
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat.1024_49
// ASM: jl{{.*}}#change_modes
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat.1024_49
func.func @native_f32_32(%a: vector<32xf32>, %b: vector<32xf32>) -> (ui32, ui32) {
  %r0 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<32xf32>, ui32
  func.call @change_modes() : () -> ()
  %r1 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<32xf32>, ui32
  return %r0, %r1 : ui32, ui32
}

// Check explicit mode writes as well as unknown calls. The final write must
// remain after the native calls. In assembly, earlier writes can be in a call's
// delay slots, which execute before entering its opaque helper.
// OPT-LABEL: define {{.*}} @mode_boundaries(
// OPT: call void @llvm.aie4.set.ctrl.reg(i32 47, i32 0)
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_49(
// OPT: call void @llvm.aie4.set.ctrl.reg(i32 47, i32 1)
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36(
// OPT: call {{.*}} @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_49(
// OPT: call void @llvm.aie4.set.ctrl.reg(i32 47, i32 0)
// OPT: ret
// ASM-LABEL: mode_boundaries:
// ASM: movx{{.*}}crfporder, #0
// ASM: movx{{.*}}crfporder, #1
// ASM: jl{{.*}}#__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_49
// ASM: movx{{.*}}crfporder, #0
func.func private @llvm.aie4.set.ctrl.reg(i32, i32)
func.func @mode_boundaries(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>, vector<16xf32>, ui32, ui32) {
  %reg = arith.constant 47 : i32
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  func.call @llvm.aie4.set.ctrl.reg(%reg, %zero) : (i32, i32) -> ()
  %m0 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<16xf32>
  %c0 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<16xf32>, ui32
  func.call @llvm.aie4.set.ctrl.reg(%reg, %one) : (i32, i32) -> ()
  %m1 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<16xf32>
  %c1 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<16xf32>, ui32
  func.call @llvm.aie4.set.ctrl.reg(%reg, %zero) : (i32, i32) -> ()
  return %m0, %m1, %c0, %c1 : vector<16xf32>, vector<16xf32>, ui32, ui32
}

// ASM-LABEL: __mlir_aie4_fporder_llvm.aie4.vaddbfloat16.512_20:
// ASM-NOT: crfporder
// ASM: vmin_ge.bf16
// ASM-NOT: crfporder
// ASM: .Lfunc_end

// ASM-LABEL: __mlir_aie4_fporder_llvm.aie4.vaddbfloat16.1024_24:
// ASM-NOT: crfporder
// ASM: vmax_lt.bf16
// ASM-NOT: crfporder
// ASM: .Lfunc_end

// ASM-LABEL: __mlir_aie4_fporder_llvm.aie4.vaddfloat16.512_41:
// ASM-NOT: crfporder
// ASM: vlt.fp16
// ASM-NOT: crfporder
// ASM: .Lfunc_end

// ASM-LABEL: __mlir_aie4_fporder_llvm.aie4.vaddfloat16.1024_45:
// ASM-NOT: crfporder
// ASM: vge.fp16
// ASM-NOT: crfporder
// ASM: .Lfunc_end

// ASM-LABEL: __mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36:
// ASM-NOT: crfporder
// ASM: vmin_ge.fp32
// ASM-NOT: crfporder
// ASM: .Lfunc_end

// ASM-LABEL: __mlir_aie4_fporder_llvm.aie4.vaddfloat.1024_49:
// ASM-NOT: crfporder
// ASM: vlt.fp32
// ASM-NOT: crfporder
// ASM: .Lfunc_end
// ASM-LABEL: __mlir_aie4_fporder_llvm.aie4.vaddfloat.512_49:
// ASM-NOT: crfporder
// ASM: vlt.fp32
// ASM-NOT: crfporder
// ASM: .Lfunc_end
