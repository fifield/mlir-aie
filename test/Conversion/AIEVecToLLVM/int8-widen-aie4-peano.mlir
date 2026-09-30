// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verify native int8 widening and MAC code generation through Peano,
// including the unpack intrinsic signature and operand signedness.
//
// REQUIRES: peano
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" \
// RUN:   --convert-aievec-to-llvm="aie-target=aie4b" --convert-func-to-llvm \
// RUN:   --convert-to-llvm --reconcile-unrealized-casts -o %t.mlir
// RUN: aie-translate --mlir-to-llvmir %t.mlir -o %t.ll
// RUN: %PEANO_INSTALL_DIR/bin/opt -S -mtriple=aie4 %t.ll -o %t.opt.ll
// RUN: FileCheck %s --check-prefix=IR < %t.opt.ll
// RUN: %PEANO_INSTALL_DIR/bin/llc -O2 -mtriple=aie4 -mcpu=aie4b -filetype=obj %t.opt.ll -o %t.o
// RUN: %PEANO_INSTALL_DIR/bin/llvm-objdump -d --no-show-raw-insn %t.o | FileCheck %s --check-prefix=ASM

// IR-LABEL: define <32 x i32> @mac_i8
// IR-DAG:   call <32 x i16> @llvm.aie4.unpack.I512.I16.I8(<32 x i8> %{{.*}}, i32 1)
// IR:       call <32 x i32> @llvm.aie4.I512.I512.ACC1024.mac.ewise.conf(
// IR-NOT:   mul <32 x i32>
// ASM-LABEL: <mac_i8>:
// ASM-DAG:  vunpack{{.*}}unpacksign1
// ASM-DAG:  vmac
func.func @mac_i8(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// IR-LABEL: define <32 x i16> @ext_u8_i16
// IR:       call <32 x i16> @llvm.aie4.unpack.I512.I16.I8(<32 x i8> %{{.*}}, i32 0)
// ASM-LABEL: <ext_u8_i16>:
// ASM:      vunpack{{.*}}unpacksign0
func.func @ext_u8_i16(%a: vector<32xi8>) -> vector<32xi16> {
  %u = arith.extui %a : vector<32xi8> to vector<32xi16>
  return %u : vector<32xi16>
}

// An extension used outside the MAC also needs standalone native widening.
// IR-LABEL: define {{.*}} @mac_i8_multiuse
// IR-NOT:   sext
// IR:       call <32 x i32> @llvm.aie4.I512.I512.ACC1024.mac.ewise.conf(
// ASM-LABEL: <mac_i8_multiuse>:
// ASM-DAG:  vunpack{{.*}}unpacksign1
// ASM-DAG:  vmac
// ASM-DAG:  vups{{.*}}upssign1
func.func @mac_i8_multiuse(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r, %ea : vector<32xi32>, vector<32xi32>
}

// IR-LABEL: define {{.*}} @mac_i16_multiuse
// IR-NOT:   sext
// IR:       call <32 x i32> @llvm.aie4.I512.I512.ACC1024.mac.ewise.conf(
// ASM-LABEL: <mac_i16_multiuse>:
// ASM-DAG:  vmac
// ASM-DAG:  vups{{.*}}upssign1
func.func @mac_i16_multiuse(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {
  %ea = arith.extsi %a : vector<32xi16> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r, %ea : vector<32xi32>, vector<32xi32>
}

// Unsigned and mixed-sign i8 MACs unpack each operand with its own signedness
// before the signed i16 VMAC; both i8 ranges fit in signed i16.
// IR-LABEL: define <32 x i32> @mac_u8_s8
// IR-DAG:   call <32 x i16> @llvm.aie4.unpack.I512.I16.I8(<32 x i8> %0, i32 0)
// IR-DAG:   call <32 x i16> @llvm.aie4.unpack.I512.I16.I8(<32 x i8> %1, i32 1)
// IR:       call <32 x i32> @llvm.aie4.I512.I512.ACC1024.mac.ewise.conf(
// IR-NOT:   mul <32 x i32>
// ASM-LABEL: <mac_u8_s8>:
// ASM-DAG:  vunpack{{.*}}unpacksign0
// ASM-DAG:  vunpack{{.*}}unpacksign1
// ASM-DAG:  vmac
func.func @mac_u8_s8(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extui %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// IR-LABEL: define <32 x i32> @mac_u8_u8
// IR-DAG:   call <32 x i16> @llvm.aie4.unpack.I512.I16.I8(<32 x i8> %0, i32 0)
// IR-DAG:   call <32 x i16> @llvm.aie4.unpack.I512.I16.I8(<32 x i8> %1, i32 0)
// IR:       call <32 x i32> @llvm.aie4.I512.I512.ACC1024.mac.ewise.conf(
// IR-NOT:   mul <32 x i32>
// ASM-LABEL: <mac_u8_u8>:
// ASM-DAG:  vunpack{{.*}}unpacksign0
// ASM-DAG:  vmac
func.func @mac_u8_u8(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extui %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extui %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}
