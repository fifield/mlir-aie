// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --verify-diagnostics

// -----
llvm.func @bad_value(%v: i64, %p: !llvm.ptr) {
  // expected-error @+1 {{operand #0 must be 32-bit signless integer}}
  "xllvm.intr.aie2.write.tm"(%v, %p) : (i64, !llvm.ptr) -> ()
  llvm.return
}

// -----
llvm.func @bad_pointer(%p: !llvm.ptr<1>) -> i32 {
  // expected-error @+1 {{operand #0 must be LLVM pointer in address space 0}}
  %r = "xllvm.intr.aie2.read.tm"(%p) : (!llvm.ptr<1>) -> i32
  llvm.return %r : i32
}

// -----
llvm.func @bad_result(%p: !llvm.ptr) -> i64 {
  // expected-error @+1 {{result #0 must be 32-bit signless integer}}
  %r = "xllvm.intr.aie2.read.tm"(%p) : (!llvm.ptr) -> i64
  llvm.return %r : i64
}

// -----
llvm.func @bad_value(%v: i64, %p: !llvm.ptr) {
  // expected-error @+1 {{operand #0 must be 32-bit signless integer}}
  "xllvm.intr.aie2p.write.tm"(%v, %p) : (i64, !llvm.ptr) -> ()
  llvm.return
}

// -----
llvm.func @bad_pointer(%p: !llvm.ptr<1>) -> i32 {
  // expected-error @+1 {{operand #0 must be LLVM pointer in address space 0}}
  %r = "xllvm.intr.aie2p.read.tm"(%p) : (!llvm.ptr<1>) -> i32
  llvm.return %r : i32
}

// -----
llvm.func @bad_result(%p: !llvm.ptr) -> i64 {
  // expected-error @+1 {{result #0 must be 32-bit signless integer}}
  %r = "xllvm.intr.aie2p.read.tm"(%p) : (!llvm.ptr) -> i64
  llvm.return %r : i64
}

// -----
llvm.func @bad_value(%v: i64, %p: !llvm.ptr) {
  // expected-error @+1 {{operand #0 must be 32-bit signless integer}}
  "xllvm.intr.aie4.write.tm"(%v, %p) : (i64, !llvm.ptr) -> ()
  llvm.return
}

// -----
llvm.func @bad_pointer(%p: !llvm.ptr<1>) -> i32 {
  // expected-error @+1 {{operand #0 must be LLVM pointer in address space 0}}
  %r = "xllvm.intr.aie4.read.tm"(%p) : (!llvm.ptr<1>) -> i32
  llvm.return %r : i32
}

// -----
llvm.func @bad_result(%p: !llvm.ptr) -> i64 {
  // expected-error @+1 {{result #0 must be 32-bit signless integer}}
  %r = "xllvm.intr.aie4.read.tm"(%p) : (!llvm.ptr) -> i64
  llvm.return %r : i64
}
