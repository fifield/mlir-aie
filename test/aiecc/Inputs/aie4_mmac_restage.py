#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Emit B0/B1/B0 staging with reused B0 values to test insert preservation."""

CHUNKS = 32
MMAC_ATTRS = "{zeroAcc, bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32}"

lines = [
    "llvm.func @restage(%b0: !llvm.ptr, %b1: !llvm.ptr, %a: vector<64xi8>,"
    " %out: !llvm.ptr) {",
    "  %poison = llvm.mlir.poison : vector<64xi32>",
]
for i in range(CHUNKS):
    lines.append(f"  %s{i} = llvm.mlir.constant({i} : i32) : i32")
for b in range(2):
    for i in range(CHUNKS):
        lines += [
            f"  %p{b}_{i} = llvm.getelementptr %b{b}[{i}]"
            " : (!llvm.ptr) -> !llvm.ptr, vector<128xi8>",
            f"  %v{b}_{i} = llvm.load %p{b}_{i} <alignment = 128>"
            " : !llvm.ptr -> vector<128xi8>",
        ]
for t, b in enumerate((0, 1, 0)):
    for i in range(CHUNKS):
        lines.append(
            f'  "xllvm.intr.aie4.insert.staging.int.I1024"(%v{b}_{i}, %s{i})'
            " {mode = 3 : i32} : (vector<128xi8>, i32) -> ()"
        )
    lines += [
        '  "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()',
        f'  %r{t} = "xllvm.intr.aie4.I512.ACC2048.mmac"(%poison, %a)'
        f" {MMAC_ATTRS} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>",
        f"  %o{t} = llvm.getelementptr %out[{t}]"
        " : (!llvm.ptr) -> !llvm.ptr, vector<64xi32>",
        f"  llvm.store %r{t}, %o{t} <alignment = 256> : vector<64xi32>, !llvm.ptr",
    ]
lines += ["  llvm.return", "}"]
print("\n".join(lines))
