<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# AIE4 Kernels

AIE4-specific kernel sources used by `programming_examples/` Makefiles when
`devicename=npu3` (medusa). Compiled with `PEANOWRAP4_FLAGS` /
`CHESSCCWRAP4_FLAGS` from `programming_examples/makefile-common`.

Kernels are added here on demand as programming examples are extended to
support `npu3`. Where an `aie2p` source compiles unchanged for AIE4, a copy or
small adaptation is preferred over indirect includes so each architecture
directory remains self-contained and grep-able.
