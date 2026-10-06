//===- AIEVecEnumBinding.td - Python enums ---------*- tablegen -*-===//
//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The Python generator rejects ShuffleModeAttr's `[` $value `]` format.
// Exclude that attribute but keep the ShuffleMode enum.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_ENUM_BINDING_TD
#define AIEVEC_ENUM_BINDING_TD

#define AIEVEC_PYTHON_ENUM_BINDINGS
include "aie/Dialect/AIEVec/IR/AIEVecAttributes.td"

#endif // AIEVEC_ENUM_BINDING_TD
