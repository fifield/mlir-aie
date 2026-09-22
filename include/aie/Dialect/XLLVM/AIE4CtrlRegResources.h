/* Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

//===- AIE4CtrlRegResources.h - AIE4 control-register resources ----C++ -*-===//
//
// MLIR side-effect resources naming the AIE4 control/status registers that
// some AIE4 vector instructions implicitly read or write.
//
// Shared by AIEVec and XLLVM to give both dialects the same resource identities.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_AIE4CTRLREGRESOURCES_H
#define AIE_DIALECT_XLLVM_AIE4CTRLREGRESOURCES_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace xilinx::xllvm {

//===----------------------------------------------------------------------===//
// AIE4 control/status register side-effect resources
//===----------------------------------------------------------------------===//
//
// MAC reads crVecSat; identical MACs may be merged without intervening writes.
// SRS and UPS omit effect declarations so MLIR treats them conservatively.

/// `crVecSat` -- vector saturation mode consulted by the integer VMAC.
struct AIE4CRVecSatResource
    : public mlir::SideEffects::Resource::Base<AIE4CRVecSatResource> {
  llvm::StringRef getName() const final { return "AIE4CRVecSat"; }
};

} // namespace xilinx::xllvm

#endif // AIE_DIALECT_XLLVM_AIE4CTRLREGRESOURCES_H
