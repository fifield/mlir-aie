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
// Separate resources model control/status dependencies independently of memory.
// SRS reads its mode registers and writes overflow status; MAC reads crVecSat.
// The SRS write effect prevents CSE. Identical MACs may be merged when no
// interfering effects occur between them.

/// `crSat` -- saturation mode consulted by SRS.
struct AIE4CRSatResource
    : public mlir::SideEffects::Resource::Base<AIE4CRSatResource> {
  llvm::StringRef getName() const final { return "AIE4CRSat"; }
};

/// `crRnd` -- rounding mode consulted by SRS.
struct AIE4CRRndResource
    : public mlir::SideEffects::Resource::Base<AIE4CRRndResource> {
  llvm::StringRef getName() const final { return "AIE4CRRnd"; }
};

/// `crSRSMode` -- SRS accumulator lane-width control.
struct AIE4CRSRSModeResource
    : public mlir::SideEffects::Resource::Base<AIE4CRSRSModeResource> {
  llvm::StringRef getName() const final { return "AIE4CRSRSMode"; }
};

/// `srSRS_of` -- overflow status written by SRS.
struct AIE4SRSOverflowResource
    : public mlir::SideEffects::Resource::Base<AIE4SRSOverflowResource> {
  llvm::StringRef getName() const final { return "AIE4SRSOverflow"; }
};

/// `crVecSat` -- vector saturation mode consulted by the integer VMAC.
struct AIE4CRVecSatResource
    : public mlir::SideEffects::Resource::Base<AIE4CRVecSatResource> {
  llvm::StringRef getName() const final { return "AIE4CRVecSat"; }
};

} // namespace xilinx::xllvm

#endif // AIE_DIALECT_XLLVM_AIE4CTRLREGRESOURCES_H
