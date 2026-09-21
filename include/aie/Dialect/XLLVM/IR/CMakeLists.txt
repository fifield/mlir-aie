#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect(XLLVM xllvm)
add_mlir_doc(XLLVMOps XLLVMOps ./ -gen-dialect-doc -dialect=xllvm)

# Add XLLVM operations
set(LLVM_TARGET_DEFINITIONS XLLVMOps.td)
mlir_tablegen(XLLVMOps.h.inc -gen-op-decls)
mlir_tablegen(XLLVMOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRXLLVMOpsIncGen)

# AIE4 config-word enums (amode/bmode). These are IntegerAttr-backed
# I32EnumAttrs, so they only need enum decls/defs -- no attribute registration.
mlir_tablegen(XLLVMEnums.h.inc -gen-enum-decls)
mlir_tablegen(XLLVMEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXLLVMEnumsIncGen)
mlir_tablegen(XLLVMConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRXLLVMConversionsIncGen)
add_dependencies(aie-headers MLIRXLLVMIncGen MLIRXLLVMOpsIncGen MLIRXLLVMConversionsIncGen MLIRXLLVMEnumsIncGen)
