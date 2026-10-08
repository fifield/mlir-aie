#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(MLIRAIEVecDialect
  AIEVecOps.cpp
  AIEVecAIE4Ops.cpp
  AIEVecTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/AIEVec

  DEPENDS
  MLIRAIEVecOpsIncGen
  MLIRAIEVecAttributesIncGen
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen

  LINK_LIBS PUBLIC
  AIE
  MLIRCallInterfaces
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRVectorDialect
  )

  add_subdirectory(AIE1)
