#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(AIE_REGDB_BUILD_DIR ${AIE_BINARY_DIR}/lib/regdb)
set(AIE_REGDB_FILES
  aie_registers_aie2.json)

set(AIE_REGDB_OUTPUTS)
foreach(regdb_file ${AIE_REGDB_FILES})
  set(src_file ${CMAKE_CURRENT_SOURCE_DIR}/${regdb_file})
  set(dst_file ${AIE_REGDB_BUILD_DIR}/${regdb_file})
  add_custom_command(
    OUTPUT ${dst_file}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${AIE_REGDB_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_file} ${dst_file}
    DEPENDS ${src_file}
    COMMENT "Copying ${regdb_file} to ${AIE_REGDB_BUILD_DIR}"
  )
  list(APPEND AIE_REGDB_OUTPUTS ${dst_file})
endforeach()

# Register databases generated from the aie-codegen register headers.
set(AIE_REGDB_HEADER_DIR
  ${AIE_SOURCE_DIR}/third_party/aie-codegen/aie-regdb/globalparams)
set(AIE_REGDB_AIE4_JSON ${AIE_REGDB_BUILD_DIR}/aie_registers_aie4.json)
add_custom_command(
  OUTPUT ${AIE_REGDB_AIE4_JSON}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${AIE_REGDB_BUILD_DIR}
  COMMAND ${Python3_EXECUTABLE}
    ${AIE_SOURCE_DIR}/utils/generate_registers_json.py
    -i ${AIE_REGDB_HEADER_DIR}/xaie4gbl_params.h
    -a aie4
    -o ${AIE_REGDB_AIE4_JSON}
  DEPENDS
    ${AIE_SOURCE_DIR}/utils/generate_registers_json.py
    ${AIE_REGDB_HEADER_DIR}/xaie4gbl_params.h
  COMMENT "Generating aie_registers_aie4.json from aie-codegen headers"
  VERBATIM
)
list(APPEND AIE_REGDB_OUTPUTS ${AIE_REGDB_AIE4_JSON})

add_custom_target(MLIRAIERegDBResources ALL DEPENDS ${AIE_REGDB_OUTPUTS})

add_mlir_dialect_library(MLIRAIEUtil
  AIERegisterDatabase.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRSupport
)

add_dependencies(MLIRAIEUtil MLIRAIERegDBResources GenerateAIEEventsTD)

install(FILES
  ${AIE_BINARY_DIR}/lib/regdb/aie_registers_aie2.json
  ${AIE_BINARY_DIR}/lib/regdb/aie_registers_aie4.json
  ${AIE_BINARY_DIR}/lib/regdb/events_database.json
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/regdb)
