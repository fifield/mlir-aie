# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(CMAKE_CXX_STANDARD 17)

include(CTest)

add_executable(target_model  target_model.cpp)
add_executable(target_model_rtti  target_model_rtti.cpp)
add_executable(target_model_npu3  target_model_npu3.cpp)
add_executable(target_model_bd_pools  target_model_bd_pools.cpp)
add_executable(target_model_bd_layout_npu3  target_model_bd_layout_npu3.cpp)
target_include_directories(target_model_bd_layout_npu3 PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/aie-codegen/aie-regdb/globalparams)
target_link_libraries(target_model_bd_pools PRIVATE AIETransforms)
add_executable(target_model_dma_channels  target_model_dma_channels.cpp)
target_link_libraries(target_model_dma_channels PRIVATE AIETransforms)
add_executable(target_model_unknown_device  target_model_unknown_device.cpp)
add_executable(target_model_tile_address  target_model_tile_address.cpp)
add_executable(target_model_local_bases  target_model_local_bases.cpp)
add_executable(dma_channel_capacity  dma_channel_capacity.cpp)
llvm_update_compile_flags(dma_channel_capacity)
add_executable(taken_bds_pools  taken_bds_pools.cpp)
llvm_update_compile_flags(taken_bds_pools)
target_link_libraries(taken_bds_pools PRIVATE AIETransforms AIEXUtils AIEX MLIRParser)
add_executable(register_database  register_database.cpp)
add_executable(bd_register_base  bd_register_base.cpp)
llvm_update_compile_flags(bd_register_base)
# BdLowering (AIEXUtils) uses helpers of the AIEX dialect library.
target_link_libraries(bd_register_base PRIVATE AIEXUtils AIEX)
add_executable(device_cache device_cache.cpp)
add_executable(stream_wait_graph stream_wait_graph.cpp)
llvm_update_compile_flags(stream_wait_graph)
target_link_libraries(stream_wait_graph PRIVATE AIETransforms MLIRParser)
add_test(NAME StreamWaitGraph COMMAND stream_wait_graph)
# aiecc's headers define cl::opt globals, whose typeinfo must match LLVM's
# RTTI setting; take aiecc's compile flags so rtti=OFF builds link.
llvm_update_compile_flags(device_cache)
target_include_directories(device_cache PRIVATE ${CMAKE_SOURCE_DIR}/tools/aiecc)
target_link_libraries(device_cache PRIVATE MLIRParser)
add_test(NAME DeviceCache COMMAND device_cache)
add_test(NAME TargetModel COMMAND target_model)
add_test(NAME TargetModelRtti COMMAND target_model_rtti)
add_test(NAME TargetModelNPU3 COMMAND target_model_npu3)
add_test(NAME TargetModelBdPools COMMAND target_model_bd_pools)
add_test(NAME TargetModelBdLayoutNPU3 COMMAND target_model_bd_layout_npu3)
add_test(NAME TargetModelDmaChannels COMMAND target_model_dma_channels)
add_test(NAME TargetModelUnknownDevice COMMAND target_model_unknown_device)
add_test(NAME TargetModelTileAddress COMMAND target_model_tile_address)
add_test(NAME TargetModelLocalBases COMMAND target_model_local_bases)
add_test(NAME DMAChannelCapacity COMMAND dma_channel_capacity)
add_test(NAME TakenBdsPools COMMAND taken_bds_pools)
add_test(NAME RegisterDatabase COMMAND register_database WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME BdRegisterBase COMMAND bd_register_base)

# Exercises the uint16 fallback path of test_utils::bfloat16_t on every
# platform by force-undefining __STDCPP_BFLOAT16_T__ for this TU only.
add_executable(bfloat16_fallback_test bfloat16_fallback_test.cpp)
target_include_directories(bfloat16_fallback_test
                           PRIVATE ${CMAKE_SOURCE_DIR}/runtime_lib/test_lib)
if(MSVC)
  target_compile_options(bfloat16_fallback_test PRIVATE /U__STDCPP_BFLOAT16_T__)
else()
  target_compile_options(bfloat16_fallback_test PRIVATE -U__STDCPP_BFLOAT16_T__)
endif()
add_test(NAME Bfloat16Fallback COMMAND bfloat16_fallback_test)

# The aievec.shuffle mode table against the permutations measured on AIE2P.
add_executable(aievec_shuffle_modes aievec_shuffle_modes.cpp)
target_link_libraries(aievec_shuffle_modes PRIVATE MLIRAIEVecDialect AIE)
add_test(NAME AIEVecShuffleModes
         COMMAND aievec_shuffle_modes
                 ${CMAKE_CURRENT_SOURCE_DIR}/aie2p_vshuffle_modes.json)

# ELF fixtures and checks are in test/aiecc/stack_size_analysis.test.
add_executable(stack_size_analysis stack_size_analysis.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/aiecc/StackSizeAnalysis.cpp)
target_include_directories(stack_size_analysis PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/aiecc)
llvm_map_components_to_libnames(stack_size_analysis_libs Object IRReader Core Support)
target_link_libraries(stack_size_analysis PRIVATE ${stack_size_analysis_libs})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(EXECUTABLES target_model target_model_rtti target_model_npu3
    target_model_bd_pools target_model_bd_layout_npu3 target_model_dma_channels
    target_model_unknown_device target_model_tile_address
    target_model_local_bases
    dma_channel_capacity taken_bds_pools register_database device_cache
    bd_register_base)

set(CPP_TEST_DEPENDS ${EXECUTABLES} bfloat16_fallback_test stack_size_analysis
    aievec_shuffle_modes stream_wait_graph)
add_custom_target(check-aie-cpp COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS ${CPP_TEST_DEPENDS})

foreach(executable ${EXECUTABLES} aievec_shuffle_modes stream_wait_graph)
    target_link_libraries(${executable}
                        PUBLIC
                        AIE
                        ${dialect_libs})
endforeach()

add_dependencies(check-aie check-aie-cpp)
