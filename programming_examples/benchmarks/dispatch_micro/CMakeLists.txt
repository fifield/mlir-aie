# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../common.cmake)

project(dispatch_micro_bench)

add_executable(bench bench.cpp)

target_include_directories(bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${XRT_INC_DIR}
)

target_link_directories(bench PRIVATE ${XRT_LIB_DIR})

target_link_libraries(bench PRIVATE
  xrt_coreutil
  pthread
  dl
)
