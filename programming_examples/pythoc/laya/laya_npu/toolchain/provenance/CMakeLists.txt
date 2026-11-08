###############################################################################
# NPU Development Tools - Simplified Build Configuration
# 
# This is a simplified mono-repo build system for NPU development tools.
# It builds LLVM/MLIR-based tools from source using CMake ExternalProject.
#
# Key features:
# - Native builds only (no cross-compilation)
# - Python venv-based dependency management
# - Ninja build system for speed
# - Clear dependency chain between projects
# - AI-friendly: simple, well-documented, easy to understand
###############################################################################

cmake_minimum_required(VERSION 3.20)

project(npu-dev-tools)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Ninja generator
if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(WARNING "This project is optimized for Ninja. Consider using: cmake -G Ninja ...")
endif()

# Build configuration
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release/Debug)")

# Installation directory - default to install/ in the repo
set(NPU_INSTALL_DIR "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Installation directory")

# Build directory for all subprojects
set(NPU_BUILD_DIR "${CMAKE_BINARY_DIR}")

# The MLIR wheel is installed via pip during setup.py
# - mlir_aie: Base MLIR/LLVM (can optionally build from llvm-project source)
# - llvm-aie: Peano compiler (built from source by build.sh)
option(USE_MLIR_WHEEL "Use mlir_aie wheel instead of building llvm-project from source" ON)

# Find Python3 (should be from venv if activated)
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python: ${Python3_EXECUTABLE}")

# Detect pybind11 from Python environment
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
    RESULT_VARIABLE PYBIND11_STATUS
    OUTPUT_VARIABLE PYBIND11_DIR
    ERROR_QUIET
)
if(NOT PYBIND11_STATUS EQUAL "0")
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()
message(STATUS "Found pybind11: ${PYBIND11_DIR}")

# Detect PyTorch from Python environment (optional, needed for torch-mlir)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch;print(torch.utils.cmake_prefix_path, end='')"
    RESULT_VARIABLE TORCH_STATUS
    OUTPUT_VARIABLE TORCH_DIR
    ERROR_QUIET
)
if(TORCH_STATUS EQUAL "0")
    message(STATUS "Found PyTorch: ${TORCH_DIR}")
else()
    message(STATUS "PyTorch not found - torch-mlir will not be built")
    message(STATUS "To build torch-mlir, install PyTorch: pip install torch")
endif()

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Include ExternalProject module
include(ExternalProject)

# Common CMake arguments for all projects
set(COMMON_CMAKE_ARGS
    -G Ninja
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -Dpybind11_DIR=${PYBIND11_DIR}
)

# Common ExternalProject settings
set(COMMON_EP_ARGS
    BUILD_ALWAYS FALSE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
    USES_TERMINAL_INSTALL TRUE
)

###############################################################################
# LLVM/MLIR Installation
###############################################################################

if(USE_MLIR_WHEEL)
    # Use MLIR wheel extracted by setup.py
    message(STATUS "Using MLIR wheel (USE_MLIR_WHEEL=ON)")
    
    set(LLVM_INSTALL_DIR "${NPU_INSTALL_DIR}/mlir")
    set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm")
    set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
    set(CLANG_DIR "${LLVM_INSTALL_DIR}/lib/cmake/clang")
    
    # Verify MLIR wheel is installed
    if(NOT EXISTS "${MLIR_DIR}")
        message(FATAL_ERROR 
            "MLIR wheel not found at ${LLVM_INSTALL_DIR}\n"
            "Please run 'python3 setup.py' first to install the MLIR wheel")
    endif()
    
    message(STATUS "Using MLIR wheel at: ${LLVM_INSTALL_DIR}")
    
else()
    # Build LLVM/MLIR from llvm-project source
    message(STATUS "Building LLVM/MLIR from source (USE_MLIR_WHEEL=OFF)")
    
    set(LLVM_INSTALL_DIR "${NPU_INSTALL_DIR}/llvm")
    set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm")
    set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
    set(CLANG_DIR "${LLVM_INSTALL_DIR}/lib/cmake/clang")
    
    ExternalProject_Add(llvm-project
        PREFIX ${NPU_BUILD_DIR}/llvm-project-build
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/llvm-project/llvm
        BINARY_DIR ${NPU_BUILD_DIR}/llvm-project
        CMAKE_ARGS
            ${COMMON_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
            -DLLVM_ENABLE_PROJECTS=mlir$<SEMICOLON>clang
            -DLLVM_TARGETS_TO_BUILD=X86$<SEMICOLON>ARM$<SEMICOLON>AArch64
            -DLLVM_ENABLE_ASSERTIONS=ON
            -DLLVM_ENABLE_RTTI=ON
            -DLLVM_BUILD_EXAMPLES=OFF
            -DLLVM_INCLUDE_EXAMPLES=OFF
            -DLLVM_BUILD_TESTS=ON
            -DLLVM_INCLUDE_TESTS=ON
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON
            -DLLVM_INSTALL_UTILS=ON
        ${COMMON_EP_ARGS}
        STEP_TARGETS build install
    )
endif()

###############################################################################
# Peano (llvm-aie) - Built from source by scripts/build-llvm-aie.sh
###############################################################################

set(PEANO_INSTALL_DIR "${NPU_INSTALL_DIR}/llvm-aie")

if(NOT EXISTS "${PEANO_INSTALL_DIR}/bin/llc")
    message(FATAL_ERROR
        "llvm-aie (Peano) not found at ${PEANO_INSTALL_DIR}.\n"
        "It is built from source before cmake runs. Please run ./build.sh or:\n"
        "  ./scripts/build-llvm-aie.sh")
endif()

message(STATUS "Found llvm-aie (Peano) at: ${PEANO_INSTALL_DIR}")

###############################################################################
# MLIR-AIE (AIE dialect and tools)
###############################################################################

set(MLIR_AIE_INSTALL_DIR "${NPU_INSTALL_DIR}/mlir-aie")

# Add dependency on llvm-project if building from source
if(NOT USE_MLIR_WHEEL)
    set(MLIR_AIE_DEPENDS llvm-project-install)
endif()

set(VITIS_COMPONENTS "AIE2;AIE2P")

ExternalProject_Add(mlir-aie
    PREFIX ${NPU_BUILD_DIR}/mlir-aie-build
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/mlir-aie
    BINARY_DIR ${NPU_BUILD_DIR}/mlir-aie
    DEPENDS ${MLIR_AIE_DEPENDS}
    CMAKE_ARGS
        ${COMMON_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=${MLIR_AIE_INSTALL_DIR}
        -DLLVM_DIR=${LLVM_DIR}
        -DMLIR_DIR=${MLIR_DIR}
        -DClang_DIR=${CLANG_DIR}
        -DLLVM_EXTERNAL_LIT=/work/npu-dev/venv/bin/lit
        -DPEANO_INSTALL_DIR=${PEANO_INSTALL_DIR}
        -DCMAKE_MODULE_PATH=${CMAKE_SOURCE_DIR}/mlir-aie/cmake/modulesXilinx
    CMAKE_CACHE_ARGS
        -DAIE_VITIS_COMPONENTS:STRING=${VITIS_COMPONENTS}
    ${COMMON_EP_ARGS}
    STEP_TARGETS build install
)

###############################################################################
# MLIR-AIR (AIR dialect and runtime)
###############################################################################

set(MLIR_AIR_INSTALL_DIR "${NPU_INSTALL_DIR}/mlir-air")

# mlir-air is not a default submodule (see SUBMODULE_META in setup.py); only
# build it when its source is actually present, like the other optional
# projects below.  Guarding on EXISTS keeps a mlir-aie/PythoC-only checkout
# (the CI default) from failing configure with "source directory is not an
# existing non-empty directory".
if(EXISTS "${CMAKE_SOURCE_DIR}/mlir-air/CMakeLists.txt")
    ExternalProject_Add(mlir-air
        PREFIX ${NPU_BUILD_DIR}/mlir-air-build
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/mlir-air
        BINARY_DIR ${NPU_BUILD_DIR}/mlir-air
        DEPENDS mlir-aie-install
        CMAKE_ARGS
            ${COMMON_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=${MLIR_AIR_INSTALL_DIR}
            -DCMAKE_MODULE_PATH=${CMAKE_SOURCE_DIR}/mlir-aie/cmake/modulesXilinx
            -DLLVM_DIR=${LLVM_DIR}
            -DMLIR_DIR=${MLIR_DIR}
            -DClang_DIR=${CLANG_DIR}
            -DXRT_ROOT=/opt/xilinx/xrt
            -DENABLE_RUN_XRT_TESTS=ON
            -DLLVM_EXTERNAL_LIT=/work/npu-dev/venv/bin/lit
            -DAIE_DIR=${MLIR_AIE_INSTALL_DIR}/lib/cmake/aie
        CMAKE_CACHE_ARGS
            -DPEANO_INSTALL_DIR:STRING=${PEANO_INSTALL_DIR}
            -DAIR_RUNTIME_TARGETS:STRING=x86_64
            -Dx86_64_TOOLCHAIN_FILE:FILEPATH=${CMAKE_SOURCE_DIR}/mlir-air/cmake/modules/toolchain_x86_64.cmake
        ${COMMON_EP_ARGS}
        STEP_TARGETS build install
    )
    set(MLIR_AIR_ENABLED TRUE)
else()
    set(MLIR_AIR_ENABLED FALSE)
endif()

###############################################################################
# Torch-MLIR (PyTorch to MLIR conversion)
###############################################################################

if(TORCH_STATUS EQUAL "0")
    set(TORCH_MLIR_INSTALL_DIR "${NPU_INSTALL_DIR}/torch-mlir")

    ExternalProject_Add(torch-mlir
        PREFIX ${NPU_BUILD_DIR}/torch-mlir-build
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/torch-mlir
        BINARY_DIR ${NPU_BUILD_DIR}/torch-mlir
        CMAKE_ARGS
            ${COMMON_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=${TORCH_MLIR_INSTALL_DIR}
            -DLLVM_DIR=${LLVM_DIR}
            -DMLIR_DIR=${MLIR_DIR}
            -DTorch_DIR=${TORCH_DIR}
        ${COMMON_EP_ARGS}
        STEP_TARGETS build install
    )
endif()

###############################################################################
# Triton-XDNA (AMD Triton compiler for XDNA/AIE/AIR workflows)
###############################################################################

if(EXISTS "${CMAKE_SOURCE_DIR}/triton-xdna/CMakeLists.txt")
    set(TRITON_XDNA_INSTALL_DIR "${NPU_INSTALL_DIR}/triton-xdna")

    ExternalProject_Add(triton-xdna
        PREFIX ${NPU_BUILD_DIR}/triton-xdna-build
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/triton-xdna
        BINARY_DIR ${NPU_BUILD_DIR}/triton-xdna
        DEPENDS mlir-air-install
        CMAKE_ARGS
            ${COMMON_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=${TRITON_XDNA_INSTALL_DIR}
            -DLLVM_DIR=${LLVM_DIR}
            -DMLIR_DIR=${MLIR_DIR}
            -DAIR_DIR=${MLIR_AIR_INSTALL_DIR}/lib/cmake/air
        ${COMMON_EP_ARGS}
        INSTALL_COMMAND ""
        STEP_TARGETS build
    )
    set(TRITON_XDNA_ENABLED TRUE)
else()
    set(TRITON_XDNA_ENABLED FALSE)
endif()

###############################################################################
# IRON (AMD NPU programming examples)
###############################################################################

# Note: IRON is a collection of NPU programming examples, not a build target.
# Clone with: CLONE_IRON=ON python3 setup.py --defaults  (or --with iron)
if(EXISTS "${CMAKE_SOURCE_DIR}/iron/README.md")
    set(IRON_AVAILABLE TRUE)
else()
    set(IRON_AVAILABLE FALSE)
endif()

###############################################################################
# XDNA Driver (AMD NPU kernel driver)
###############################################################################

# Note: xdna-driver is a kernel module and requires different build process
# It's included as a submodule for reference but not built by this CMake
message(STATUS "xdna-driver submodule available at: ${CMAKE_SOURCE_DIR}/xdna-driver")
message(STATUS "Build xdna-driver separately following its documentation")

###############################################################################
# Summary
###############################################################################

message(STATUS "")
message(STATUS "=== NPU Development Tools Build Configuration ===")
message(STATUS "Build directory: ${NPU_BUILD_DIR}")
message(STATUS "Install directory: ${NPU_INSTALL_DIR}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "LLVM/MLIR Dependencies:")
if(USE_MLIR_WHEEL)
    message(STATUS "  - mlir_aie wheel: ${LLVM_INSTALL_DIR}")
else()
    message(STATUS "  - llvm-project: Building from source")
endif()
message(STATUS "  - llvm-aie (Peano) wheel: ${PEANO_INSTALL_DIR}")
message(STATUS "")
message(STATUS "Projects to build:")
message(STATUS "  - mlir-aie (AIE dialect)")
if(MLIR_AIR_ENABLED)
    message(STATUS "  - mlir-air (AIR dialect)")
else()
    message(STATUS "  - mlir-air (SKIPPED - submodule not present)")
endif()
if(TRITON_XDNA_ENABLED)
    message(STATUS "  - triton-xdna (AMD Triton/XDNA compiler)")
else()
    message(STATUS "  - triton-xdna (SKIPPED - submodule not present)")
endif()
if(TORCH_STATUS EQUAL "0")
    message(STATUS "  - torch-mlir (PyTorch frontend)")
else()
    message(STATUS "  - torch-mlir (SKIPPED - PyTorch not found)")
endif()
if(IRON_AVAILABLE)
    message(STATUS "  - iron (NPU programming examples)")
else()
    message(STATUS "  - iron (NOT CLONED - enable with CLONE_IRON=ON or --with iron)")
endif()
message(STATUS "")
message(STATUS "To build: cmake --build build -j$(nproc)")
if(USE_MLIR_WHEEL)
    message(STATUS "To build LLVM from source: cmake -B build -DUSE_MLIR_WHEEL=OFF")
endif()
message(STATUS "==============================================")
message(STATUS "")
