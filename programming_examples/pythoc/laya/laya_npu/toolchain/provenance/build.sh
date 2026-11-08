#!/bin/bash
###############################################################################
# NPU Development Tools - Build Script
#
# This script provides a simplified interface to build the NPU tools.
# It activates the venv and runs cmake with sensible defaults.
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source environment helper (creates venv, activates it, sources XRT, etc.)
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    echo "Sourcing environment from ${SCRIPT_DIR}/env.sh..."
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
else
    echo "ERROR: env.sh not found. Please run 'python3 setup.py' first."
    exit 1
fi

# Optional: source XRT environment if available
if [ -z "${XILINX_XRT:-}" ] && [ -f /opt/xilinx/xrt/setup.sh ]; then
    echo "Sourcing XRT environment at /opt/xilinx/xrt/setup.sh for build/runtime..."
    # shellcheck disable=SC1091
    source /opt/xilinx/xrt/setup.sh
elif [ -n "${XILINX_XRT:-}" ]; then
    echo "Using existing XRT environment: XILINX_XRT=$XILINX_XRT"
else
    echo "XRT environment not found (optional). Install XRT and run 'source /opt/xilinx/xrt/setup.sh' if needed."
fi

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
JOBS=$(nproc)

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug       Build in Debug mode (default: Release)"
            echo "  --clean       Clean build directory before building"
            echo "  -j N          Use N parallel jobs (default: nproc)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build build/llvm-aie
fi

# === Build llvm-aie from source (must come before cmake — it provides Peano) ===
if [ ! -f install/llvm-aie/bin/llvm-config ]; then
    echo ""
    echo "=== Building llvm-aie ==="
    echo ""
    LLVM_AIE_OPTS="--jobs $JOBS"
    if [ "$BUILD_TYPE" = "Debug" ]; then
        LLVM_AIE_OPTS="$LLVM_AIE_OPTS --debug"
    fi
    ./scripts/build-llvm-aie.sh $LLVM_AIE_OPTS
else
    echo "llvm-aie already built, skipping (use --clean to rebuild)"
fi

# Configure
echo ""
echo "=== Configuring Build ==="
echo "Build type: $BUILD_TYPE"
echo "Parallel jobs: $JOBS"
echo ""

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX=install

# Build
echo ""
echo "=== Building Projects ==="
echo ""

cmake --build build -j$JOBS

# === Build llvmlite against llvm-aie ===
echo ""
echo "=== Building llvmlite ==="
echo ""
./scripts/build-llvmlite-aie.sh

# === Install PythoC ===
echo ""
echo "=== Installing PythoC ==="
echo ""
# --no-deps is critical: PythoC's only runtime dependency is "llvmlite>=0.40.0",
# but the llvmlite we just built (build-llvmlite-aie.sh) is our peano/llvm-aie
# fork. From a fresh checkout it has no git tags, so versioneer reports its
# version as "0+untagged...", which does NOT satisfy ">=0.40.0" -- without
# --no-deps, pip "fixes" that by downloading stock llvmlite from PyPI and
# uninstalling our fork, leaving an llvmlite with no AIE target (aie2p) and no
# ir.Poison. llvmlite MUST remain the peano-built fork.
pip install --no-build-isolation --no-deps -e PythoC

echo ""
echo "=== Build Complete ==="
echo ""
echo "Installation directory: $SCRIPT_DIR/install"
echo ""
echo "To use the tools, add them to your PATH:"
echo "  export PATH=\$PATH:$SCRIPT_DIR/install/llvm/bin"
echo "  export PATH=\$PATH:$SCRIPT_DIR/install/llvm-aie/bin"
echo "  export PATH=\$PATH:$SCRIPT_DIR/install/mlir-aie/bin"
echo "  export PATH=\$PATH:$SCRIPT_DIR/install/mlir-air/bin"
echo ""
