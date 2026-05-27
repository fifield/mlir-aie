#!/bin/bash
##===- build-wheels-rhel8.sh - Build MLIR/mlir-aie wheels for RHEL8 --*- Script -*-===##
#
# Builds MLIR and mlir-aie Python wheels inside a CentOS 8 container.
# Expects mlir-aie source to be mounted at /workspace/mlir-aie (read-only).
#
# Usage (inside container):
#   /workspace/build-wheels-rhel8.sh [--skip-mlir-wheel]
#
# Environment variables:
#   PARALLEL_LEVEL  - ninja parallelism (default: nproc)
#   MLIR_AIE_SRC    - path to mounted mlir-aie source (default: /workspace/mlir-aie)
#
##===----------------------------------------------------------------------===##

set -euo pipefail

ulimit -n 65536

BUILD_MLIR_WHEEL=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-mlir-wheel) BUILD_MLIR_WHEEL=false; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

NPROC=${PARALLEL_LEVEL:-$(nproc)}
WHEELHOUSE=/workspace/wheelhouse
WORK=/workspace/work
mkdir -p "$WHEELHOUSE" "$WORK"

PYTHON=python3.11

###############################################################################
# Step 0: Create a venv for the build
###############################################################################
echo "=== Creating build venv ==="
$PYTHON -m venv /workspace/build-venv
source /workspace/build-venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install cmake ninja numpy "pybind11[global]>=2.10.4" nanobind>=2.5
pip install auditwheel importlib-metadata rich dataclasses pylint pyyaml markdown jinja2

###############################################################################
# Step 1: Copy mlir-aie source (mounted read-only) to writable work dir
###############################################################################
MLIR_AIE_SRC=${MLIR_AIE_SRC:-/workspace/mlir-aie}
echo "=== Copying mlir-aie source from $MLIR_AIE_SRC ==="
cp -a "$MLIR_AIE_SRC" "$WORK/mlir-aie"
MLIR_AIE_DIR="$WORK/mlir-aie"
cd "$MLIR_AIE_DIR"
AIE_PROJECT_COMMIT=$(git rev-parse --short=8 HEAD)

# Patch aiebu: skip Debug build, skip tests, inject -lstdc++fs
# (GCC 8 needs -lstdc++fs for std::filesystem, must come after objects)
AIEBU_BUILD_SH="$MLIR_AIE_DIR/third_party/aiebu/build/build.sh"
sed -i 's/^compile Debug/#compile Debug/' "$AIEBU_BUILD_SH"
sed -i 's/^run_test="yes"/run_test="no"/' "$AIEBU_BUILD_SH"
sed -i 's|cmake -B $BUILDDIR/$config $cmakeflags $BUILDDIR/..|cmake -B $BUILDDIR/$config $cmakeflags -DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs $BUILDDIR/..|' "$AIEBU_BUILD_SH"
echo "Patched aiebu build.sh for GCC 8 compatibility"

# Get LLVM version info from clone-llvm.sh
WHEEL_VERSION=$("$MLIR_AIE_DIR/utils/clone-llvm.sh" --get-wheel-version)
source <(grep -E '^(LLVM_PROJECT_COMMIT|DATETIME)=' "$MLIR_AIE_DIR/utils/clone-llvm.sh")
echo "LLVM commit: $LLVM_PROJECT_COMMIT"
echo "MLIR wheel version: $WHEEL_VERSION"

###############################################################################
# Step 2: Build MLIR wheel (optional)
###############################################################################
if [ "$BUILD_MLIR_WHEEL" = true ]; then
  echo "=== Building MLIR wheel ==="

  cd "$WORK"
  MLIR_WHEEL_DIR="$MLIR_AIE_DIR/utils/mlir_wheels"

  curl -sL "https://codeload.github.com/llvm/llvm-project/zip/${LLVM_PROJECT_COMMIT}" -o llvm.zip
  unzip -q llvm.zip
  rm llvm.zip
  mv "llvm-project-${LLVM_PROJECT_COMMIT}" "$MLIR_WHEEL_DIR/llvm-project"

  if [ -d "$MLIR_WHEEL_DIR/patches" ]; then
    cd "$MLIR_WHEEL_DIR"
    APPLY_PATCHES=true bash scripts/apply_patches.sh || true
    cd "$WORK"
  fi

  cd "$MLIR_WHEEL_DIR"
  CIBW_ARCHS=x86_64 \
  CMAKE_GENERATOR=Ninja \
  DATETIME="$DATETIME" \
  LLVM_PROJECT_COMMIT="${LLVM_PROJECT_COMMIT:0:8}" \
  ENABLE_RTTI=ON \
  PARALLEL_LEVEL="$NPROC" \
  pip wheel . -v -w "$WHEELHOUSE" --no-build-isolation

  rm -rf "$MLIR_WHEEL_DIR/llvm-project" "$MLIR_WHEEL_DIR/build"

  cd "$WHEELHOUSE"
  for f in mlir-*.whl; do
    if [[ "$f" == *"cp311-cp311"* ]]; then
      newname=$(echo "$f" | sed 's/cp311-cp311/py3-none/')
      mv "$f" "$newname"
    fi
  done

  echo "=== MLIR wheel built ==="
  ls -lh "$WHEELHOUSE"/mlir*.whl

  MLIR_INSTALL_DIR="$WORK/mlir-install"
  mkdir -p "$MLIR_INSTALL_DIR"
  cd "$MLIR_INSTALL_DIR"
  unzip -o -q "$WHEELHOUSE"/mlir-*.whl
else
  echo "=== Skipping MLIR wheel build ==="

  MLIR_INSTALL_DIR="$WORK/mlir-install"
  mkdir -p "$MLIR_INSTALL_DIR"
  cd "$MLIR_INSTALL_DIR"
  if ls "$WHEELHOUSE"/mlir-*.whl 1>/dev/null 2>&1; then
    echo "Using MLIR wheel from wheelhouse"
    unzip -o -q "$WHEELHOUSE"/mlir-*.whl
  else
    echo "Downloading MLIR wheel"
    pip -q download "mlir==$WHEEL_VERSION" \
      -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
    unzip -o -q mlir-*.whl
  fi
fi

###############################################################################
# Step 3: Build mlir-aie wheel
###############################################################################
echo "=== Building mlir-aie wheel ==="

pip install -r "$MLIR_AIE_DIR/python/requirements.txt" || true
pip install -r "$MLIR_AIE_DIR/utils/mlir_aie_wheels/requirements.txt"

# CentOS 8 Boost 1.66 lacks BoostConfig.cmake needed by cmake 4.x
BOOST_CMAKE_DIR="/usr/lib64/cmake/Boost-1.66.0"
mkdir -p "$BOOST_CMAKE_DIR"
cat > "$BOOST_CMAKE_DIR/BoostConfig.cmake" << 'BOOSTCFG'
set(Boost_FOUND TRUE)
set(Boost_VERSION_STRING "1.66.0")
set(Boost_VERSION "1.66.0")
set(Boost_VERSION_MAJOR 1)
set(Boost_VERSION_MINOR 66)
set(Boost_VERSION_PATCH 0)
set(Boost_INCLUDE_DIRS "/usr/include")
set(Boost_LIBRARIES "")
BOOSTCFG
cat > "$BOOST_CMAKE_DIR/BoostConfigVersion.cmake" << 'BOOSTVER'
set(PACKAGE_VERSION "1.66.0")
if("${PACKAGE_FIND_VERSION}" VERSION_GREATER "${PACKAGE_VERSION}")
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if("${PACKAGE_FIND_VERSION}" VERSION_EQUAL "${PACKAGE_VERSION}")
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
BOOSTVER

pip install "mlir-native-tools==$WHEEL_VERSION" --force -U || true

MLIR_AIE_WHEEL_DIR="$MLIR_AIE_DIR/utils/mlir_aie_wheels"
cd "$MLIR_AIE_WHEEL_DIR"

ln -sfn "$MLIR_AIE_DIR" "$MLIR_AIE_WHEEL_DIR/mlir-aie"
cp "$MLIR_AIE_DIR/utils/clone-llvm.sh" "$MLIR_AIE_WHEEL_DIR/"

NOW_DATETIME=$(date +"%Y%m%d%H")

CIBW_ARCHS=x86_64 \
CMAKE_GENERATOR=Ninja \
CMAKE_ARGS="-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs" \
DATETIME="$NOW_DATETIME" \
AIE_PROJECT_COMMIT="$AIE_PROJECT_COMMIT" \
ENABLE_RTTI=ON \
PARALLEL_LEVEL="$NPROC" \
MLIR_AIE_SOURCE_DIR="$MLIR_AIE_DIR" \
MLIR_INSTALL_ABS_PATH="$MLIR_INSTALL_DIR/mlir" \
pip wheel . -v -w "$WHEELHOUSE" --no-build-isolation

cd "$WHEELHOUSE"
for f in mlir_aie-*.whl; do
  if [[ "$f" == *"cp311-cp311"* ]]; then
    newname=$(echo "$f" | sed 's/cp311-cp311/py3-none/')
    mv "$f" "$newname"
  fi
done

echo "=== mlir-aie wheel built ==="
ls -lh "$WHEELHOUSE"/mlir_aie*.whl

###############################################################################
# Step 4: Repair wheels with auditwheel
###############################################################################
echo "=== Repairing wheels ==="
cd "$WHEELHOUSE"

REPAIRED_DIR="$WHEELHOUSE/repaired"
mkdir -p "$REPAIRED_DIR"

for whl in mlir-*.whl mlir_aie-*.whl; do
  [ -f "$whl" ] || continue
  echo "Repairing $whl"
  auditwheel repair "$whl" -w "$REPAIRED_DIR" \
    --exclude libmlir_async_runtime.so \
    --exclude libmlir_c_runner_utils.so \
    --exclude libmlir_float16_utils.so \
    --exclude libmlir_runner_utils.so \
    --exclude libmlir_apfloat_wrappers.so \
    || {
      echo "auditwheel repair failed for $whl, copying as-is"
      cp "$whl" "$REPAIRED_DIR/"
    }
done

mv "$REPAIRED_DIR"/*.whl "$WHEELHOUSE/"
rm -rf "$REPAIRED_DIR"

echo ""
echo "=== Build complete ==="
echo "Wheels in $WHEELHOUSE:"
ls -lh "$WHEELHOUSE"/*.whl
