# AIE4 Medusa Branch README

## Getting started from the nightly wheel

The [nightly wheel workflow](https://gitenterprise.xilinx.com/XRLabs/mlir-aie/actions/workflows/buildWheel.yml)
builds `medusa-dev` with Python 3.12 and publishes it to
`/group/xrlabs/jefff/aie4/latest-wheels`. The same workflow installs its newly
built wheel on an NPU3 Medusa runner and runs the `passthrough_kernel`
programming example below.

On a Medusa/NPU3 runner, clone the branch and create a clean Python 3.12
environment:

```bash
git clone -b medusa-dev --single-branch https://gitenterprise.xilinx.com/XRLabs/mlir-aie.git
cd mlir-aie

python3.12 -m venv aie-venv
source aie-venv/bin/activate
python -m pip install --upgrade pip

# Install mlir-aie from the latest nightly wheel and generate medusa_env.sh.
source utils/medusa_setup.sh

# Install the AIE4 Peano toolchain available on the Medusa runner.
python -m pip install /scratch/github/llvm_aie-1.9.0.dev20260611145819+g33c9a5c2-py3-none-manylinux_2_38_x86_64.whl
export PEANO_INSTALL_DIR="$(python -m pip show llvm-aie | awk '/^Location:/{print $2}')/lnx64.o/tools/peano"

source /opt/xilinx/xrt/setup.sh
source medusa_env.sh
```

Build and run a small full-ELF example on the attached NPU3:

```bash
make -C programming_examples/basic/passthrough_kernel \
  devicename=npu3 use_placed=1 run
```

The command builds the placed AIE4 design and its C++ host executable, then
loads the full ELF through XRT. A successful run exits with status zero.
## Building from Source

This branch must be built against a version of Vitis containing the aie4
libraries and simulator, like the daily latest head build:

```bash
# use nightly build of vitis
export VITIS=/proj/primebuilds/HEAD_PRIME_daily_latest/installs/lin64/HEAD/Vitis
export PATH=${VITIS}/aietools/bin:${VITIS}/bin:$PATH
export LM_LICENSE_FILE=1757@xsjlicsrvip:2100@aiengine-eng
```

Checkout the branch:
```base
git clone https://gitenterprise.xilinx.com/XRLabs/mlir-aie.git
cd mlir-aie
git checkout -b medusa-dev origin/medusa-dev
git submodule update --init --recursive
```

Build mlir-aie in the usual ways. Here's one example:

```bash
MLIR_AIE_DIR='<path to mlir-aie>'
MLIR_DIR='<where to put llvm/mlir install>'
BUILD_DIR='<where to build>'
INSTALL_DIR='<where to install>'

# setup python env
python -m venv aie-venv
source aie-venv/bin/activate
pip install --upgrade pip
pip install -r $MLIR_AIE_DIR/python/requirements.txt

# get llvm/mlir
pushd ${MLIR_DIR}
VERSION=$(mlir-aie/utils/clone-llvm.sh --get-wheel-version)
pip -q download mlir==$VERSION \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
unzip -q mlir-*.whl
popd

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# configure (customize as needed)
cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_DIR=${MLIR_DIR}/mlir/lib/cmake/llvm \
    -DMLIR_DIR=${MLIR_DIR}/mlir/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=$MLIR_AIE_DIR/cmake/modulesXilinx \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DCMAKE_ASM_COMPILER=clang-15 \
    -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
    -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
    -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DAIE_RUNTIME_TARGETS:STRING="x86_64" \
    -DAIE_RUNTIME_TEST_TARGET=x86_64 \
    -DLLVM_EXTERNAL_LIT=`which lit` \
    -S$MLIR_AIE_DIR

# build
ninja install
```
