# AIE4 Medusa Branch README

## Quick setup from pre-built wheel

The quickest way to get started with medusa-dev branch is to use a prebuilt wheel from `/group/xrlabs/jefff/aie4/latest-wheels`
The wheels are [built nightly](https://gitenterprise.xilinx.com/XRLabs/mlir-aie/actions/workflows/buildWheel.yml) from medusa-dev branch.

The
[`utils/medusa_setup.sh`](https://gitenterprise.xilinx.com/XRLabs/mlir-aie/blob/medusa-dev/utils/medusa_setup.sh)
script can be used to install from the latest nightly wheel, or used as an example:

```bash
# The setup script will pip install mlir-aie and its dependencies, so create and activate a fresh venv
# The wheel is built for python 3.12
python3.12 -m venv venv
source venv/bin/activate

# install mlir-aie from wheel using the example setup script
git clone -b medusa-dev --single-branch https://gitenterprise.xilinx.com/XRLabs/mlir-aie.git
mlir-aie/utils/medusa_setup.sh
```

The `medusa_setup.sh` script will create an environment setup script named
`medusa_env.sh`. This script sets up the environment for mlir-aie (iron), 
mesimulato and vitis. It can be used directly or used as an example.

Source the envrionment setup script and try an example:

```bash
# setup environment
source medusa_env.sh

# build programming example
cd mlir-aie/programming_examples/aie4/eltwise_add

# make test.exe and eltwise_add_placed.elf:
# - test.cpp → test.exe
# - eltwise_add_placed.py → eltwise_add_placed.mlir
# - aiecc.py --generate-full-elf --full-elf-name=eltwise_add_placed.elf eltwise_add_placed.mlir
make
```
The makefile
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
