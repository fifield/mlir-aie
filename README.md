# AIE4 Branch README

## Quick setup from pre-built wheel

The quickest way to get started with aie4-dev branch is to use a prebuilt wheel.
The
[`utils/pacifica_setup.sh`](https://gitenterprise.xilinx.com/XRLabs/mlir-aie/blob/aie4-dev/utils/pacifica_setup.sh)
script can be used to install from the latest nightly wheel, or used as an example:

```bash
# The setup script will pip install mlir-aie and its
# dependencies, so create and activate a fresh venv
python3 -m venv venv
source venv/bin/activate

# install mlir-aie from wheel using the example setup script
git clone -b aie4-dev --single-branch https://gitenterprise.xilinx.com/XRLabs/mlir-aie.git
mlir-aie/utils/pacifica_setup.sh
```

The `pacifica_setup.sh` script will create an environment setup script named
`pacifia_env.sh`. This script sets up the environment for mlir-aie (iron), 
mesimulato and vitis. It can be used directly or used as an example.

Source the envrionment setup script and try an example:

```bash
# setup environment
source pacifica_env.sh

# build programming example
cd mlir-aie/programming_examples/aie4/ddr_read_bw
make

# make will build the project in the 'project' directory
# run the simulation:
project/mesim.sh
```
The simulator will print a lot of information, including printfs from the
application's host code (`test.cpp`). The output for the above example should
include something like this:

```
Achieved bandwidth for column 0 channel 0: 63.719982 GB/s, 99.56% efficient
...
PASS!
```
The aiecc generated mesimulator wrapper supports passing args to the user host
code. Try the example again with more columns and channels:
```bash
# run the ddr_read_bw example with 3 columns, 4 channels per column
project/mesim.sh -c 3 -n 4
```
Output:
```
...
num_cols: 3, num_channels: 4
...
Perf counter 0 for column 0: 23159
Perf counter 1 for column 0: 23839
Perf counter 2 for column 0: 23846
Perf counter 3 for column 0: 23853
Transferred 262144 bytes per channel for column 0
Achieved bandwidth for column 0 channel 0: 11.319314 GB/s, 17.69% efficient
Achieved bandwidth for column 0 channel 1: 10.996434 GB/s, 17.18% efficient
Achieved bandwidth for column 0 channel 2: 10.993206 GB/s, 17.18% efficient
Achieved bandwidth for column 0 channel 3: 10.989981 GB/s, 17.17% efficient
Perf counter 0 for column 1: 23875
Perf counter 1 for column 1: 24174
Perf counter 2 for column 1: 24181
Perf counter 3 for column 1: 24188
Transferred 262144 bytes per channel for column 1
Achieved bandwidth for column 1 channel 0: 10.979854 GB/s, 17.16% efficient
Achieved bandwidth for column 1 channel 1: 10.844048 GB/s, 16.94% efficient
Achieved bandwidth for column 1 channel 2: 10.840908 GB/s, 16.94% efficient
Achieved bandwidth for column 1 channel 3: 10.837770 GB/s, 16.93% efficient
Perf counter 0 for column 2: 24203
Perf counter 1 for column 2: 24254
Perf counter 2 for column 2: 24261
Perf counter 3 for column 2: 24260
Transferred 262144 bytes per channel for column 2
Achieved bandwidth for column 2 channel 0: 10.831054 GB/s, 16.92% efficient
Achieved bandwidth for column 2 channel 1: 10.808279 GB/s, 16.89% efficient
Achieved bandwidth for column 2 channel 2: 10.805161 GB/s, 16.88% efficient
...
```
## Building from Source

This branch must be built against a version of Vitis containing the aie4
libraries and simulator, like the daily latest head build:

```bash
# use nightly build of vitis
export VITIS=/proj/primebuilds/HEAD_INT_daily_latest/installs/lin64/HEAD/Vitis
export PATH=${VITIS}/aietools/bin:${VITIS}/bin:$PATH
export LM_LICENSE_FILE=1757@xsjlicsrvip:2100@aiengine-eng
```

Checkout the branch:
```base
git clone https://gitenterprise.xilinx.com/XRLabs/mlir-aie.git
cd mlir-aie
git checkout -b aie4-dev origin/aie4-dev
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
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie pip install -r $MLIR_AIE_DIR/python/requirements_extras.txt
pip install -r $MLIR_AIE_DIR/python/requirements_ml.txt

# get llvm/mlir
pushd ${MLIR_DIR}
VERSION=$(mlir-aie/utils/clone-llvm.sh --get-wheel-version)
pip -q download mlir==$VERSION \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
unzip -q mlir-*.whl
popd

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# configure
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
The aie4 tests can be executed using the llvm lit tool. There are some aie4 unit
tests in `test/unit_tests/aie4` and `test/unit_tests/chess_compiler_tests_aie4`:

```bash
# [optional] setup environment for pacifica mesimulator, otherwise tests run on vitis aiesimulator only
export PACIFICA_SIM_ROOT=<path to pacifica-sim>
export CERVINO_ROOT=$PACIFICA_SIM_ROOT/cervino
export ME_ROOT=$PACIFICA_SIM_ROOT/cervino/aie4
export me_DIR=$ME_ROOT/core_model/sipp/lib
export LD_LIBRARY_PATH=$CERVINO_ROOT/ext/systemc/systemc-2.3.3/lib-linux64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CERVINO_ROOT/ext/Boost_1_72_0/boost_1_72_0/rdi/lib/gcc8.3.0/lnx64.o:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$ME_ROOT/python:$ME_ROOT/applications

# run tests
cd $BUILD_DIR
lit -sv test/unit_tests/aie4 test/unit_tests/chess_compiler_tests_aie4
```

## Building and running aie4 programs in simulation

The current aie4 examples (host code) interact with the simulation with a simple
API which can be used to read and write address in the AIE array and to
configure the array with transaction binaries generated by mlir-aie.

```c++
// Write 'Data' to 'Addr'
void mlir_aie_sim_write32(uint64_t Addr, uint Data);
// Read and return 32-bits from 'Addr'
uint mlir_aie_sim_read32(uint64_t Addr);
// Write the transaction binary from location 'data' with size 'nbytes'
void test_utils::write_transaction_binary(unsigned char *data, size_t nbytes);

// Example: Read the value of lock #1 from compute tile 'col', 'row'
uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
uint32_t lock1 = mlir_aie_sim_read32(tile_lock_1);

// Example: Configure the AIE array using a transation binary
std::string txn_path = "transaction.bin";
std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                     data.size() * sizeof(uint32_t));
```

Transaction binaries are not generated by aiecc by default but can be manually
generated. See the aie4 tests for examples.

```bash
# Compile C++ kernel
xchesscc_wrapper aie4 -c kernel.cc
# Build mlir-aie design
# --aiesim to generate simulation binaries including "host code"
aiecc.py --aiesim --xchesscc --xbridge --no-compile-host --aie-generate-txn --txn-name=transaction.mlir aie.mlir test.cpp
# Translate transaction mlir to binary
aie-translate --aie-npu-to-binary transaction.mlir -o transaction.bin
# Run the simulation (vitis aiesimulator)
aie.mlir.prj/aiesim.sh
# Run the simulation (pacifica mesimulator)
aie.mlir.prj/mesim.sh
```

The simulation will produce a vcd trace called `foo.vcd` which can be viewed with gtkwave.
