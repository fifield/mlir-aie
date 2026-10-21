# AIE4 Medusa Development Branch

`medusa-dev` develops MLIR-AIE support for AIE4 on Medusa/NPU3, including
placed designs, full-ELF compilation and execution through XRT, AIE4 compute
kernels, DMA, and event tracing. The wheel quick start below runs a small
hardware example; the source build supports compiler development and testing.

## Prerequisites

These instructions target an internal Linux x86-64 Medusa runner with:

- Python 3.12 and virtual environment support.
- Access to the internal Git server, wheel share, Peano Artifactory, and Vitis
  installation and license servers.
- An NPU3 device with a compatible driver and XRT installed at
  `/opt/xilinx/xrt`, including development headers and libraries.
- Git and Git LFS, GNU Make, CMake 3.30 or newer, and GCC/G++ 13 for the example
  host executable. Source builds also need Ninja, Clang, LLD, and `unzip`.

The AIE4 Peano wheel in `utils/aie4-peano-requirements.txt` requires glibc 2.38
or newer. Use this internal AIE4-capable toolchain for these instructions.

## Getting started from the nightly wheel

The [nightly wheel workflow](https://gitenterprise.xilinx.com/XRLabs/mlir-aie/actions/workflows/buildWheel.yml),
maintained on `main`, builds `medusa-dev` with Python 3.12. It installs its
newly built wheel with matching sources on an NPU3 Medusa runner and runs the
`passthrough_kernel` example below. After the smoke test passes, it publishes
the wheel to `/group/xrlabs/jefff/aie4/latest-wheels`.

Clone the branch and create a clean Python 3.12 environment:

```bash
git clone -b medusa-dev --single-branch https://gitenterprise.xilinx.com/XRLabs/mlir-aie.git
cd mlir-aie
git submodule update --init --recursive
git lfs install --local
git lfs pull
git submodule foreach --recursive git lfs pull

python3.12 -m venv aie-venv
source aie-venv/bin/activate
python -m pip install --upgrade pip

# Install mlir-aie from the latest nightly wheel and generate medusa_env.sh.
source utils/medusa_setup.sh

# Install a CMake that meets the example's minimum.
python -m pip install 'cmake>=3.30'

# Install the pinned AIE4 Peano toolchain.
python -m pip install -r utils/aie4-peano-requirements.txt
export PEANO_INSTALL_DIR="$(python -m pip show llvm-aie | awk '/^Location:/{print $2}')/lnx64.o/tools/peano"

source /opt/xilinx/xrt/setup.sh
source medusa_env.sh
xrt-smi examine
```

`xrt-smi examine` should show the attached NPU3. `medusa_env.sh` configures
MLIR-AIE, the simulator, Vitis, and licenses using internal runner paths.
Set `VITIS`, `MEDUSA_SIM_ROOT`, `CERVINO_ROOT`, `ME_ROOT`,
`MEDUSA_SYSTEMC_LIB_DIR`, or `MEDUSA_BOOST_LIB_DIR` before sourcing the generated
file to use another installation. Existing license settings are preserved.
The default Vitis build matches Medusa CI. Simulator search paths are added
only when their directories exist; hardware execution can use this environment
without a simulator installation. The setup script installs `toml`, `bitstring`,
and `cffi` even when reusing an existing MLIR-AIE installation.

Build and run a small full-ELF example on the attached NPU3:

```bash
make -C programming_examples/basic/passthrough_kernel \
  devicename=npu3 use_placed=1 run
```

The command builds the placed AIE4 design and its C++ host executable, then
loads the full ELF through XRT. A successful run exits with status zero.
`use_placed=1` selects the explicitly placed design required by this example's
NPU3 flow.

The smoke test on `main` must stay aligned with this quick start whenever its
commands change. It installs the wheel built by that workflow directly before
running setup, so it tests that exact artifact instead of the latest published
wheel. For an existing environment, `source utils/medusa_setup.sh --force-install`
refreshes the installed MLIR-AIE wheel.

## Building from source

Start from a checkout with submodules and Git LFS objects initialized as above.
Use a separate Python environment for the source build:

```bash
MLIR_AIE_SOURCE_DIR="$PWD"
LLVM_WHEEL_DIR="$MLIR_AIE_SOURCE_DIR/llvm-wheel"
AIE_BUILD_DIR="$MLIR_AIE_SOURCE_DIR/build"
AIE_INSTALL_DIR="$MLIR_AIE_SOURCE_DIR/install"

python3.12 -m venv aie-dev-venv
source aie-dev-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
python -m pip install -r python/requirements_dev.txt
python -m pip install -r python/requirements_ml.txt
python -m pip install -r python/requirements_notebook.txt
python -m pip install -r utils/aie4-peano-requirements.txt
export PEANO_INSTALL_DIR="$(python -m pip show llvm-aie | awk '/^Location:/{print $2}')/lnx64.o/tools/peano"

# AIE4-capable Vitis build used by the medusa-dev CI workflow.
export VITIS=/proj/primebuilds/2026.2_PRIME_0601_1/installs/lin64/HEAD/Vitis
export PATH="${VITIS}/aietools/bin:${VITIS}/bin:$PATH"
export LM_LICENSE_FILE=1757@xsjlicsrvip:2100@aiengine-eng
source /opt/xilinx/xrt/setup.sh

# Download the LLVM/MLIR version pinned by this checkout.
mkdir -p "$LLVM_WHEEL_DIR"
pushd "$LLVM_WHEEL_DIR"
VERSION=$("$MLIR_AIE_SOURCE_DIR/utils/clone-llvm.sh" --get-wheel-version)
python -m pip download --no-deps "mlir==$VERSION" \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
unzip -q -o mlir-*.whl
popd

cmake -G Ninja -S "$MLIR_AIE_SOURCE_DIR" -B "$AIE_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DLLVM_DIR="$LLVM_WHEEL_DIR/mlir/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_WHEEL_DIR/mlir/lib/cmake/mlir" \
  -DCMAKE_MODULE_PATH="$MLIR_AIE_SOURCE_DIR/cmake/modulesXilinx" \
  -DCMAKE_INSTALL_PREFIX="$AIE_INSTALL_DIR" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DAIE_ENABLE_BINDINGS_PYTHON=ON \
  -DAIE_ENABLE_PYTHON_PASSES=OFF \
  -DAIE_ENABLE_XRT_PYTHON_BINDINGS=ON \
  -DAIE_INCLUDE_INTEGRATION_TESTS=OFF \
  -DAIE_RUNTIME_TARGETS=x86_64 \
  -DAIE_RUNTIME_TEST_TARGET=x86_64 \
  -DXRT_ROOT=/opt/xilinx/xrt \
  -DLLVM_EXTERNAL_LIT="$(command -v lit)"

cmake --build "$AIE_BUILD_DIR" --target install --parallel 4

# Pass both paths explicitly for the internal Peano wheel layout.
source "$MLIR_AIE_SOURCE_DIR/utils/env_setup.sh" "$AIE_INSTALL_DIR" "$PEANO_INSTALL_DIR"

make -C "$MLIR_AIE_SOURCE_DIR/programming_examples/basic/passthrough_kernel" \
  devicename=npu3 use_placed=1 run
```

Use a fresh LLVM wheel directory when changing the pinned LLVM version so the
wheel glob selects a single download. Set `CMAKE_BUILD_TYPE=Debug` if needed
for compiler debugging, and adjust build parallelism to the machine's memory.
When switching between wheel and source installations, clean the example's
build outputs before rebuilding (`make -C programming_examples/basic/passthrough_kernel clean`).

The [Medusa CI workflow](.github/workflows/buildAndTestMedusaDev.yml) is the
reference for the current NPU3 build and test configuration. To run the same
suite targets from the source build on an NPU3 runner:

```bash
export AIECC_JOBS=1
export AIE_KERNEL_COMPILE_JOBS=1
export LIT_OPTS='-sv --time-tests -j8 --timeout 600 --show-unsupported --show-pass --order=random'
cmake --build "$AIE_BUILD_DIR" --target check-aie-cpp check-aie check-reference-designs
```

## Examples and documentation

- [AIE4 compute examples](programming_examples/aie4/README_COMPUTE.md): vector
  arithmetic and matrix multiplication across integer, floating-point, and MX
  formats, with hardware and simulator flows.
- [Full-array GEMM](programming_examples/aie4/gemm/README.md): integer and
  floating-point GEMM, configuration options, and verification.
- [AIE4 event tracing](programming_examples/aie4/event_trace) and
  [simulator tracing](programming_examples/aie4/event_trace/sim/README.md).
- [AIE4 kernels](aie_kernels/aie4/README.md) and
  [Peano intrinsic checks](utils/aie4/README.md).
- [Compiler guide](tools/aiecc/README.md): `aiecc` options, including
  `--get-full-elf` and artifact naming.
- [Programming guide](programming_guide/README.md) and
  [MLIR-AIE documentation](https://xilinx.github.io/mlir-aie/): general concepts
  and APIs. Follow the Medusa-specific setup and per-example target requirements
  when using these resources on NPU3.
