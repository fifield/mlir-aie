# M29 toolchain prerequisites

M29 rebuilds application kernels, graphs, runtime streams, and packed tensors from local source and raw model inputs. It carries no compiler. It builds against the SDK that the enclosing repository builds and installs: the Peano/llvm-aie compiler, MLIR-AIE, the custom llvmlite, PythoC, the Python virtual environment, and XRT. The Python interpreter, system libraries, kernel, and NPU driver are host prerequisites in the same way.

## Using the SDK

Activate the enclosing repository's environment before any M29 command. From the directory that contains `laya_npu/`:

```bash
source ../../../../env.sh
```

## Which SDK

The expected outputs in `tests/expected/` were produced by the SDK recorded in `provenance/`. `python-requirements.txt` lists the Python package versions of that environment. A different SDK is fine as long as `pytest -m build` still passes. If it does not, the SDK changed generated code.

## Provenance

`provenance/` records how the qualifying SDK was built: the compiler source heads (llvm-aie, MLIR-AIE, llvmlite, PythoC), their patches (empty: no local modifications), the observed CMake configuration, the repository's setup scripts, Python versions, and the host package list. Rebuilding the compiler and reproducing its binaries has not been qualified.

The recorded MLIR-AIE head `9c04a2e` is M29's own first checkpoint commit. It was later rewritten as `6bd844c6d30` to drop M29's reference binaries and model checkpoint from history; the compiler sources are identical in both.

The installed custom llvmlite reports revision `69dc57f`; the recorded source head `31a0d28` adds wheel version override support.

The `npu` tests stop if the amdxdna driver logs anything new during a run. No script installs or changes a kernel driver.

## History

M29 at first archived the installed SDK binaries, compiler sources, and a Python wheelhouse (5.6 GB) with a `provision.py` that recreated the environment offline. They were removed on 2026-10-04: the SDK is the enclosing repository's build output, and copying it is no more part of self-containment than copying the operating system.
