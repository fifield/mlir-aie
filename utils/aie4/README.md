<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# AIE4 intrinsic drift checking

These tools check AIE4 intrinsic signatures and memory effects against the
**installed Peano toolchain**. The checked-in snapshot is a baseline from
llvm-aie TableGen sources; it is regenerated manually.

- `aie4_intrinsics_snapshot.yaml` records intrinsic names, return and parameter
  types, properties, and source locations.
- `check_aie4_intrinsics.py` scans `include/` and `lib/` for referenced names,
  synthesizes LLVM IR from their snapshot signatures, and checks it with
  Peano's `opt -S -mtriple=aie4`.
- `gen_aie4_intrinsics_snapshot.py` generates the snapshot from a local
  llvm-aie checkout using `llvm-tblgen`.

## Run the check

From the mlir-aie repository root, with Peano installed:

```sh
python3 utils/aie4/check_aie4_intrinsics.py \
    --peano-install-dir "$PEANO_INSTALL_DIR" \
    --repo-root . --build-dir /path/to/mlir-aie-build
```

The check fails for missing snapshot entries, unknown intrinsics, rejected
signatures, memory-effect mismatches, or an empty source scan. It prints the
installed wheel version, Clang revision, and Peano pin; version differences
produce warnings only.

Options:

- `--build-dir DIR` compares scanned XLLVM names with generated
  `include/aie/Dialect/XLLVM/IR/XLLVMConversions.inc`. This cross-check is
  skipped if the option or generated file is absent.
- `--all` checks every snapshot record instead of only referenced names.
- `--snapshot PATH` selects another snapshot.
- `--peano-install-dir DIR` overrides `$PEANO_INSTALL_DIR`.

The checker uses PyYAML when available and otherwise parses the generated
snapshot format with a built-in fallback.

## Tests and coverage

`test/aie4-intrinsics/` runs in the existing lit suite when the `peano` feature
is available; it needs no separate CI workflow.

- `drift.test` checks referenced names, requires the generated-name cross-check
  to pass, and checks all 527 snapshot records.
- `xllvm_verify.test` translates four AIE4 XLLVM test inputs and verifies the
  emitted LLVM IR with Peano. It requires translated calls for every generated
  XLLVM intrinsic name and checks that omitting an input fails coverage.
- Snapshot mutation tests exercise changed parameter types, changed memory
  properties, missing entries, unknown intrinsics, and empty source scans.
  Their snapshot helper requires PyYAML.
- `mutation_xllvm_result_type.test` changes the SRS result to i32 in translated LLVM
  IR and requires Peano to reject the intrinsic's return type.

The snapshot checker validates types from the snapshot, so actual XLLVM types
are covered only by the translation tests. Hand-written C++ intrinsic
signatures are not verified. The source scan also skips partial names built
by string concatenation.

Memory checks compare the printed `memory(...)` attribute with the snapshot's
properties. They do not check XLLVM `Pure` traits, other intrinsic attributes,
or side-effect changes that leave `memory(...)` unchanged. The separate
superproject script `utils/check_aie4_intrinsic_drift.py` compares the snapshot
with llvm-aie sources.

## Regenerate the snapshot

With PyYAML, a local llvm-aie checkout, and a built `llvm-tblgen`, run from the
mlir-aie repository root:

```sh
python3 utils/aie4/gen_aie4_intrinsics_snapshot.py \
    --llvm-tblgen /path/to/install/mlir/bin/llvm-tblgen \
    --llvm-aie-include /path/to/llvm-aie/llvm/include \
    --output utils/aie4/aie4_intrinsics_snapshot.yaml
```

Input paths default to the enclosing superproject's layout. The generator
replaces the entire file, including its provenance header. Restore a header
with the source llvm-aie commit and date, noting that the Peano wheel pin is
independent. Review the diff, run the checker with `--all`, and rerun the lit
tests before committing. Neither lit nor CI regenerates the snapshot.
