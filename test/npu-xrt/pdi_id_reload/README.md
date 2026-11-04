<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# PDI ids across separately compiled ELFs on NPU3

Two AIE4 full ELFs, compiled separately, share one hardware context and are
dispatched alternately. Each dispatch must run its own configuration. This is
an end-to-end hardware regression test for deriving cert `LOAD_PDI` ids from the
configuration's content (`aie-cert-assign-pdi-ids`).

## The firmware skip rule

The cert firmware skips a `LOAD_PDI` whose id equals the id of the last PDI it
loaded on that uC in the hardware context. It assumes equal ids mean identical PDIs, even
across ELFs. `aie-npu-to-cert` used to number the sections `1, 2, ...` per
module, so two separately compiled ELFs both used id 1. The second ELF's
configuration was skipped, and its dispatch ran on the first ELF's design.

## What the test checks

`pdi_id_reload_placed.py` generates one design per ELF: shim -> core(0,2) ->
shim, whose core loops forever computing `out = in + K` over 64 x i32. The
entry device runs `aiex.configure @<cfg> { aiex.run @sequence }`, so its
program starts with the `LOAD_PDI` of section `<cfg>`.

| ELF | entry (kernel) | section | K |
| --- | --- | --- | --- |
| `a.elf` | `main_a:seq` | `cfga` | 1 |
| `b.elf` | `main_b:seq` | `cfgb` | 2 |
| `a_copy.elf` | `main_a_copy:seq` | `cfga` | 1 |

`a.elf` and `a_copy.elf` have the same content under different entry names,
so they get the same id. `b.elf` gets a different one.
`make` and `make ids` check that each ELF has exactly one `LOAD_PDI`, that
`a` and `a_copy` share an id, that `b` differs, and that no id is reserved.
These checks establish which firmware cache paths the hardware test exercises.

`test.cpp` creates the context from the first ELF, `add_config()`s the others,
and dispatches `<entry>:seq` in the order given by `--order` (a = first ELF,
b = second, ...). Each step gets fresh inputs and an output filled with `0xEE`,
and every element is checked exactly against `in + K`. Every wait is bounded
(`--timeout-ms`, default 10000), and setup or dispatch errors print `FAIL`.
The entry device is a pure dispatcher: its program is the `LOAD_PDI` of the
section followed by the sequence. The lit test runs:

| Command | Shows |
| --- | --- |
| `--order abab a.elf:main_a:1 b.elf:main_b:2` | different ids: B gets its own configuration |
| `--order abab b.elf:main_b:2 a.elf:main_a:1` | the same, in the other order |
| `--order abab a.elf:main_a:1 a_copy.elf:main_a_copy:1` | same id: B's load may be skipped; results still exact |
| `--order aaaa a.elf:main_a:1` | repeating one ELF |

Equal ids across ELFs also need equal content. A section's configuration
disables the stream switch connections of the other devices of the same
type in its module, so two ELFs share an id only when those sibling devices
match too (here, both entry devices are a shim tile with the same
task-complete-token routes).

## Build and run

Run from `test/npu-xrt/pdi_id_reload` with the mlir-aie environment enabled.

```bash
make                  # builds a.elf, b.elf, a_copy.elf and test.exe
make ids              # checks and prints each ELF's LOAD_PDI id
make run              # all four cases
./test.exe --order abab a.elf:main_a:1 b.elf:main_b:2
```

Expected output of the first case:

```
step 0 (main_a, k=1): ok
step 1 (main_b, k=2): ok
step 2 (main_a, k=1): ok
step 3 (main_b, k=2): ok
4/4 runs exact
PASS
```
