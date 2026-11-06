<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Back-to-back dispatch on NPU3

A memtile objectFifo design (`out = in + 1` over 1024 x i32), dispatched many
times with no gap between dispatches. Every run must return exactly its own
output. This is a hardware regression test for delivering an AIE4 full ELF's
configuration as a skippable `LOAD_PDI` to prevent displaced output when
dispatches reuse live memtile channels.

## Reusing live memtile channels

ObjectFifo memtile BD chains loop forever, so when a dispatch ends, memtile
channels are still live. If the firmware does not reset the array between
dispatches, configuring the next dispatch rewrites locks and BDs and queues
more work on those live channels, and later runs return their data displaced.

The entry's configuration is a cert section, and the program's first job is
`LOAD_PDI <id>, @main`. cert skips a `LOAD_PDI`
whose id equals the one it loaded last, so a repeated dispatch does not touch
the configuration: the cores keep looping, the memtile chains keep running,
and the sequence only pushes the shim tasks of its run. This relies on the
design ending each dispatch in its post-configuration state (cores that loop
forever, balanced DMA chains).

Whenever a configuration does run (the first dispatch, or a load of another
configuration), it starts with a reset of the whole partition: every core is
held in reset, every core-tile and memtile DMA channel is reset, the memtile
pad values and shim data channels return to their defaults, and then the
configuration is written. In a section it is at the top of the section; under
`--expand-load-pdis` an `aiex.npu.write_reset` writes it in line, before the
configuration.

## What the test checks

`back_to_back_dispatch_placed.py` is the placed `vector_scalar_add` dataflow:
shim -> memtile (64-element slots) -> core (32-element buffers) -> memtile ->
shim, all depth-2 objectFifos. Its options change the device name (`--name`),
the added constant (`--k`), and, with `--output-first`, make the core acquire
its output buffer before its input.

`test.cpp` dispatches the kernel `N` times with no gap. Run `r` gets the input
`((r + 1) << 16) | i` and an output buffer filled with `0xdeadbeef`. Every
element of every run is checked; on a mismatch the host prints the first wrong
index and where the value came from. Every wait is bounded (`--timeout-ms`,
default 5000), so a hung dispatch is reported as a `TIMEOUT`. The host also
prints the time per dispatch: in `seq` mode the first run and the median of
the others, otherwise the total divided by `N`.

| `--mode` | Dispatch |
| --- | --- |
| `seq` | start a run, wait for it, start the next, with no sleep |
| `runlist` | one `xrt::runlist` holding all N runs |
| `startall` | start all N runs, then wait for each |
| `alternate` | `seq`, alternating between `--elf` and `--alt-elf` on one hardware context (at most 8 runs) |
| `aba` | `seq` over `back_to_back_dispatch_aba.elf` (kernel `main:seq`): each run configures A (`in + 1`), B (`in + 2`), then A again, on the same tiles and locks, into three outputs (at most 8 runs) |

## Build and run

Run from `test/npu-xrt/back_to_back_dispatch` with the mlir-aie environment enabled.

```bash
make                  # back_to_back_dispatch.elf, back_to_back_dispatch_outfirst.elf, test.exe
make run              # seq, runlist and startall, N=200 each, then output-first
./test.exe --elf back_to_back_dispatch.elf --mode runlist --n 200
```

The lit test `run_npu3_makefile.lit` runs `seq`, `runlist` and `startall` with
N=200, the output-first design in `seq` and `runlist` mode, and the four
reloading cases below with N=8.

## Reloading cases

`make extra` builds three more ELFs that reconfigure live memtile channels.
The configure-time reset makes these normal regression cases;
`make run-extra` runs each with N=8:

```bash
make run-expand       # ./test.exe --elf back_to_back_dispatch_expand.elf --mode seq --n 8
make run-aba          # ./test.exe --elf back_to_back_dispatch_aba.elf --mode aba --n 8
make run-alternate    # ./test.exe --elf back_to_back_dispatch.elf \
                      #   --alt-elf back_to_back_dispatch_b.elf --mode alternate --n 8
```

**Keep these runs bounded.** Disabling the reset displaces output data and can
leave the device needing a reboot.

## Notes

* The design's core loops `sys.maxsize` times. A core that runs to completion
  would not run again on a repeated dispatch, since the load is skipped.
* This design runs on one microcontroller. A program on several (one
  `cert.attach_to_group` per uC) keeps its configuration as job 1.
