<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Interleaving two separately compiled ELFs that reuse a stream port

Two AIE4 full ELFs, compiled separately, are loaded into one hardware context
and dispatched alternately (A, B, A, B, ...) with no gap. Both receive the same
shim stream on the same memtile port but send it to different memtile DMA
channels. Every run must return exactly its own output. This is the hardware
check for the stream switch clear (phase S) of the AIE4 configure-time reset
(see [The stream switch clear](#the-stream-switch-clear) below).

## Why it needs the stream switch clear

Each dispatch of A after B (or B after A) configures the array again: an
AIE4 full ELF's configuration is a cert `LOAD_PDI` section, which the firmware
skips only when the same PDI id was loaded last.
A and B have different ids, so every transition reloads. A configuration
writes only the switch ports its own design uses. When two separately compiled
designs alternate on one context, a connection that the previous design set up
and the next one does not use stays in place.

| | memtile (0, 1) stream connections | memtile DMA |
| --- | --- | --- |
| A (`design_a.mlir`) | `South:0 -> DMA:0`, `DMA:0 -> South:0` | S2MM 0 -> MM2S 0, plain copy |
| B (`design_b.mlir`) | `South:0 -> DMA:2`, `DMA:2 -> South:0` | S2MM 2 -> MM2S 2, row permutation |

Both use shim (0, 0) MM2S 0 for the input and S2MM 0 for the output, and both
routes go through shim North:0 and memtile South:0 (the switchboxes are written
by hand so the router cannot pick another port). Memtile South:0 can reach only
DMA 0, 2 and 3, so S2MM 0 and S2MM 2 are the two choices for the same input
port.

If B's configuration leaves A's `South:0 -> DMA:0` in place, the stream
manager port DMA:0 still selects South:0, so B's input is broadcast to both
S2MM 2 and S2MM 0. S2MM 0 has no BD queued in B (the configure-step DMA reset
has just reset it), so it would never accept data: the broadcast is expected to
stall, and B would never complete. The same is expected for A after B, with
`South:0 -> DMA:2`.

Phase S zeroes every stream switch configuration register in the partition
before the configuration writes its own, so each design starts from no
connections.

### The stream switch clear

The configure-time reset quiesces the whole partition at the top of every AIE4
configuration: of each `LOAD_PDI` section, and of each configuration written in
line under `--expand-load-pdis`, which follows an `aiex.npu.write_reset`.
Phase S runs after the cores are held in reset and read
back, and before the DMA channel reset. On every NoC shim tile, memtile and
core tile it writes zero to every writable stream switch configuration
register of both the 512-bit data switch and the 32-bit switch: manager and
subordinate port configuration and deterministic merge (not parity, event
port selection or status registers), plus the packet slots of the tiles where
the incoming configuration has packet rules; elsewhere no port is left in
packet mode, so a stale slot is inert. The registers come from the register
database, in runs of consecutive offsets (15 per core tile, 30 per memtile, 21
per shim tile without packet slots), each emitted as one zero block write,
which becomes a uC-DMA chain in cert. On a full npu3 without packet routes
that is about 6.4 KB of zeros, so the configuration needs more than one cert
page.

The configuration then installs its own routes, control-packet overlay
included. A configuration also disables the connections of the other devices
in the same MLIR module (`disableOtherDeviceSwitches`), which cannot see a
separately compiled ELF; after S those writes are redundant.

S does not drain data already in switch FIFOs: the outgoing design must have
finished its streams.

### Static evidence

`stale_route_static.lit` (no hardware) checks the generated configuration. On
memtile (0, 1), `Stream_Switch_Manager_Config_DMA_0` is at `0x120000` and
`Stream_Switch_Manager_Config_DMA_2` at `0x120008`; `0x8000000e` enables a
manager port and selects subordinate South:0.

* A's configuration writes `0x120000 = 0x8000000e` (`South:0 -> DMA:0`).
* B's configuration alone (`convert-aie-to-transaction`, which does not add
  the reset) writes `0x120008 = 0x8000000e` and nothing writes `0x120000`, so
  A's connection would survive.
* B's `LOAD_PDI` section: S writes zeros to `0x120000..0x12000c` (one
  `UC_DMA_BD`), then B writes `0x120008 = 0x8000000e`.

## What the test checks

`test.cpp` creates one `xrt::hw_context` from `design_a.elf`, adds
`design_b.elf` with `hw_context::add_config`, and gets the kernels
`stream_a:sequence` and `stream_b:sequence`. Run `r` uses A when `r` is even
and B when `r` is odd. Each run gets the input `((r + 1) << 16) | i` and an
output buffer filled with `0xdeadbeef`; every element of every run is checked
against A's copy or B's permutation (`out row 4 * i + k = in row i + 8 * k`,
rows of 32 elements). Every wait is bounded (`--timeout-ms`, default 5000), so
a hung dispatch is reported as a `TIMEOUT` instead of hanging the test.

| `--mode` | Dispatch |
| --- | --- |
| `seq` | start a run, wait for it, start the next, with no sleep |
| `runlist` | one `xrt::runlist` holding all N runs of both kernels |
| `startall` | start all N runs, then wait for each |

A and B have no cores, so no kernel is compiled.

## Build and run

```bash
make          # design_a.elf, design_b.elf and test.exe
make run      # seq, runlist and startall, N=32 each (override: make run N=40)
./test.exe --mode runlist --n 32
```

The lit test `run_npu3_makefile.lit` runs the three modes for both builds.
`stale_route_static.lit` also checks, without hardware, that B's `LOAD_PDI`
section carries the clear.

Expected output, per mode:

```
mode seq: 32/32 runs exact (A,B alternating)
PASS
```

On npu9 every mode of both builds returned 32/32 runs exact.

## Seeing the bug

There is no build without the clear: every AIE4 configuration section starts
with the reset. `stale_route_static.lit` shows the hazard statically instead:
B's whole configuration touches memtile switch registers `0x120008`,
`0x12005c`, `0x120014` and `0x120048` only, so without S, A's
`0x120000 = 0x8000000e` (`South:0 -> DMA:0`) would still be set when B streams.
The hardware PASS above shows that with the clear, separately compiled ELFs
interleaved on one hardware context return exact output; the hazard itself has
not been shown on hardware (a hung dispatch on npu3 has needed a reboot to
recover).
