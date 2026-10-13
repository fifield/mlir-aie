<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Core-side tile-mapped register access

XLLVM exposes Peano's existing `read.tm`, `write.tm`, and `sched.barrier`
intrinsics for AIE2, AIE2P, and AIE4. They execute inside the compute core and
can appear in loops, conditionals, or device helpers. They require no external
C++ kernel. Use the operations matching the core's architecture.

These are direct intrinsic wrappers. They do not configure the device from the
host, enable the processor bus, or synchronize neighboring cores.
`aiex.npu.write32` remains the runtime-sequence interface for array writes.

## Python

`aie.dialects.xllvm` supplies generated raw builders and three convenience
functions. The convenience functions take tile register offsets, like Peano's
C++ `read_tm` / `write_tm` with their default aperture:

```python
from aie.dialects.xllvm import read_tm, write_tm, sched_barrier

# AIE4 Accumulator_Control: input West, output East.
# Execute only after the previous cascade phase has drained.
sched_barrier(arch="aie4")
write_tm(3, 0xA1060, arch="aie4")
sched_barrier(arch="aie4")
direction = read_tm(0xA1060, arch="aie4")
sched_barrier(arch="aie4")
```

Values and addresses can be Python integers, signless i32 SSA values, or
single-result operations producing one. Inside `aie.device`, `arch` defaults to
the device's target and an explicit `arch` must match it; outside a device it is
required. Addresses must be 4-byte aligned register offsets within the target's
processor-bus window: `[0x10000, 0x80000)` on AIE2/AIE2P and
`[0xA0000, 0xB5000)` on AIE4 (core module, memory module and stream switch).
Constant addresses are checked against this; dynamic values must obey the same
contract at execution. Pass a tile-relative offset, not an array address with
column/row bits or an address already translated into the TM aperture. The
helper does not validate the register's permissions or reserved bits.

## Address interpretation

The raw XLLVM operations use the LLVM intrinsic ABI: `!llvm.ptr` in address
space zero already containing the **core-visible** address. No address
translation occurs in these operations.

| Target | Python helper translation from tile register offset R |
|---|---|
| AIE2 | R + 0x80000 |
| AIE2P | R + 0x80000 |
| AIE4 | R - 0x58000 |

The AIE4 expression is Peano's `0x40000 + R - 0x98000`; for
`Accumulator_Control`, `0xA1060` becomes `0x49060`. Arithmetic uses i32,
matching the unsigned C++ wrappers. A custom aperture can be expressed by
constructing the pointer and using a raw builder.

```mlir
%addr = llvm.mlir.constant(299104 : i32) : i32 // 0x49060
%ptr = llvm.inttoptr %addr : i32 to !llvm.ptr
%horizontal = llvm.mlir.constant(3 : i32) : i32
xllvm.intr.aie4.sched.barrier
xllvm.intr.aie4.write.tm %horizontal, %ptr : !llvm.ptr
xllvm.intr.aie4.sched.barrier
%value = xllvm.intr.aie4.read.tm %ptr : !llvm.ptr
xllvm.intr.aie4.sched.barrier
```

`write.tm` is a full 32-bit write. There is no implicit read-modify-write.
`read.tm` returns an i32 and cannot be folded or commoned even if its result
is unused: device state may change and reads can have side effects.

## Ordering and prerequisites

All three XLLVM operations have unknown effects, preserving their execution
and ordering through generic MLIR optimization. They translate one-for-one to
`llvm.aie2.*`, `llvm.aie2p.*`, or `llvm.aie4.*`. Peano supplies their
instruction semantics and schedules the resulting TM accesses.

Scheduling barriers remain explicit, just as in the C++ intrinsic API.
The existing `programming_examples/basic/dma_compression/kernel.cc` uses
barriers between TM accesses to avoid bundling multiple `st.tm` / `lda.tm`
instructions together. Bracket a configuration change when both prior and
subsequent operations must stay on their respective sides of it.

A scheduling barrier is a **compiler** barrier. It does not wait for a
neighbor, drain a stream/cascade, or provide a hardware completion handshake.
For a cascade direction flip, establish that all affected cores have drained
the old phase before changing direction; arrange any required locks or other
protocol separately.

Enable the core processor bus from device configuration/the runtime before
executing these operations. For the NPU3 cascade example this is bit 0 of tile
register `0xA3060`; AIE2/AIE2P use their own register maps. Access through a
disabled processor bus can hang. The operations intentionally do not perform
this setup themselves.

The primitive operations expose backend behavior; they do not prove a
particular reconfiguration protocol safe. Backend code-generation tests check
intrinsic selection and TM access separation. Hardware tests must additionally
validate the resource-specific protocol and repeated execution.
