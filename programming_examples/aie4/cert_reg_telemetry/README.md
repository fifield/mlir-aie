# CERT register telemetry

This example uses CERT job-runner registers to move data with control code
and uses the uC debug stream as telemetry. It contains two designs with
different record formats:

- `cert_reg_snapshot.mlir` performs a pointer-chasing copy and captures its two
  results with `cert.save_register`.
- `cert_reg_profile.mlir` uses `cert.save_timestamps` to measure an AXI-MM read
  block and an AXI-MM copy block.

The ISA has no register arithmetic and no way to load an immediate into a
register (`MOV` and `ADD` were removed in aiebu #397), so registers
are only ever loaded from memory with `READ_32` / `READ_32_D`.

These are application-shaped examples rather than another opcode-by-opcode
test. The exhaustive semantic regressions for the individual operations live
under `test/npu-xrt/npu3_cert_*`.

## Build and run

An NPU3 system with XRT and the mlir-aie tools on `PATH` is required.

```sh
make
make run
```

The individual commands are:

```sh
./test_snapshot.exe cert_reg_snapshot.elf
./test_profile.exe cert_reg_profile.elf
```

Both designs are coreless: no AIE core compiler is needed. CERT writes a
tile(0,2) buffer, releases a lock, and a core-tile DMA returns the buffer to the
host. That ordinary DMA path is deliberately independent of the debug-buffer
path and serves as a liveness and value cross-check.

## Snapshot design

`cert_reg_snapshot.mlir` uses registers as pointers and values. The pointers
and payloads are first seeded into `obuf` with `WRITE_32`:

```text
r0 = obuf[0]        // &obuf[1]
r3 = obuf[3]        // &obuf[4]
r1 = *r3
*r0 = r1
SAVE_REGISTER(&obuf[1], 201)

r2 = obuf[5]
obuf[2] = r2
SAVE_REGISTER(&obuf[2], 202)
```

The final buffer is:

| Word | Value | Meaning |
|---|---:|---|
| `obuf[0]` | `0x600404` | destination pointer (`&obuf[1]`) |
| `obuf[1]` | `0x123` | dynamic-address/dynamic-value result |
| `obuf[2]` | `0x223` | immediate-address/dynamic-value result |
| `obuf[3]` | `0x600410` | source pointer (`&obuf[4]`) |
| `obuf[4]` | `0x123` | first payload |
| `obuf[5]` | `0x223` | second payload |

The debug buffer contains two 8-byte records:

```text
{201, 0x123}
{202, 0x223}
```

The host requires both paths to agree. A bad job-runner register load or store,
a bad `SAVE_REGISTER`, and a missing DMA therefore produce distinct failures.

## Profiling design

`cert_reg_profile.mlir` places four timestamp records around two workloads:

```text
101 --[ 32 x READ_32 ]-- 102 --[ adjacent-stamp baseline ]-- 103
    --[ 8 x READ_32 + WRITE_32_D ]-- 104
```

The host verifies `{101, 102, 103, 104}` ordering, checks that both workload
gaps exceed the adjacent-stamp baseline, and prints the measured tick counts.
It also checks the independently returned results:

- eight AXI-MM copies propagate the `obuf[0]` seed through `obuf[1..8]`;
- the value the read block last loaded into `r0` is written to `obuf[9]`.

The comparison is intentionally descriptive, not a performance contract.
Absolute uC timing varies by firmware and device, so the example does not
assert that one workload must be a particular multiple of the other.

## Why there are two designs

The firmware debug stream has no per-record type tag. `SAVE_REGISTER` emits
two 32-bit words (`{unq_id, value}`), while `SAVE_TIMESTAMPS` emits three
(`{unq_id, timestamp_high, timestamp_low}`). Keeping the record types in
separate designs lets a reader decode each stream without relying on a
hard-coded mixed-record schedule.

## XRT debug-buffer API

The uC debug buffer is not an ordinary `xrt::bo`. It must be allocated with the
`uc_debug` use type and registered with the hardware context so firmware knows
its address and size. XRT currently exports the required functions from
`libxrt_coreutil`, but declares them only in its non-installed
`core/common/api/bo_int.h` header.

`cert_debug_buffer.h` reproduces those two declarations and the `use_type`
enumeration. This is an intentional internal-ABI dependency: if XRT changes the
declarations, the example should fail loudly at link time, at which point the
header must be synchronized with the XRT source.
