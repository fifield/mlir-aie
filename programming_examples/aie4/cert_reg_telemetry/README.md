# CERT register telemetry

This example uses CERT job-runner registers as a tiny control-code compute
engine and uses the uC debug stream as telemetry. It contains two designs with
different record formats:

- `cert_reg_snapshot.mlir` performs a pointer-walking computation and captures
  its two results with `cert.save_register`.
- `cert_reg_profile.mlir` uses `cert.save_timestamps` to measure a register-only
  arithmetic block and an AXI-MM read/modify/write block.

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

`cert_reg_snapshot.mlir` uses registers as pointers and values:

```text
r0 = &obuf[0]
r1 = *r0
r1 += 0x23
r0 += 4
*r0 = r1
SAVE_REGISTER(&obuf[1], 201)

r2 = obuf[1]
r2 += 0x100
obuf[2] = r2
SAVE_REGISTER(&obuf[2], 202)
```

Starting from `obuf[0] = 0x100`, the final buffer is:

| Word | Value | Meaning |
|---|---:|---|
| `obuf[0]` | `0x100` | seed |
| `obuf[1]` | `0x123` | dynamic-address/dynamic-value result |
| `obuf[2]` | `0x223` | immediate-address/dynamic-value result |

The debug buffer contains two 8-byte records:

```text
{201, 0x123}
{202, 0x223}
```

The host requires both paths to agree. A bad job-runner register computation,
a bad `SAVE_REGISTER`, and a missing DMA therefore produce distinct failures.

## Profiling design

`cert_reg_profile.mlir` places four timestamp records around two workloads:

```text
101 --[ 32 x ADD ]-- 102 --[ adjacent-stamp baseline ]-- 103
    --[ 8 x READ_32_D + ADD + WRITE_32_D ]-- 104
```

The host verifies `{101, 102, 103, 104}` ordering, checks that both workload
gaps exceed the adjacent-stamp baseline, and prints the measured tick counts.
It also checks the independently returned results:

- the arithmetic register reaches 32 and is written to `obuf[1]`;
- eight in-place AXI-MM increments change `obuf[0]` from 1 to 9.

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
