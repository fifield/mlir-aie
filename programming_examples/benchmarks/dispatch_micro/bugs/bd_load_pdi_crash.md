# Firmware timeout: `aiex.npu.load_pdi` + high shim-BD count

> Filed: 2026-05-18. Found during the dispatch_micro v1 sweep, validated
> during v2 follow-up.

## Symptom

Any dispatch of an MLIR runtime sequence that contains both
`aiex.npu.load_pdi` and ≥ 16 `aiex.dma_configure_task_for` ops (8 BDs
per direction in our generator) times out on the first run with no
diagnostic from the firmware:

```
terminate called after throwing an instance of 'xrt::run::aie_error'
  what():  Command failed to complete successfully (ERT_CMD_STATE_TIMEOUT)
txn_op_idx       = 0xFFFFFFFF
ctx_pc           = 0x28B060AD
fatal_error_type = 0x00000000
fatal_error_exception_type = 0x00000000
fatal_error_exception_pc   = 0x00000000
fatal_error_app_module     = 0x00000000
```

`txn_op_idx = 0xFFFFFFFF` indicates the firmware doesn't know which
opcode was executing — i.e. the transaction stream completed (or
deadlocked) before any handler ran, which is consistent with a stuck
BD chain rather than an exception in a specific op.

After the timeout, subsequent dispatches against any artifact often
fail to load (`std::runtime_error: not a valid ELF file`) until the
device is power-cycled.

## Reproducer

Generator + Makefile in `programming_examples/benchmarks/dispatch_micro/`:

```bash
cd programming_examples/benchmarks/dispatch_micro
make MECH=load_pdi_fw DEVICE=npu2_1col TILES=1 ROWS_PER_COL=1 BDS=8 \
     TOPOLOGY=linear
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib ./bench \
  --build-dir=build/load_pdi_fw_npu2_1col_t1_r1_b8_linear \
  --mechanism=load_pdi_fw --metric=pure_dispatch \
  --tiles=1 --rows-per-col=1 --bds=8 --warmup=0 --iters=1
```

Minimal failing MLIR (1 shim column, 1 compute tile, 8 BDs per
direction, plus `aiex.npu.load_pdi`): captured in
`bugs/repro_load_pdi_bds8.mlir`. It's 110 lines, dominated by 16
identical `aiex.dma_configure_task_for @{in,out}_c0 { aie.dma_bd ... }`
ops.

## What works (boundary)

The same harness at the same shape, but **without** the load_pdi op,
runs cleanly:

```bash
make MECH=baseline DEVICE=npu2_1col TILES=1 ROWS_PER_COL=1 BDS=8 \
     TOPOLOGY=linear
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib ./bench \
  --build-dir=build/baseline_npu2_1col_t1_r1_b8_linear \
  --mechanism=baseline --metric=pure_dispatch \
  --tiles=1 --rows-per-col=1 --bds=8 --warmup=0 --iters=1
# {"mechanism":"baseline",...,"ns":{"min":...,"p50":...}}
```

The two builds are identical placed IRON except for the
`aiex.npu.load_pdi { device_ref = @main }` op at the top of the
runtime sequence. The generator inserts this op only for
`load_pdi_fw` / `load_pdi_expanded` mechanisms.

Lower BD counts (2, 4) work with `load_pdi_*`. So does ≥ 1 tile with
`load_pdi_*` at BDs ≤ 4.

| mechanism            | tiles | bds | result                       |
|----------------------|------:|----:|------------------------------|
| `baseline`           |     1 |   2 | OK                           |
| `baseline`           |     1 |   8 | OK                           |
| `baseline`           |     2 |   8 | OK                           |
| `baseline`           |     8 |   8 | OK (intermittent)            |
| `load_pdi_fw`        |     1 |   2 | OK                           |
| `load_pdi_fw`        |     1 |   4 | OK                           |
| `load_pdi_fw`        |     1 |   8 | **TIMEOUT**                  |
| `load_pdi_fw`        |     2 |   8 | **TIMEOUT**                  |
| `load_pdi_fw`        |     4 |   8 | **TIMEOUT**                  |
| `load_pdi_fw`        |     8 |   8 | OK (sometimes, see below)    |
| `load_pdi_expanded`  |     2 |   8 | **TIMEOUT**                  |
| `load_pdi_expanded`  |     4 |   8 | **TIMEOUT**                  |
| `load_pdi_expanded`  |     8 |   8 | OK (sometimes, see below)    |

### History note

v1's REPORT.md called this out as "`load_pdi_* × tiles ∈ {2,4} × bds=8`
crashes the firmware" — t=1 and t=8 at bds=8 were marked OK in v1
data. v2 re-investigation with `warmup=0 --iters=1` (i.e. measure
the very first dispatch from a fresh process) shows the crash
reproduces at **t=1** and is intermittent at **t=8**. v1's run used
`warmup=10 --iters=100`, which likely meant the firmware sometimes
recovered before the first measured iteration. The boundary is
broader than v1 reported: the crash is a function of `load_pdi op
present + bds ≥ 8`, not specifically tile count.

## Suspected cause

Speculation only — needs firmware-side instrumentation to confirm:

- Each `aiex.dma_configure_task_for` emits one shim BD configuration.
  At `bds = 8 × {in, out}` we generate 16 BD setup ops in the txn
  stream.
- The shim DMA has hardware BD pools per channel. With 8 BDs queued
  on one MM2S channel (or S2MM), we may already be at or near the
  per-channel pool limit on the device the harness is targeting
  (Strix npu2).
- The `aiex.npu.load_pdi` op runs at the start of the runtime
  sequence. If the firmware's PDI load path internally allocates a
  scratch BD on the same shim channel — even briefly — it could
  collide with the bookkeeping for a BD index later in the stream,
  leaving the chain in a state where the firmware is waiting on a
  BD that will never fire.
- The `txn_op_idx = 0xFFFFFFFF` and silent timeout (no fatal_error_*)
  is consistent with the firmware's command dispatcher believing it
  has handed off to hardware and gone back to sleep, while the
  hardware sits on a never-completing BD chain.
- It's also consistent with a packet-routing bug where a token
  expected by `dma_await_task` never arrives, because the load_pdi
  expansion (or the firmware-side equivalent) interfered with the
  packet stream.

Counterpoints:
- `baseline + bds=8` also schedules 16 BD configs and does not
  crash. So pure BD count isn't the issue; the load_pdi op is
  load-bearing.
- `load_pdi_fw + bds=4` works at every tile count we tested. So the
  load_pdi op alone isn't the issue either; the combination matters.

## Suggested next steps

1. Capture XRT trace + `dmesg` around a timeout to see if the
   driver/firmware logs anything useful that `xrt::run::aie_error`
   doesn't surface.
2. Try reducing the BD count one at a time (5, 6, 7) at fixed tile
   count to find the exact threshold — if the boundary is sharp at 8,
   that's a strong clue for a hardware queue depth.
3. Run the failing MLIR through `aiecc -v --keep-loc` to see exactly
   what txn opcodes get emitted, and whether the load_pdi expansion
   path differs from the firmware path in BD bookkeeping.
4. Inspect whether the `aiex.dma_configure_task_for` ops are getting
   distinct `bd_id` attributes assigned, or whether they share one
   that collides with internal load_pdi state.

## Workaround for users

Don't combine `aiex.npu.load_pdi` with `bds ≥ 8` per shim channel
direction until this is resolved. Either:
- Reduce BD count to ≤ 4 per direction (works);
- Use the `baseline` mechanism (`--aie-generate-xclbin`, no
  `--generate-full-elf`) — works at any BD count we tested.

## Related: the `--no-self-reload` failure mode

Task #2 in V2_TODO.md (`--no-self-reload`: omit the `aiex.npu.load_pdi`
op at the top of the runtime sequence for `load_pdi_*` builds) was
intended to give us a "ELF baseline without load_pdi" measurement.
Every dispatch produces **the same `ERT_CMD_STATE_TIMEOUT,
txn_op_idx = 0xFFFFFFFF, fatal_error_* = 0` signature as this bug.**
Reproduced 6/6 times across mech ∈ {fw, expanded} × t ∈ {1, 4, 8}.

The two failure modes may share a root cause. Both are dispatched
through the full-ELF / `xrt::ext::kernel` path. In one case the
load_pdi op is *present* but the BD count is too high; in the other
the load_pdi op is *absent*. Both leave the firmware in a state where
no opcode is executing (`txn_op_idx = 0xFFFFFFFF`) and the command
times out without any specific exception.

Plausible shared cause: `load_pdi` is an XRT patch point (per the
comment in `test/npu-xrt/loadpdi/aie.mlir`, "XRT will load the PDI
into memory and patch the address of this load_pdi to the correct
address"). The patch process may need to coordinate with BD setup
in the same txn stream, and either skipping it or overloading it
(too many BDs to associate with the loaded PDI) puts the dispatch
in an unrecoverable state.

## Linked from

- REPORT.md "Failures" section and Anomaly #0 in "Anomalies worth chasing".
- V2_TODO.md tasks #5 and #2 (this file is the deliverable for #5
  and includes #2's negative finding above).
