# AIE4 MLIR Trace Example

AIE4 (npu3) port of the `basic/event_trace` example. Demonstrates declarative trace configuration on an AMD NPU3 device using a vector-scalar multiply kernel.

## Contents

- `aie_trace.mlir` - declarative trace configuration for npu3 (`aie.trace`, `aie.trace.event`, `aie.trace.start_config`)
- `aie_trace.py` - Python equivalent of `aie_trace.mlir`
- `vector_scalar_mul.cc` - AIE4 kernel (compiled with chess `xchesscc_wrapper aie4`)
- `test.cpp` / `test.py` - host runners (XRT-based, same interface as AIE2)
- `visualize_trace.py` - renders a PNG timeline from parsed trace JSON

## AIE4 Differences from AIE2

| Feature | AIE2 | AIE4 (npu3) |
|---------|------|-------------|
| Device  | `npu1_1col` | `npu3` |
| Core trace unit 0 | `Trace_Control0/1`, `Trace_Event0/1` | `Trace0_Control0/1`, `Trace0_Event0/1` |
| Core trace unit 1 | Memory module: `Trace_Control0/1` | Same tile, Trace1: `Trace1_Control0/1`, `Trace1_Event0/1` |
| Shim trace | `Trace_Control0/1` | `Trace_A_Control0/1`, `Trace_A_Event0/1` |
| Shim broadcast | `Event_Broadcast15_A` @ 0x3404C | `Event_Broadcast_A_15` @ 0x40EC |
| Shim timer | `Timer_Control` @ 0x34000 | `Timer_A_Control` @ 0x4090 |
| Core timer | `Timer_Control` @ 0x34000 | `Timer_Control` @ 0xA0110 |
| Trace DMA buffer | S2MM ch1 BD15 @ 0x1D1E0 | S2MM ch1 BD0 @ 0x58000 |
| Port field name | `Port_N_Master_Slave` | `Port_N_Manager_Subordinate` |
| Kernel compiler | Peano (`clang++`) | Chess (`xchesscc_wrapper aie4`) |
| BROADCAST_15 (core) | event 122 | event 233 |
| INSTR_EVENT_0 (core) | event 33 | event 62 |

## Trace Buffer DMA Setup (AIE4)

AIE4 shim DMA S2MM channel 1 uses a separate BD region starting at `0x58000`:
- BD0 base: `0x58000`
- BD0 Base_Address_Low (address field): `0x58004 = 360452`
- DMA_S2MM_1_Task_Queue: `0x58544 = 361796`

`npu_writebd` requires `channel=1, direction=0` (S2MM) to use the correct BD space.

## Run

```bash
make run_trace_py
```

## Inspect intermediate IR

```bash
aie-opt -aie-trace-to-config aie_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes aie_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config aie_trace.mlir
```

## Trace Configuration

```mlir
aie.trace @core_trace(%tile_0_2) {
  aie.trace.mode "Event-Time"
  aie.trace.packet id=1 type=core   // -> Trace0 registers
  aie.trace.event<"INSTR_EVENT_0">
  aie.trace.event<"INSTR_EVENT_1">
  aie.trace.start event=<"BROADCAST_15">
  aie.trace.stop event=<"BROADCAST_14">
}

aie.trace @mem_trace(%tile_0_2) {
  aie.trace.packet id=3 type=mem    // -> Trace1 registers (AIE4: no memory module)
  aie.trace.event<"DMA_S2MM_0_START_TASK">
  ...
}
```

Copyright 2026 Advanced Micro Devices, Inc.
