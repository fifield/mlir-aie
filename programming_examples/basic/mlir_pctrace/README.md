# MLIR Event-PC Trace Example

This example demonstrates **Event-PC trace mode** on the AMD NPU. It runs a simple
vector-scalar multiplication on a compute tile, captures trace data during execution,
and decodes the trace to show which instructions triggered which events and which PC
addresses were hottest.

## Trace Modes

The NPU trace unit supports three modes. Each example in this directory family
demonstrates one:

| Mode | Example | What it captures |
|------|---------|-----------------|
| [Event-Time](../event_trace) | `event_trace` | A timestamp each time a configured event fires |
| **Event-PC** | **this example** | The Program Counter address when each event fires |
| [Execution](../mlir_exetrace) | `mlir_exetrace` | Control flow: branches taken/not-taken, loop counters |

Event-PC mode is useful when you want to know *where in the code* an event occurs —
e.g., which instruction causes a memory stall, or what the load/store PC looks like
during the hot loop. It does not give you wall-clock timing; use Event-Time mode
for that.

## The Kernel

`vector_scalar_mul.cc` multiplies a 4096-element `int32` vector by a scalar:

```cpp
void vector_scalar_mul_aie_scalar(int32_t *a, int32_t *c, int32_t *factor, int32_t N) {
    event0(); // INSTR_EVENT_0 — marks entry into the kernel
    for (int i = 0; i < N; i++) {
        c[i] = *factor * a[i];
    }
    event1(); // INSTR_EVENT_1 — marks exit from the kernel
}
```

`event0()` and `event1()` are AIE intrinsics that fire `INSTR_EVENT_0` and
`INSTR_EVENT_1` hardware events. These appear in the trace as PC captures at the
call site, letting you bracket the execution of a function.

## Trace Configuration (`aie_new_trace.mlir`)

```mlir
aie.trace @core_trace(%tile_0_2) {
    aie.trace.mode "Event-PC"
    aie.trace.packet id=1 type=core
    aie.trace.event<"INSTR_EVENT_0">   // user start marker
    aie.trace.event<"INSTR_EVENT_1">   // user end marker
    aie.trace.event<"INSTR_VECTOR">    // vector instruction issued
    aie.trace.event<"MEMORY_STALL">    // core stalled waiting for memory
    aie.trace.event<"LOCK_STALL">      // core stalled waiting for a lock
    aie.trace.event<"INSTR_LOAD">      // load instruction issued
    aie.trace.event<"INSTR_STORE">     // store instruction issued
    aie.trace.start event=<"TRUE">
    aie.trace.stop  event=<"NONE">
}

aie.runtime_sequence(...) {
    aie.trace.host_config buffer_size = 1048576
    aie.trace.start_config @core_trace
    // ... data transfers
}
```

Up to 8 events can be traced simultaneously. Each event fires independently; a
single 32-bit trace word encodes the PC address plus a bitmask of which of the 8
configured events fired at that PC.

Packet flows and buffer descriptors are generated automatically by the compiler
trace lowering pipeline.

## Building and Running

```bash
# Build xclbin and kernel object
make

# Run with C++ host — produces trace.txt and trace.json
make run_trace

# Run with Python host — same outputs
make run_trace_py

# Remove build artifacts and trace files
make clean
```

`run_trace` and `run_trace_py` both:
1. Execute the design on the NPU
2. Write raw trace words to `trace.txt` (one hex word per line)
3. Decode `trace.txt` → `trace.json` via `python/utils/trace/parse_trace.py --mode eventpc`
4. Print a summary to stdout

## Files

| File | Description |
|------|-------------|
| `aie_new_trace.mlir` | **Primary source** — declarative Event-PC trace API |
| `vector_scalar_mul.cc` | AIE kernel — scalar loop with `event0()`/`event1()` markers |
| `test.cpp` | C++ host application (XRT direct) |
| `test.py` | Python host application (pyxrt) |
| `Makefile` | Build system |
| `CMakeLists.txt` | CMake config for building the C++ host |

## Compiler Pipeline

The declarative `aie.trace` block replaces manual register writes with named event
strings and structured configuration. The compiler lowering pipeline converts them
to `aiex.npu.write32` instructions:

1. `-aie-insert-trace-flows` — adds packet flows for trace routing
2. `-aie-trace-to-config` — converts trace ops to register config
3. `-aie-trace-pack-reg-writes` — packs register writes
4. `-aie-inline-trace-config` — inlines trace config into runtime sequence

## Trace Parser

Event-PC traces use a dedicated parser that understands the PC+event-bitmask format:

```
python/utils/trace/parse_trace.py --input trace.txt --output trace.json --mode eventpc
```

The standard `parse.py` only handles Event-Time traces.

## Related Examples

- [`../event_trace`](../event_trace) — Event-**Time** mode: timestamps when events fire
- [`../mlir_exetrace`](../mlir_exetrace) — Execution mode: branch/loop control flow

