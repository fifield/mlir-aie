# MLIR Execution Trace Example

This example demonstrates **Execution trace mode** on the AMD NPU. It runs a simple
vector-scalar multiplication on a compute tile, captures trace data during execution,
and reconstructs the program's control flow — branches taken/not-taken, indirect jumps,
and loop iterations.

## Trace Modes

The NPU trace unit supports three modes. Each example in this directory family
demonstrates one:

| Mode | Example | What it captures |
|------|---------|-----------------|
| [Event-Time](../event_trace) | `event_trace` | A timestamp each time a configured event fires |
| [Event-PC](../mlir_pctrace) | `mlir_pctrace` | The Program Counter address when each event fires |
| **Execution** | **this example** | Control flow: branches taken/not-taken, loop counters |

Execution mode automatically traces every control-flow change without requiring you
to select specific events. The trace stream contains:

- **E_atom** — direct branch *taken*
- **N_atom** — direct branch *not taken*
- **New_PC** — indirect branch with new PC address
- **LC** — loop counter update

This is useful for reconstructing the exact execution path, analysing branch
prediction, and understanding loop behaviour. For timing, use Event-Time mode;
for instruction-level hotspot profiling, use Event-PC mode.

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

## Trace Configuration (`aie_new_trace.mlir`)

```mlir
aie.trace @exec_trace(%tile_0_2) {
    aie.trace.mode "Execution"
    aie.trace.packet id=1 type=core
    aie.trace.event<"INSTR_EVENT_0">
    aie.trace.start event=<"TRUE">
    aie.trace.stop  event=<"NONE">
}

aie.trace @shim_trace(%shim_noc_tile_0_0) {
    aie.trace.packet id=2 type=shimtile
    aie.trace.event<"DMA_S2MM_0_START_TASK">
    // ... shim DMA events
    aie.trace.start event=<"TRUE">
    aie.trace.stop  event=<"NONE">
}

aie.runtime_sequence(...) {
    aie.trace.host_config buffer_size = 8192
    aie.trace.start_config @shim_trace
    aie.trace.start_config @exec_trace
    // ... data transfers
}
```

Execution mode doesn't require selecting 8 events like Event-Time/Event-PC — it
automatically captures all branch and loop control flow. Packet flows and buffer
descriptors are generated automatically by the compiler trace lowering pipeline.

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
3. Decode `trace.txt` → `trace.json` via `python/utils/trace/parse_trace.py --mode execution`
4. Print a summary with branch statistics

## Files

| File | Description |
|------|-------------|
| `aie_new_trace.mlir` | **Primary source** — declarative Execution trace API |
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

Execution traces use a dedicated parser that understands the branch/loop frame format:

```
python/utils/trace/parse_trace.py --input trace.txt --output trace.json --mode execution
```

The standard `parse.py` only handles Event-Time traces.

## Related Examples

- [`../event_trace`](../event_trace) — Event-**Time** mode: timestamps when events fire
- [`../mlir_pctrace`](../mlir_pctrace) — Event-**PC** mode: PC address when events fire
