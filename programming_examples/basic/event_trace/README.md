# MLIR Event-Time Trace Example

This example demonstrates **Event-Time trace mode** on the AMD NPU. It runs a simple
vector-scalar multiplication on a compute tile, captures trace data during execution,
and produces a timeline of when events occurred.

## Trace Modes

The NPU trace unit supports three modes. Each example in this directory family
demonstrates one:

| Mode | Example | What it captures |
|------|---------|-----------------|
| **Event-Time** | **this example** | A timestamp each time a configured event fires |
| [Event-PC](../mlir_pctrace) | `mlir_pctrace` | The Program Counter address when each event fires |
| [Execution](../mlir_exetrace) | `mlir_exetrace` | Control flow: branches taken/not-taken, loop counters |

Event-Time mode is the most common choice for **performance profiling** — it tells you
*when* and *how long* events take. Use Event-PC mode if you need to know *where in the
code* an event occurs, and Execution mode for full branch/loop control-flow reconstruction.

## The Kernel

`vector_scalar_mul.cc` multiplies a 4096-element `int32` vector by a scalar:

```cpp
void vector_scalar_mul_aie_scalar(int32_t *a, int32_t *c, int32_t *factor, int32_t N) {
    event0(); // INSTR_EVENT_0 — marks entry into the kernel
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * (*factor);
    }
    event1(); // INSTR_EVENT_1 — marks exit from the kernel
}
```

`event0()` and `event1()` are AIE intrinsics that fire `INSTR_EVENT_0` and
`INSTR_EVENT_1` hardware events, letting you bracket the execution of a function.

## Trace Configuration

This example traces four tile types: core events, core memory events, mem tile
events, and shim tile events. Both a Python API and direct MLIR syntax are provided.

### Python API (`aie_trace.py`)

```python
import aie.utils.trace as trace_utils
from aie.utils.trace.events import PortEvent, CoreEvent, MemEvent

tiles_to_trace = [tile_0_2, tile_0_2, mem_tile_0_1, shim_tile_0_0]
trace_utils.configure_trace(
    tiles_to_trace,
    coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, ...],
)

@runtime_sequence(...)
def sequence(...):
    trace_utils.start_trace()
    # ... data transfers
```

**Note:** To trace both core and memory events on a core tile, list it twice in
`tiles_to_trace`.

### MLIR Syntax (`aie_trace.mlir`)

```mlir
aie.trace @core_trace(%tile_0_2) {
  aie.trace.mode "Event-Time"
  aie.trace.packet id=1 type=core
  aie.trace.event<"INSTR_EVENT_0">
  aie.trace.event<"INSTR_EVENT_1">
  aie.trace.port<0> port=DMA channel=0 direction=S2MM
  aie.trace.start broadcast=15
  aie.trace.stop broadcast=14
}

aie.runtime_sequence(...) {
  aie.trace.host_config buffer_size = 8192
  aie.trace.start_config @core_trace
  // ... data transfers
}
```

Packet flows and buffer descriptors are generated automatically by the compiler
trace lowering pipeline.

## Building and Running

```bash
# Build xclbin and kernel object
make

# Run with C++ host — produces trace.txt, trace.json, trace_timeline.png
make run_trace

# Run with Python host — same outputs
make run_trace_py

# Remove build artifacts and trace files
make clean
```

`run_trace` and `run_trace_py` both:
1. Execute the design on the NPU
2. Write raw trace words to `trace.txt` (one hex word per line)
3. Decode `trace.txt` → `trace.json` via `python/utils/trace/parse.py`
4. Print a summary and generate `trace_timeline.png`

## Files

| File | Description |
|------|-------------|
| `aie_trace.mlir` | Declarative Event-Time trace (MLIR) |
| `aie_trace.py` | Declarative Event-Time trace (Python `configure_trace()` API) |
| `vector_scalar_mul.cc` | AIE kernel — scalar loop with `event0()`/`event1()` markers |
| `test.cpp` | C++ host application (XRT direct) |
| `test.py` | Python host application (pyxrt) |
| `visualize_trace.py` | Renders a PNG timeline from parsed trace JSON |
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

Inspect intermediate IR:

```bash
aie-opt -aie-insert-trace-flows aie_trace.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config aie_trace.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes aie_trace.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config aie_trace.mlir
```

## Trace Parser

Event-Time traces are decoded by the standard library parser:

```
python/utils/trace/parse.py --input trace.txt --mlir <input_with_addresses.mlir> --output trace.json
```

## Related Examples

- [`../mlir_pctrace`](../mlir_pctrace) — Event-**PC** mode: PC address when events fire
- [`../mlir_exetrace`](../mlir_exetrace) — Execution mode: branch/loop control flow