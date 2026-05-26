# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Insert AIE hardware trace ops into an already-generated stitched MLIR-AIE
module, using mlir-aie's Python bindings.

Ported from npu-harness/scripts/trace_instrumentation.py and adapted for the
multi-launch stitched form this project emits: many named ``aie.device(npu2)
@<sub>`` blocks plus a top-level unnamed dispatcher device with
``aie.runtime_sequence @<instance>(...)``. Trace state is per-``aie.device``,
so a single call instruments exactly one chosen sub-device. The dispatcher
forwards all args through ``aiex.run @<sub>_sequence(...)``, so ``ddr_id=-1``
appends trace bytes to the dispatcher's last BO — the same host BO the user
sees.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class TraceInstrumentationError(RuntimeError):
    """The IR could not be instrumented (sub-device missing, tile missing,
    already instrumented, etc.)."""


def _build_default_core_events():
    """Eight events for the core trace unit (port mapping slots 0..2 cover
    S2MM ch0/1 and MM2S ch0 on the traced compute tile)."""
    from aie.utils.trace.events import CoreEvent, PortEvent
    from aie.dialects.aie import WireBundle
    return [
        CoreEvent.INSTR_EVENT_0,
        CoreEvent.INSTR_EVENT_1,
        CoreEvent.INSTR_VECTOR,
        CoreEvent.LOCK_STALL,
        CoreEvent.MEMORY_STALL,
        PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA,
                  channel=0, master=True),
        PortEvent(CoreEvent.PORT_RUNNING_1, port=WireBundle.DMA,
                  channel=1, master=True),
        PortEvent(CoreEvent.PORT_RUNNING_2, port=WireBundle.DMA,
                  channel=0, master=False),
    ]


def _build_default_core_mem_events():
    """Eight events for the memory-module trace unit on the same compute tile."""
    from aie.utils.trace.events import MemEvent
    return [
        MemEvent.DMA_S2MM_0_START_TASK,
        MemEvent.DMA_S2MM_1_START_TASK,
        MemEvent.DMA_MM2S_0_START_TASK,
        MemEvent.DMA_S2MM_0_FINISHED_TASK,
        MemEvent.DMA_S2MM_1_FINISHED_TASK,
        MemEvent.DMA_MM2S_0_FINISHED_TASK,
        MemEvent.DMA_S2MM_0_STREAM_STARVATION,
        MemEvent.DMA_S2MM_1_STREAM_STARVATION,
    ]


_CORE_EVENTS_DEFAULT_NAMES = (
    "INSTR_EVENT_0", "INSTR_EVENT_1", "INSTR_VECTOR",
    "LOCK_STALL", "MEMORY_STALL",
    "PORT_RUNNING_0", "PORT_RUNNING_1", "PORT_RUNNING_2",
)
_CORE_MEM_EVENTS_DEFAULT_NAMES = (
    "DMA_S2MM_0_START_TASK", "DMA_S2MM_1_START_TASK", "DMA_MM2S_0_START_TASK",
    "DMA_S2MM_0_FINISHED_TASK", "DMA_S2MM_1_FINISHED_TASK",
    "DMA_MM2S_0_FINISHED_TASK",
    "DMA_S2MM_0_STREAM_STARVATION", "DMA_S2MM_1_STREAM_STARVATION",
)


def _build_context():
    from aie.ir import Context, DialectRegistry
    from aie.dialects.aie import register_dialect as aie_register
    from aie.dialects.aiex import register_dialect as aiex_register

    ctx = Context()
    reg = DialectRegistry()
    aie_register(reg)
    aiex_register(reg)
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    return ctx


def _find_named_device(module, sub_device: str):
    """Return the named ``aie.device(npu2) @<sub_device>`` op, or raise."""
    found = []
    for op in module.body.operations:
        if op.operation.name != "aie.device":
            continue
        sym = op.attributes.get("sym_name")
        if sym is None:
            continue
        name = str(sym).strip('"')
        found.append(name)
        if name == sub_device:
            return op
    raise TraceInstrumentationError(
        f"sub_device '{sub_device}' not found. Available: {found}"
    )


def _find_runtime_sequence(device_op):
    for op in device_op.regions[0].blocks[0].operations:
        if op.operation.name == "aie.runtime_sequence":
            return op
    raise TraceInstrumentationError(
        "no aie.runtime_sequence inside chosen sub-device"
    )


def _find_tile_at(device_op, col: int, row: int):
    """Exact-coords tile lookup inside ``device_op``."""
    if row < 2:
        raise TraceInstrumentationError(
            f"tile row {row} < 2 (only compute tiles are traceable)"
        )
    for op in device_op.regions[0].blocks[0].operations:
        if op.operation.name != "aie.tile":
            continue
        c = int(op.attributes["col"])
        r = int(op.attributes["row"])
        if c == col and r == row:
            # Confirm there's an aie.core on it.
            if _find_core_for_tile(device_op, col, row) is None:
                raise TraceInstrumentationError(
                    f"tile ({col},{row}) declared but has no aie.core"
                )
            return op
    declared = sorted(
        (int(op.attributes["col"]), int(op.attributes["row"]))
        for op in device_op.regions[0].blocks[0].operations
        if op.operation.name == "aie.tile"
    )
    raise TraceInstrumentationError(
        f"tile ({col},{row}) not declared in sub-device. "
        f"Declared tiles: {declared}"
    )


def _runtime_sequence_has_args(runtime_seq_op) -> bool:
    block = runtime_seq_op.regions[0].blocks[0]
    return len(block.arguments) > 0


_INTMAX = 2147483647

_SHIM_S2MM_CHANNELS = 2
_DEVICE_SHIM_COLUMNS = 8


def _shim_s2mm_usage(device_op):
    """Count shim S2MM channel uses by walking objectfifo / shim_dma_allocation."""
    usage: dict[tuple[int, int], int] = {}
    block = device_op.regions[0].blocks[0]
    for op in block.operations:
        if op.operation.name != "aie.tile":
            continue
        col = int(op.attributes["col"])
        row = int(op.attributes["row"])
        if row == 0:
            usage[(col, row)] = 0
    # Walk shim_dma_allocation ops which directly declare an (S2MM, channel) on a shim tile.
    for op in block.operations:
        if op.operation.name != "aie.shim_dma_allocation":
            continue
        # Operand 0 is the shim tile; the direction is an attribute "channelDir".
        # Be tolerant of name differences across mlir-aie versions.
        chan_dir = None
        for k in ("channelDir", "channel_dir"):
            try:
                chan_dir = str(op.attributes[k])
                break
            except (KeyError, AttributeError):
                continue
        if chan_dir is None or "S2MM" not in chan_dir:
            continue
        tile_def = op.operands[0].owner
        if tile_def.operation.name != "aie.tile":
            continue
        col = int(tile_def.attributes["col"])
        row = int(tile_def.attributes["row"])
        if (col, row) in usage:
            usage[(col, row)] += 1
    return usage


def _check_shim_capacity(device_op) -> None:
    """Refuse only if all eight shim columns are declared AND each is full.
    Cheap insurance — never fires on the IRs this project emits today."""
    usage = _shim_s2mm_usage(device_op)
    if not usage:
        return
    declared = len(usage)
    if declared < _DEVICE_SHIM_COLUMNS:
        return
    if all(n >= _SHIM_S2MM_CHANNELS for n in usage.values()):
        used = ", ".join(f"({c},{r})={n}" for (c, r), n in sorted(usage.items()))
        raise TraceInstrumentationError(
            "all_shim_s2mm_channels_occupied: every column declared and full "
            f"({used}); trace has no free channel device-wide."
        )


def _find_core_for_tile(device_op, col: int, row: int):
    for op in device_op.regions[0].blocks[0].operations:
        if op.operation.name != "aie.core":
            continue
        tile_def = op.operands[0].owner
        if tile_def.operation.name != "aie.tile":
            continue
        if (int(tile_def.attributes["col"]) == col
                and int(tile_def.attributes["row"]) == row):
            return op
    return None


def _find_outer_dispatch_loop(core_op):
    """Outer scf.for with upper bound == INT_MAX (the kernel's forever loop)."""
    core_body = core_op.regions[0].blocks[0]
    for op in core_body.operations:
        if op.operation.name != "scf.for":
            continue
        upper = op.operands[1]
        upper_def = upper.owner
        if upper_def.operation.name == "arith.constant":
            try:
                if int(upper_def.attributes["value"]) == _INTMAX:
                    return op
            except (TypeError, ValueError):
                pass
    return None


def _insert_event_markers(core_op) -> bool:
    """Bracket the outer dispatch loop body with aie.event(0) / aie.event(1).

    Returns False if the INTMAX outer-loop pattern isn't present (e.g. some
    eltwise / rms sub-devices have a different control-flow shape).
    """
    from aie.ir import InsertionPoint
    from aie.dialects.aie import EventOp
    outer_for = _find_outer_dispatch_loop(core_op)
    if outer_for is None:
        return False
    for_body = outer_for.regions[0].blocks[0]
    with InsertionPoint.at_block_begin(for_body):
        EventOp(0)
    terminator = list(for_body.operations)[-1]
    with InsertionPoint(terminator):
        EventOp(1)
    return True


def _strip_invalid_terminator(runtime_seq_op) -> bool:
    """Drop a stray ``aie.end`` at the end of the runtime_sequence body, if any.

    Returns True if stripped. The default compile path tolerates this, but
    some downstream verifiers don't, so we normalize it.
    """
    block = runtime_seq_op.regions[0].blocks[0]
    ops = list(block.operations)
    if ops and ops[-1].operation.name == "aie.end":
        ops[-1].operation.erase()
        return True
    return False


def instrument_ir_for_trace(
    ir_text: str,
    *,
    sub_device: str,
    col: int,
    row: int,
    trace_size: int = 8 * 1024 * 1024,
    add_event_markers: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Parse ``ir_text``, insert trace ops on (sub_device, col, row), return
    (instrumented_ir_text, info_dict).

    Raises ``TraceInstrumentationError`` on any precondition failure.
    """
    if "aie.trace" in ir_text:
        raise TraceInstrumentationError(
            "IR already contains aie.trace ops (already instrumented)"
        )

    from aie.ir import InsertionPoint, Location, Module
    from aie.utils.trace import configure_trace, start_trace

    core_events = _build_default_core_events()
    core_mem_events = _build_default_core_mem_events()

    ctx = _build_context()
    with ctx, Location.unknown():
        module = Module.parse(ir_text)
        device_op = _find_named_device(module, sub_device)
        rt_seq = _find_runtime_sequence(device_op)

        if not _runtime_sequence_has_args(rt_seq):
            raise TraceInstrumentationError(
                "sub-device runtime_sequence has no tensor args (ddr_id=-1 "
                "needs a last buffer to append to)"
            )

        _check_shim_capacity(device_op)
        stripped_terminator = _strip_invalid_terminator(rt_seq)

        tile_op = _find_tile_at(device_op, col, row)

        with InsertionPoint(rt_seq):
            # Pass the same tile twice — configure_trace treats the second
            # occurrence of a core tile as the memory-module trace unit,
            # giving us per-DMA-task events multiplexed by packet id.
            configure_trace(
                tiles_to_trace=[tile_op, tile_op],
                coretile_events=core_events,
                coremem_events=core_mem_events,
            )

        rt_body = rt_seq.regions[0].blocks[0]
        with InsertionPoint.at_block_begin(rt_body):
            start_trace(trace_size=trace_size, ddr_id=-1)

        event_markers_inserted = False
        if add_event_markers:
            core_op = _find_core_for_tile(device_op, col, row)
            if core_op is not None:
                event_markers_inserted = _insert_event_markers(core_op)

        return str(module), {
            "sub_device": sub_device,
            "tile": [col, row],
            "trace_size_bytes": trace_size,
            "core_events": list(_CORE_EVENTS_DEFAULT_NAMES),
            "core_mem_events": list(_CORE_MEM_EVENTS_DEFAULT_NAMES),
            "stripped_invalid_terminator": stripped_terminator,
            "event_markers_inserted": event_markers_inserted,
        }


def parse_trace_spec(spec: str) -> tuple[str, str, int, int]:
    """Parse ``KERNEL:SUB_DEVICE:COL:ROW`` into a 4-tuple."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"trace target must be KERNEL:SUB_DEVICE:COL:ROW, got {spec!r}"
        )
    kernel, sub_device, col_s, row_s = parts
    try:
        col = int(col_s)
        row = int(row_s)
    except ValueError as e:
        raise ValueError(
            f"trace col/row must be ints, got col={col_s!r} row={row_s!r}"
        ) from e
    return kernel, sub_device, col, row


def instrument_ir_for_trace_subprocess(
    ir_text: str,
    *,
    sub_device: str,
    col: int,
    row: int,
    trace_size: int = 8 * 1024 * 1024,
    add_event_markers: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Same contract as ``instrument_ir_for_trace`` but runs in a clean
    subprocess so it can't be contaminated by a parent-process MLIR context.

    Required because the per-kernel IR-gen step uses mlir-air's Python
    bindings to build the upstream module, leaving global pybind state in
    this process. Running ``Module.parse`` on the post-aircc text inside the
    same process trips an MLIR ``Operation::create`` assertion ("unexpected
    successors in a non-terminator operation"). A fresh process avoids the
    interaction entirely.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="aie_trace_instr_") as tmp:
        in_path = os.path.join(tmp, "in.mlir")
        out_path = os.path.join(tmp, "out.mlir")
        info_path = os.path.join(tmp, "info.json")
        with open(in_path, "w") as f:
            f.write(ir_text)
        cmd = [
            sys.executable, "-m", "kernel_builder.aie_trace_instrument",
            "--in", in_path, "--out", out_path,
            "--info", info_path,
            "--sub-device", sub_device,
            "--col", str(col), "--row", str(row),
            "--trace-size", str(trace_size),
        ]
        if not add_event_markers:
            cmd.append("--no-event-markers")
        # kernel_builder/ lives next to this file; run with PYTHONPATH set to
        # its parent so `-m kernel_builder.aie_trace_instrument` resolves.
        pkg_parent = str(Path(__file__).resolve().parent.parent)
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            pkg_parent
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise TraceInstrumentationError(
                f"trace instrument subprocess failed:\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        with open(out_path) as f:
            instrumented = f.read()
        with open(info_path) as f:
            info = json.load(f)
        return instrumented, info


def _cli(argv):
    """CLI entry point used by ``instrument_ir_for_trace_subprocess``."""
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--info", dest="info_path", required=True)
    p.add_argument("--sub-device", required=True)
    p.add_argument("--col", type=int, required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--trace-size", type=int, default=8 * 1024 * 1024)
    p.add_argument("--no-event-markers", action="store_true")
    args = p.parse_args(argv)
    with open(args.in_path) as f:
        ir_text = f.read()
    out, info = instrument_ir_for_trace(
        ir_text,
        sub_device=args.sub_device,
        col=args.col, row=args.row,
        trace_size=args.trace_size,
        add_event_markers=not args.no_event_markers,
    )
    with open(args.out_path, "w") as f:
        f.write(out)
    with open(args.info_path, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    import sys
    _cli(sys.argv[1:])
