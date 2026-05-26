# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side support for AIE event-trace capture.

A ``TraceTarget`` (parsed from ``KERNEL:SUB_DEVICE:COL:ROW``) tells the
KernelCache which kernel name to instrument and which compute tile to trace.
A ``TraceState`` collects raw uint32 trace words across all launches of that
kernel during a run, then ``flush()`` writes ``trace/trace.npy`` and
``trace/raw_trace.txt`` (and the instrumented MLIR is cached alongside as
``<kernel>.npu.air.mlir``, so postmortem parsers have everything they need).

The on-disk artifacts match the mlir-aie reference format
(``test_utils::write_out_trace``: zero-filtered uint32 words, 8 lowercase hex
chars per line), so ``aie.utils.trace.parse_trace`` can consume the raw text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TraceTarget:
    """Names the single kernel to instrument and the tile to trace."""

    kernel: str            # cache key, == XRT main:<instance>
    sub_device: str        # named aie.device(npu2) @<sub_device>
    col: int               # compute tile col
    row: int               # compute tile row (>= 2)
    trace_size: int = 8 * 1024 * 1024


@dataclass
class TraceState:
    """Accumulates raw trace bytes across many launches of the target kernel.

    ``append(bytes)`` is called from KernelCache.load_and_run after each run
    of the instrumented kernel. ``flush(out_dir)`` writes the combined raw
    trace to disk.
    """

    target: TraceTarget
    info: dict = field(default_factory=dict)  # filled in from instrument_ir_for_trace
    launches: list[np.ndarray] = field(default_factory=list)
    sanity_failures: int = 0
    sanity_checks: int = 0

    def append(self, trace_bytes: bytes) -> None:
        """Record one launch's worth of raw trace bytes."""
        if not trace_bytes:
            return
        # Pad to a multiple of 4 so frombuffer doesn't complain.
        n = len(trace_bytes) - (len(trace_bytes) % 4)
        if n <= 0:
            return
        words = np.frombuffer(trace_bytes[:n], dtype=np.uint32).copy()
        self.launches.append(words)

    def record_sanity(self, ok: bool) -> None:
        self.sanity_checks += 1
        if not ok:
            self.sanity_failures += 1

    def total_words(self) -> int:
        return sum(int(a.size) for a in self.launches)

    def flush(self, out_dir: Path) -> dict:
        """Write trace.npy + raw_trace.txt + meta.json into out_dir/trace/.

        Also writes trace.json (chrome://tracing format) if mlir-aie's
        ``parse_trace`` succeeds against the cached lowered IR
        (``<out_dir>/<kernel>.aie.mlir``).
        """
        out_dir = Path(out_dir)
        trace_dir = out_dir / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)

        all_words = (
            np.concatenate(self.launches)
            if self.launches
            else np.zeros(0, dtype=np.uint32)
        )
        np.save(str(trace_dir / "trace.npy"), all_words)

        # write_out_trace format: drop zero words, one lowercase 8-hex per line.
        lines = (f"{int(w):08x}" for w in all_words if int(w) != 0)
        (trace_dir / "raw_trace.txt").write_text("\n".join(lines), encoding="utf-8")

        meta = {
            "target": {
                "kernel": self.target.kernel,
                "sub_device": self.target.sub_device,
                "col": self.target.col,
                "row": self.target.row,
                "trace_size_bytes": self.target.trace_size,
            },
            "info": self.info,
            "launches": len(self.launches),
            "total_words": int(all_words.size),
            "nonzero_words": int(np.count_nonzero(all_words)),
            "sanity_checks": self.sanity_checks,
            "sanity_failures": self.sanity_failures,
            "json": None,
        }

        # Try to produce a chrome-trace JSON. parse_trace needs the
        # post-lowering IR (with aiex.npu.write32 ops); aie_compile.py
        # copies that to <kernel>.aie.mlir next to the ELF.
        lowered_ir = out_dir / f"{self.target.kernel}.aie.mlir"
        if lowered_ir.exists():
            try:
                json_path = try_parse_trace_to_json(
                    trace_dir / "trace.npy",
                    lowered_ir,
                    trace_dir / "trace.json",
                )
                if json_path is not None:
                    meta["json"] = str(json_path)
            except Exception as e:
                print(f"  [trace] parse_trace failed: {e}")

        (trace_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def parse_trace_spec(spec: str) -> TraceTarget:
    """Parse ``KERNEL:SUB_DEVICE:COL:ROW`` (optional ``:TRACE_SIZE``)."""
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(
            f"trace target must be KERNEL:SUB_DEVICE:COL:ROW[:TRACE_SIZE], "
            f"got {spec!r}"
        )
    kernel, sub_device, col_s, row_s = parts[:4]
    trace_size = 8 * 1024 * 1024
    if len(parts) == 5:
        trace_size = int(parts[4])
    return TraceTarget(
        kernel=kernel,
        sub_device=sub_device,
        col=int(col_s),
        row=int(row_s),
        trace_size=trace_size,
    )


def try_parse_trace_to_json(
    raw_trace_npy: Path,
    lowered_ir: Path,
    out_json: Optional[Path] = None,
) -> Optional[Path]:
    """Call mlir-aie's ``parse_trace`` to convert raw uint32 trace words to a
    chrome://tracing JSON list. Returns the JSON path on success, None if
    parsing fails or the parser isn't available.

    ``lowered_ir`` must be the post-aiecc form with ``aiex.npu.write32`` ops
    (i.e. aiecc's ``input_with_addresses.mlir``, which ``aie_compile.py``
    copies to ``<kernel>.aie.mlir`` next to the ELF). Feeding the pre-lowering
    IR with high-level ``aie.trace.*`` ops makes ``parse_trace`` find zero
    tiles and exit.

    Runs in a subprocess for the same reason ``instrument_ir_for_trace`` does:
    parent-process mlir-air pybind state from the IR-gen step trips an MLIR
    ``Operation::create`` assertion when ``parse_trace`` calls ``Module.parse``.
    ``parse_trace`` also calls ``sys.exit(1)`` on its own error paths, which a
    subprocess turns into a clean nonzero return code.
    """
    import os
    import subprocess
    import sys

    if out_json is None:
        out_json = Path(raw_trace_npy).with_suffix(".json")

    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        pkg_parent
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    cmd = [
        sys.executable, "-m", "kernel_builder.aie_trace_capture",
        "--npy", str(raw_trace_npy),
        "--mlir", str(lowered_ir),
        "--out", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        print(
            f"  [trace] parse_trace subprocess returned {proc.returncode}: "
            f"{'; '.join(msg)}"
        )
        return None
    return Path(out_json)


def _cli(argv):
    """CLI entry point for the parse subprocess (see try_parse_trace_to_json)."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True)
    p.add_argument("--mlir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    from aie.utils.trace import parse_trace
    words = np.load(args.npy)
    mlir_text = Path(args.mlir).read_text()
    trace_events = parse_trace(words, mlir_text)
    Path(args.out).write_text(json.dumps(trace_events))


if __name__ == "__main__":
    import sys
    _cli(sys.argv[1:])
