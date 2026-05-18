#!/usr/bin/env python3
# shim_dma_memtile_bw/host.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from shim_dma_bw_config import MAX_CHANNELS, build_config, parse_size

try:
    import pyxrt as xrt
except ImportError:
    print("Error: pyxrt not found. Please install XRT Python bindings.", file=sys.stderr)
    sys.exit(1)


def size_arg(value: str) -> int:
    try:
        return parse_size(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def completed(state) -> bool:
    if not hasattr(xrt, "ert_cmd_state"):
        return True
    return state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


def fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    scaled = float(value)
    for unit in units:
        if scaled < 1024.0 or unit == units[-1]:
            return f"{scaled:.2f} {unit}"
        scaled /= 1024.0
    return f"{value} B"


def run_kernel(kernel, bo, sync_each_sample: bool) -> int:
    if sync_each_sample:
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    start = time.perf_counter_ns()
    run = kernel(bo)
    state = run.wait()
    stop = time.perf_counter_ns()
    if not completed(state):
        raise RuntimeError(f"kernel returned {state}")
    return stop - start


def main(argv):
    parser = argparse.ArgumentParser(
        description="PyXRT host for the AIE4 shim DMA memtile bandwidth benchmark"
    )
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--kernel", default="main:sequence")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--channels", type=int, default=MAX_CHANNELS)
    parser.add_argument("--channel-order", choices=["roundrobin", "packed"], default="roundrobin")
    parser.add_argument("--bytes-per-channel", type=size_arg, default=parse_size("16MiB"))
    parser.add_argument("--sink-bytes", type=size_arg, default=parse_size("4KiB"))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--pattern",
        choices=["linear", "strided", "block-strided", "interleaved"],
        default="linear",
    )
    parser.add_argument("--stride-elements", type=int, default=16)
    parser.add_argument("--block-bytes", type=size_arg, default=parse_size("4KiB"))
    parser.add_argument("--gap-bytes", type=size_arg, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sync-each-sample", action="store_true")
    parser.add_argument("--bo-kind", choices=["ext", "host-only"], default="ext")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.samples < 1:
        raise ValueError("samples must be >= 1")
    if args.warmups < 0:
        raise ValueError("warmups must be >= 0")

    cfg = build_config(
        channels=args.channels,
        bytes_per_channel=args.bytes_per_channel,
        sink_bytes=args.sink_bytes,
        iterations=args.iterations,
        pattern=args.pattern,
        channel_order_name=args.channel_order,
        stride_elements=args.stride_elements,
        block_bytes=args.block_bytes,
        gap_bytes=args.gap_bytes,
    )

    if not args.elf.exists():
        raise FileNotFoundError(args.elf)

    device = xrt.device(args.device_index)
    elf = xrt.elf(str(args.elf))
    context = xrt.hw_context(device, elf)
    kernel = xrt.ext.kernel(context, args.kernel)

    if args.bo_kind == "ext":
        bo = xrt.ext.bo(device, cfg.buffer_bytes)
    else:
        bo = xrt.bo(device, cfg.buffer_bytes, xrt.bo.host_only, 0)
    mapped = bo.map()
    data = np.frombuffer(mapped, dtype=np.uint32, count=cfg.buffer_elems)
    data[:] = np.arange(cfg.buffer_elems, dtype=np.uint32)
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    if args.verbose:
        print("channels:")
        for channel in cfg.channels:
            print(
                f"  logical {channel.index}: col {channel.col}, "
                f"shim MM2S {channel.shim_channel}, mem S2MM {channel.mem_channel}"
            )

    print(f"ELF: {args.elf}")
    print(f"kernel: {args.kernel}")
    print(
        "config: "
        f"{cfg.active_channels} channels, {cfg.pattern}, "
        f"{fmt_bytes(cfg.bytes_per_channel)} per channel x {cfg.iterations} iterations"
    )
    print(
        f"host buffer span: {fmt_bytes(cfg.buffer_bytes)}, "
        f"transferred per run: {fmt_bytes(cfg.bytes_per_kernel)}"
    )

    for _ in range(args.warmups):
        run_kernel(kernel, bo, args.sync_each_sample)

    times_ns = [
        run_kernel(kernel, bo, args.sync_each_sample) for _ in range(args.samples)
    ]
    best_ns = min(times_ns)
    avg_ns = int(statistics.mean(times_ns))
    bytes_per_run = cfg.bytes_per_kernel

    best_gbs = bytes_per_run / (best_ns / 1e9) / 1e9
    avg_gbs = bytes_per_run / (avg_ns / 1e9) / 1e9

    print(f"best: {best_ns / 1000.0:.0f} us, {best_gbs:.2f} GB/s")
    print(f"avg:  {avg_ns / 1000.0:.0f} us, {avg_gbs:.2f} GB/s")
    if args.samples > 1:
        print("samples_us: " + ", ".join(f"{t / 1000.0:.0f}" for t in times_ns))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
