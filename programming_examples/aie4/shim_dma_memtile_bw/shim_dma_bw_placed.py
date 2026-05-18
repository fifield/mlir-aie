# shim_dma_memtile_bw/shim_dma_bw_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

from shim_dma_bw_config import MAX_CHANNELS, build_config, parse_size


def size_arg(value: str) -> int:
    try:
        return parse_size(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def create_memtile_sink(mem_tile, channels, ping_buffers, pong_buffers):
    data_block = lambda i: 1 + 2 * i
    start_block = lambda i: 1 + 2 * len(channels) + (i - 1)
    end_block = 2 * len(channels) + len(channels)

    @memtile_dma(mem_tile)
    def sink_dma(block):
        def next_start_or_end(local_idx):
            if local_idx + 1 < len(channels):
                return block[start_block(local_idx + 1)]
            return block[end_block]

        first = channels[0]
        dma_start(
            DMAChannelDir.S2MM,
            first.mem_channel,
            dest=block[data_block(0)],
            chain=next_start_or_end(0),
        )

        for local_idx, channel in enumerate(channels[1:], start=1):
            with block[start_block(local_idx)]:
                dma_start(
                    DMAChannelDir.S2MM,
                    channel.mem_channel,
                    dest=block[data_block(local_idx)],
                    chain=next_start_or_end(local_idx),
                )

        for local_idx, channel in enumerate(channels):
            first_bd = data_block(local_idx)
            second_bd = first_bd + 1
            with block[first_bd]:
                dma_bd(ping_buffers[channel.shim_channel])
                next_bd(block[second_bd])
            with block[second_bd]:
                dma_bd(pong_buffers[channel.shim_channel])
                next_bd(block[first_bd])

        with block[end_block]:
            EndOp()


def emit_benchmark(opts):
    cfg = build_config(
        channels=opts.channels,
        bytes_per_channel=opts.bytes_per_channel,
        sink_bytes=opts.sink_bytes,
        iterations=opts.iterations,
        pattern=opts.pattern,
        channel_order_name=opts.channel_order,
        stride_elements=opts.stride_elements,
        block_bytes=opts.block_bytes,
        gap_bytes=opts.gap_bytes,
    )

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu3)
        def device_body():
            host_ty = np.ndarray[(cfg.buffer_elems,), np.dtype[np.uint32]]
            sink_ty = np.ndarray[(cfg.sink_elems,), np.dtype[np.uint32]]

            shim_tiles = [tile(col, 0) for col in range(3)]
            mem_tiles = [tile(col, 1) for col in range(3)]

            for col, shim_tile in enumerate(shim_tiles):
                shim_tile.attributes["controller_id"] = AttrBuilder.get(
                    "PacketInfoAttr"
                )([0, col])

            controller = shim_tiles[0]
            for col, shim_tile in enumerate(shim_tiles):
                for ctrl_ch in (0, 1):
                    packetflow(
                        col,
                        shim_tile,
                        WireBundle.Control32,
                        ctrl_ch,
                        [
                            {
                                "dest": controller,
                                "port": WireBundle.Controller32,
                                "channel": 0,
                            }
                        ],
                    )

            channels_by_col = {col: [] for col in range(3)}
            for channel in cfg.channels:
                channels_by_col[channel.col].append(channel)
                flow(
                    shim_tiles[channel.col],
                    WireBundle.DMA,
                    channel.shim_channel,
                    mem_tiles[channel.col],
                    WireBundle.DMA,
                    channel.mem_channel,
                )

            for col in range(3):
                if not channels_by_col[col]:
                    continue
                ping_buffers = [
                    buffer(
                        mem_tiles[col],
                        sink_ty,
                        name=f"sink_c{col}_ch{shim_channel}_ping",
                    )
                    for shim_channel in range(4)
                ]
                pong_buffers = [
                    buffer(
                        mem_tiles[col],
                        sink_ty,
                        name=f"sink_c{col}_ch{shim_channel}_pong",
                    )
                    for shim_channel in range(4)
                ]
                create_memtile_sink(
                    mem_tiles[col], channels_by_col[col], ping_buffers, pong_buffers
                )

            @runtime_sequence(host_ty)
            def sequence(A):
                tasks = []
                for channel, access in zip(cfg.channels, cfg.accesses):
                    task = dma_configure_task(
                        shim_tiles[channel.col],
                        DMAChannelDir.MM2S,
                        channel.shim_channel,
                        repeat_count=cfg.iterations - 1,
                        issue_token=True,
                    )
                    with bds(task) as bd:
                        with bd[0]:
                            shim_dma_bd(
                                A,
                                offset=access.offset_elems,
                                sizes=access.sizes,
                                strides=access.strides,
                                transfer_len=access.transfer_elems,
                                burst_length=opts.burst_length,
                            )
                            EndOp()
                    tasks.append(task)

                dma_start_task(*tasks)
                dma_await_task(*tasks)

        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify, file=sys.stderr)
            return 1
        print(ctx.module)
    return 0


def main(argv):
    parser = argparse.ArgumentParser(
        description="Placed AIE4 shim-DMA-to-memtile memory bandwidth benchmark"
    )
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
    parser.add_argument("--burst-length", type=int, default=0)

    opts = parser.parse_args(argv)
    try:
        return emit_benchmark(opts)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
