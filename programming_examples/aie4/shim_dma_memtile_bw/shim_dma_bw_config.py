# shim_dma_memtile_bw/shim_dma_bw_config.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

from dataclasses import dataclass


DTYPE_BYTES = 4
NPU3_COLUMNS = 3
SHIM_MM2S_CHANNELS_PER_COLUMN = 4
MEMTILE_S2MM_CHANNELS = [0, 1, 2, 4]
MAX_CHANNELS = NPU3_COLUMNS * SHIM_MM2S_CHANNELS_PER_COLUMN


@dataclass(frozen=True)
class Channel:
    index: int
    col: int
    shim_channel: int
    mem_channel: int


@dataclass(frozen=True)
class Access:
    offset_elems: int
    sizes: list[int]
    strides: list[int]
    transfer_elems: int


@dataclass(frozen=True)
class BenchmarkConfig:
    channels: list[Channel]
    accesses: list[Access]
    buffer_elems: int
    bytes_per_channel: int
    sink_elems: int
    iterations: int
    pattern: str

    @property
    def active_channels(self) -> int:
        return len(self.channels)

    @property
    def buffer_bytes(self) -> int:
        return self.buffer_elems * DTYPE_BYTES

    @property
    def bytes_per_kernel(self) -> int:
        return self.bytes_per_channel * self.active_channels * self.iterations


def parse_size(value: str) -> int:
    value = str(value).strip()
    if not value:
        raise ValueError("empty size")

    suffixes = {
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
    }

    lower = value.lower()
    for suffix, multiplier in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if lower.endswith(suffix):
            number = lower[: -len(suffix)]
            return int(number, 0) * multiplier
    return int(lower, 0)


def channel_order(num_channels: int, order: str) -> list[Channel]:
    if order == "roundrobin":
        ordered = [
            (col, shim_channel)
            for shim_channel in range(SHIM_MM2S_CHANNELS_PER_COLUMN)
            for col in range(NPU3_COLUMNS)
        ]
    elif order == "packed":
        ordered = [
            (col, shim_channel)
            for col in range(NPU3_COLUMNS)
            for shim_channel in range(SHIM_MM2S_CHANNELS_PER_COLUMN)
        ]
    else:
        raise ValueError(f"unknown channel order: {order}")

    return [
        Channel(
            index=i,
            col=col,
            shim_channel=shim_channel,
            mem_channel=MEMTILE_S2MM_CHANNELS[shim_channel],
        )
        for i, (col, shim_channel) in enumerate(ordered[:num_channels])
    ]


def _validate_common(
    channels: int, bytes_per_channel: int, sink_bytes: int, iterations: int
) -> None:
    if channels < 1 or channels > MAX_CHANNELS:
        raise ValueError(f"channels must be in [1, {MAX_CHANNELS}], got {channels}")
    if bytes_per_channel <= 0 or bytes_per_channel % DTYPE_BYTES:
        raise ValueError("bytes_per_channel must be positive and 4-byte aligned")
    if sink_bytes <= 0 or sink_bytes % DTYPE_BYTES:
        raise ValueError("sink_bytes must be positive and 4-byte aligned")
    if bytes_per_channel % sink_bytes:
        raise ValueError("bytes_per_channel must be a multiple of sink_bytes")
    if iterations < 1 or iterations > 64:
        raise ValueError("iterations must be in [1, 64]")


def build_config(
    channels: int,
    bytes_per_channel: int,
    sink_bytes: int,
    iterations: int,
    pattern: str,
    channel_order_name: str = "roundrobin",
    stride_elements: int = 16,
    block_bytes: int = 4096,
    gap_bytes: int = 0,
) -> BenchmarkConfig:
    _validate_common(channels, bytes_per_channel, sink_bytes, iterations)

    transfer_elems = bytes_per_channel // DTYPE_BYTES
    sink_elems = sink_bytes // DTYPE_BYTES
    active_channels = channel_order(channels, channel_order_name)
    accesses: list[Access] = []

    if pattern == "linear":
        span_elems = transfer_elems
        buffer_elems = span_elems * channels
        for ch in active_channels:
            accesses.append(
                Access(
                    offset_elems=ch.index * span_elems,
                    sizes=[1, 1, 1, transfer_elems],
                    strides=[0, 0, 0, 1],
                    transfer_elems=transfer_elems,
                )
            )
    elif pattern == "strided":
        if stride_elements < 1:
            raise ValueError("stride_elements must be >= 1")
        span_elems = (transfer_elems - 1) * stride_elements + 1
        buffer_elems = span_elems * channels
        for ch in active_channels:
            accesses.append(
                Access(
                    offset_elems=ch.index * span_elems,
                    sizes=[1, 1, 1, transfer_elems],
                    strides=[0, 0, 0, stride_elements],
                    transfer_elems=transfer_elems,
                )
            )
    elif pattern == "block-strided":
        if block_bytes <= 0 or block_bytes % DTYPE_BYTES:
            raise ValueError("block_bytes must be positive and 4-byte aligned")
        if gap_bytes < 0 or gap_bytes % DTYPE_BYTES:
            raise ValueError("gap_bytes must be non-negative and 4-byte aligned")
        if bytes_per_channel % block_bytes:
            raise ValueError("bytes_per_channel must be a multiple of block_bytes")
        block_elems = block_bytes // DTYPE_BYTES
        gap_elems = gap_bytes // DTYPE_BYTES
        blocks = transfer_elems // block_elems
        block_stride = block_elems + gap_elems
        span_elems = (blocks - 1) * block_stride + block_elems
        buffer_elems = span_elems * channels
        for ch in active_channels:
            accesses.append(
                Access(
                    offset_elems=ch.index * span_elems,
                    sizes=[1, 1, blocks, block_elems],
                    strides=[0, 0, block_stride, 1],
                    transfer_elems=transfer_elems,
                )
            )
    elif pattern == "interleaved":
        if block_bytes <= 0 or block_bytes % DTYPE_BYTES:
            raise ValueError("block_bytes must be positive and 4-byte aligned")
        if gap_bytes < 0 or gap_bytes % DTYPE_BYTES:
            raise ValueError("gap_bytes must be non-negative and 4-byte aligned")
        if bytes_per_channel % block_bytes:
            raise ValueError("bytes_per_channel must be a multiple of block_bytes")
        block_elems = block_bytes // DTYPE_BYTES
        gap_elems = gap_bytes // DTYPE_BYTES
        blocks = transfer_elems // block_elems
        round_stride = channels * block_elems + gap_elems
        buffer_elems = (blocks - 1) * round_stride + channels * block_elems
        for ch in active_channels:
            accesses.append(
                Access(
                    offset_elems=ch.index * block_elems,
                    sizes=[1, 1, blocks, block_elems],
                    strides=[0, 0, round_stride, 1],
                    transfer_elems=transfer_elems,
                )
            )
    else:
        raise ValueError(
            "pattern must be one of: linear, strided, block-strided, interleaved"
        )

    return BenchmarkConfig(
        channels=active_channels,
        accesses=accesses,
        buffer_elems=buffer_elems,
        bytes_per_channel=bytes_per_channel,
        sink_elems=sink_elems,
        iterations=iterations,
        pattern=pattern,
    )
