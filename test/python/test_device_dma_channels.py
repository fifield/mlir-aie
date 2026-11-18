# test_device_dma_channels.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""A device's DMA channel budget comes from the target model's DMA channel
counts, whether or not its shim tiles have a shim mux."""

import pytest

from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.dialects.aie import AIEDevice, get_target_model
from aie.iron.device import from_name, NPU3, NPU3Col1


@pytest.mark.parametrize(
    "name, n_cols, shim_in, shim_out, core_in, core_out",
    [
        ("npu1", 1, 2, 2, 2, 2),
        ("npu1", None, 8, 8, 2, 2),
        ("npu2", 1, 2, 2, 2, 2),
        ("npu2", 4, 8, 8, 2, 2),
        ("npu2", None, 16, 16, 2, 2),
    ],
)
def test_dma_channel_budget(name, n_cols, shim_in, shim_out, core_in, core_out):
    device = from_name(name, n_cols=n_cols)
    assert device.shim_dma_channels_in == shim_in
    assert device.shim_dma_channels_out == shim_out
    assert device.core_dma_channels_in == core_in
    assert device.core_dma_channels_out == core_out


def test_target_model_dma_channels_follow_the_direction():
    tm = get_target_model(AIEDevice.npu2)
    assert tm.get_num_dma_channels(0, 0, DMAChannelDir.MM2S) == 2
    assert tm.get_num_dma_channels(0, 1, DMAChannelDir.MM2S) == 6
    assert tm.get_num_dma_channels(0, 1, DMAChannelDir.S2MM) == 6


# npu3 has no shim mux; shim DMA channels (4 MM2S, 2 S2MM) are switchbox ports.
@pytest.mark.parametrize(
    "device, cols",
    [(NPU3(), 3), (NPU3Col1(), 1)],
)
def test_npu3_dma_channel_budget(device, cols):
    assert device.shim_dma_channels_in == 4 * cols
    assert device.shim_dma_channels_out == 2 * cols
    assert device.core_dma_channels_in == 2
    assert device.core_dma_channels_out == 1
