# test_npu3_python_hooks.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""npu3 (AIE4) in the device, compile, host-runtime and trace helpers.

None of this needs AIE4 hardware or an AIE4 Peano.
"""

import argparse

import pytest

from aie.dialects._aie_enum_gen import (
    AIEArch,
    CoreEventAIE4,
    MemTileEventAIE4,
    ShimTileEventAIE4,
)
from aie.dialects.aie import AIEDevice
from aie.iron.device import NPU3, NPU3Col1, NPU3Col2, from_name
from aie.utils.compile import resolve_target_arch
from aie.utils.hostruntime import cli, set_current_device
from aie.utils.hostruntime.argparse import DEFAULT_DEV_CHOICES, add_compile_args
from aie.utils.trace.events import PacketType, get_events_for_device
from aie.utils.trace.parse import lookup_event_name_by_type


@pytest.mark.parametrize(
    "n_cols, cls", [(1, NPU3Col1), (2, NPU3Col2), (3, NPU3), (None, NPU3)]
)
def test_from_name_npu3(n_cols, cls):
    device = from_name("npu3", n_cols=n_cols)
    assert type(device) is cls
    assert device.arch is AIEArch.AIE4


@pytest.mark.parametrize("n_cols", [0, 4])
def test_from_name_npu3_column_range(n_cols):
    with pytest.raises(ValueError, match=r"1\.\.3"):
        from_name("npu3", n_cols=n_cols)


@pytest.mark.parametrize(
    "device, arch",
    [
        (NPU3(), "aie4"),
        (NPU3Col1(), "aie4"),
        (AIEDevice.npu3_2col, "aie4"),
        (from_name("npu1"), "aie2"),
        (from_name("npu2"), "aie2p"),
        (None, "aie2"),
    ],
)
def test_resolve_target_arch(device, arch):
    assert resolve_target_arch(device) == arch


@pytest.mark.parametrize(
    "device, name",
    [(NPU3Col1(), "npu3"), (from_name("npu1"), "npu"), (from_name("npu2"), "npu2")],
)
def test_runtime_device_name(device, name):
    assert cli._runtime_device_name(device) == name


def test_dev_choices_accept_npu3():
    assert "npu3" in DEFAULT_DEV_CHOICES
    parser = argparse.ArgumentParser()
    add_compile_args(parser, default_dev="npu2")
    assert parser.parse_args(["--dev", "npu3"]).dev == "npu3"


def test_npu3_trace_events():
    events = get_events_for_device("npu3_1col")
    assert events.CoreEvent is CoreEventAIE4
    assert events.MemTileEvent is MemTileEventAIE4
    assert events.ShimTileEvent is ShimTileEventAIE4
    # No memory module: compute-tile DMA and lock events are core events.
    assert events.MemEvent is None
    assert events.CoreEvent.TRUE == 1
    assert lookup_event_name_by_type(PacketType.MEM, 1, events) == "Unknown"
    assert lookup_event_name_by_type(PacketType.CORE, 1, events) == "TRUE"


def test_iron_kernels_refuse_npu3():
    # aie.iron.kernels has no AIE4 builds; a bound npu3 says so up front.
    from aie.iron.kernels._common import _bound_arch

    with set_current_device(NPU3Col1()):
        with pytest.raises(RuntimeError, match="no kernels for aie4"):
            _bound_arch()
    with set_current_device(from_name("npu2")):
        assert _bound_arch() == "aie2p"
