# test_target_model_addresses.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""The target model's tile address map, as the Python bindings expose it."""

import pytest

from aie.dialects.aie import AIEDevice, get_target_model

MIB = 0x100000


def _tiles(tm):
    return [(c, r) for c in range(tm.columns()) for r in range(tm.rows())]


@pytest.mark.parametrize("device", [AIEDevice.npu1, AIEDevice.npu2, AIEDevice.xcvc1902])
def test_one_address_row_per_tile(device):
    tm = get_target_model(device)
    for col, row in _tiles(tm):
        address = tm.get_tile_address(col, row)
        assert address == (col << tm.get_column_shift()) | (row << tm.get_row_shift())
        assert tm.get_column_from_address(address + 0x1D004) == col
        assert tm.get_row_from_address(address + 0x1D004) == row
        assert tm.get_data_memory_offset(col, row) == 0


@pytest.mark.parametrize(
    "device", [AIEDevice.npu3, AIEDevice.npu3_1col, AIEDevice.npu3_2col]
)
def test_aie4_address_map(device):
    # A memtile spans five 1 MiB address rows and its data memory starts
    # 1 MiB in; core row r sits at address row r + 4.
    tm = get_target_model(device)
    for col in range(tm.columns()):
        column = col << 25
        assert tm.get_tile_address(col, 0) == column
        assert tm.get_tile_address(col, 1) == column | MIB
        assert tm.get_data_memory_offset(col, 1) == MIB
        for row in range(2, tm.rows()):
            assert tm.get_tile_address(col, row) == column | (row + 4) * MIB
            assert tm.get_data_memory_offset(col, row) == 0
        for offset in (0, MIB, 4 * MIB + 0xFFFFC):
            address = tm.get_tile_address(col, 1) + offset
            assert tm.get_row_from_address(address) == 1
            assert tm.get_column_from_address(address) == col
        assert tm.get_row_from_address(tm.get_tile_address(col, 2)) == 2
