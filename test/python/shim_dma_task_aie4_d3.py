# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Exercise the Python helper and task lowering without requiring a simulator.
# Five entries on AIE4 mean [iteration, d3, d2, d1, d0]. The non-contiguous
# access pattern keeps D3 active rather than folding into a linear transfer.
from aie.dialects.aie import *
from aie.dialects.aiex import runtime_sequence, shim_dma_single_bd_task
from aie.extras.context import mlir_mod_ctx
from aie.helpers.taplib import TensorAccessPattern
from aie.passmanager import PassManager


def build(
    dev,
    use_tap=False,
    sizes=None,
    strides=None,
    default_strides=False,
    expected_sizes=None,
    expected_length=48,
    expected_repeat=0,
):
    sizes = [1, 2, 2, 3, 4] if sizes is None else sizes
    strides = [0, 30, 12, 4, 1] if strides is None else strides
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def body():
            shim = tile(0, 0)
            shim_dma_allocation("input", shim, DMAChannelDir.MM2S, 0)

            @runtime_sequence(T.memref(64, T.i32()))
            def sequence(arg):
                if use_tap:
                    tap = TensorAccessPattern((64,), 0, sizes, strides)
                    shim_dma_single_bd_task("input", arg, tap=tap)
                else:
                    shim_dma_single_bd_task(
                        "input",
                        arg,
                        sizes=sizes,
                        strides=None if default_strides else strides,
                    )

        assert ctx.module.operation.verify()
        sequence_op = next(
            op
            for op in ctx.module.body.operations[0].body_region.blocks[0].operations
            if isinstance(op, RuntimeSequenceOp)
        )
        task = sequence_op.body.blocks[0].operations[0]
        dma_bd_op = task.body.blocks[0].operations[0]
        expected_sizes = sizes if expected_sizes is None else expected_sizes
        assert list(dma_bd_op.static_sizes) == expected_sizes
        expected_strides = [0] * (len(sizes) - 1) + [1] if default_strides else strides
        expected_strides = [0] * (len(expected_sizes) - len(sizes)) + expected_strides
        assert list(dma_bd_op.static_strides) == expected_strides
        assert dma_bd_op.static_len.value == expected_length
        assert task.repeat_count.value == expected_repeat
        print(ctx.module)
        PassManager.parse(
            "builtin.module(aie.device(aie-normalize-dma-bd-dims,"
            "aie-substitute-shim-dma-allocations,"
            "aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))"
        ).run(ctx.module.operation)
        assert ctx.module.operation.verify()
        writebd = sequence_op.body.blocks[0].operations[0]
        assert writebd.buffer_length.value == expected_length
        if len(expected_sizes) == 5 and not default_strides:
            assert writebd.d2_size.value == expected_sizes[2]
            assert writebd.d3_size.value == expected_sizes[1]
            assert writebd.d3_stride.value == (
                strides[1] if expected_sizes[1] > 1 else 0
            )
        else:
            assert writebd.d3_size.value == 0
            assert writebd.d3_stride.value == 0
        print(ctx.module)


# CHECK: len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]
# CHECK: buffer_length = 48 : i32
# CHECK-SAME: d2_size = 2 : i32
# CHECK-SAME: d3_size = 2 : i32, d3_stride = 30 : i32
for dev in (AIEDevice.npu3, AIEDevice.npu3_1col, AIEDevice.npu3_2col):
    build(dev)
build(AIEDevice.npu3, use_tap=True)

# Iteration remains independent of all four addressing dimensions.
build(AIEDevice.npu3, sizes=[3, 2, 2, 3, 4], expected_repeat=2)

# Singleton addressing entries retain their positions, including D3. Removing
# any of them would reinterpret the remaining outer entry as iteration.
for address_dim in (1, 2, 3, 4):
    singleton_sizes = [1, 2, 2, 3, 4]
    expected_length = 48 // singleton_sizes[address_dim]
    singleton_sizes[address_dim] = 1
    build(
        AIEDevice.npu3,
        sizes=singleton_sizes,
        strides=[0, 30, 16, 4, 1] if address_dim == 1 else [0, 30, 12, 4, 1],
        expected_length=expected_length,
    )

# CHECK: len = 64 sizes = [1, 1, 1, 1, 64] strides = [0, 0, 0, 0, 1]
build(
    AIEDevice.npu3,
    sizes=[1, 1, 1, 1, 64],
    default_strides=True,
    expected_length=64,
)

# Four entries retain their iteration meaning even on AIE4.
# CHECK: len = 8 sizes = [3, 1, 1, 8] strides = [0, 0, 0, 1]
# CHECK: repeat_count = 2 : i32
build(
    AIEDevice.npu3,
    sizes=[3, 1, 1, 8],
    strides=[0, 0, 0, 1],
    expected_length=8,
    expected_repeat=2,
)

# Short patterns are padded; their leading size must not become a repeat.
# CHECK: len = 16 sizes = [1, 1, 2, 8] strides = [0, 0, 8, 1]
build(
    AIEDevice.npu3,
    sizes=[2, 8],
    strides=[8, 1],
    expected_sizes=[1, 1, 2, 8],
    expected_length=16,
)


def reject(dev, sizes, strides):
    try:
        build(dev, sizes=sizes, strides=strides)
    except ValueError as error:
        print("REJECTED:", error)
    else:
        raise AssertionError(f"invalid pattern accepted on {dev}")


# Before AIE4 a fifth entry is another iteration dimension, which
# aie-decompose-large-dma-bd splits off, so only AIE4 bounds the rank here.
# CHECK: REJECTED: AIE4 shim DMA BD supports at most 5 dimensions, got 6
reject(AIEDevice.npu3, [1, 1, 2, 2, 3, 4], [0, 0, 30, 12, 4, 1])
