# Copyright (C) 2022 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from contextlib import contextmanager
import itertools
from operator import itemgetter

import numpy as np

from ._aiex_ops_gen import *
from ._aiex_ops_gen import (
    npu_write32 as _npu_write32,
    npu_maskwrite32 as _npu_maskwrite32,
    npu_sync as _npu_sync,
    npu_address_patch as _npu_address_patch,
    npu_rtp_write as _npu_rtp_write,
    npu_push_queue as _npu_push_queue,
)
from ._aie_ops_gen import ObjectFifoCreateOp, EndOp, RuntimeSequenceOp
from . import aie
from .aie import (
    BdIteration,
    DMAChannelDir,
    LockAction,
    Neighbors,
    TileOp,
    bds,
    dma_bd,
    _as_i32,
)
from .transform.structured import MixedValues, _dispatch_mixed_values
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *
from ..helpers.npdtypes import v8bfp16ebs8, v16bfp16ebs16
from ..ir import (
    DictAttr,
    IntegerAttr,
    IntegerType,
    UnitAttr,
    Type,
    InsertionPoint,
    Attribute,
    AttrBuilder,
    Value,
)

# noinspection PyUnresolvedReferences
from ..extras import types as T
from ..extras.util import find_parent_of_type
from ..helpers.util import try_convert_np_type_to_mlir_type
from ..helpers.taplib import TensorAccessPattern
from ..helpers.taplib._symbolic import is_sym, si32, sprod
from ..helpers.taplib.utils import zero_leading_unit_strides

# Comes from _aie
register_dialect(get_dialect_registry())


def shim_dims(tap: TensorAccessPattern, ndims: int = 4):
    """Return `tap`'s offset, sizes and strides as a shim buffer descriptor takes them.

    A walk deeper than `ndims` is coalesced, and one that still does not fit
    passes through for the compiler to split into BDs of `ndims` dimensions.
    A shallower one is padded with unit dimensions. Slot 0 is the queue
    repeat: a leading stride-0 dimension whose size is not the constant 1,
    staged or not, stays there and the padding goes after it, so
    `[R, th, tw]` becomes `[R, 1, th, tw]`.

    Args:
        tap: The access pattern.
        ndims: The dimensions a buffer descriptor takes.

    Returns:
        `(offset, sizes, strides)`, with `ndims` sizes and strides unless the
        walk does not fit.

    Raises:
        ValueError: If `tap` is padded.
    """
    if tap.padding is not None:
        raise ValueError("a shim DMA cannot pad; padding is a memtile feature")
    if tap.rank > ndims:
        tap = tap.coalesce()
    if tap.rank > ndims:
        return tap.offset, list(tap.sizes), list(tap.strides)
    sizes, strides = list(tap.sizes), list(tap.strides)
    is_repeat = (
        tap.rank > 1
        and not is_sym(strides[0])
        and strides[0] == 0
        and (is_sym(sizes[0]) or sizes[0] != 1)
    )
    at = int(is_repeat)
    sizes[at:at] = [1] * (ndims - tap.rank)
    strides[at:at] = [0] * (ndims - tap.rank)
    return tap.offset, sizes, zero_leading_unit_strides(sizes, strides, start=at)


def npu_write32(address, value, buffer=None, column=None, row=None, **kwargs):
    return _npu_write32(
        _as_i32(address),
        _as_i32(value),
        buffer=buffer,
        column=column,
        row=row,
        **kwargs,
    )


def npu_maskwrite32(address, value, mask, buffer=None, column=None, row=None, **kwargs):
    return _npu_maskwrite32(
        _as_i32(address),
        _as_i32(value),
        _as_i32(mask),
        buffer=buffer,
        column=column,
        row=row,
        **kwargs,
    )


def npu_sync(column, row, direction, channel, column_num=1, row_num=1, **kwargs):
    return _npu_sync(
        _as_i32(column),
        _as_i32(row),
        _as_i32(direction),
        _as_i32(channel),
        _as_i32(column_num),
        _as_i32(row_num),
        **kwargs,
    )


def npu_address_patch(addr, arg_idx, arg_plus, **kwargs):
    return _npu_address_patch(addr, _as_i32(arg_plus), arg_idx=arg_idx, **kwargs)


def npu_rtp_write(buffer, index, value, **kwargs):
    return _npu_rtp_write(buffer, index, _as_i32(value), **kwargs)


def npu_push_queue(
    column, row, direction, channel, issue_token, repeat_count, bd_id, **kwargs
):
    return _npu_push_queue(
        column,
        row,
        direction,
        channel,
        issue_token,
        _as_i32(repeat_count),
        _as_i32(bd_id),
        **kwargs,
    )


def dma_wait(*args: ObjectFifoCreateOp | str):
    if len(args) == 0:
        raise ValueError(
            "dma_wait must receive at least one dma_meta information to wait for"
        )
    for dma_meta in args:
        str_name = dma_meta
        if isinstance(dma_meta, ObjectFifoCreateOp):
            str_name = dma_meta.sym_name.value
        npu_dma_wait(str_name)


class NpuDmaMemcpyNd(NpuDmaMemcpyNdOp):
    """
    Enables data transfers between the AIE Engine array and external memory.

    Args:
        metadata: This is a reference to the object FIFO or the string name of an object FIFO that records a Shim Tile and one of its DMA channels allocated for the host-side memory transfer. In order to associate the memcpy operation with an object FIFO, this metadata string needs to match the object FIFO name string.
        bd_id: Identifier integer for the particular Buffer Descriptor control registers used for this memcpy. A buffer descriptor contains all information needed for a DMA transfer described in the parameters below.
        mem: Reference to a host buffer, given as an argument to the sequence function, that this transfer will read from or write to.
        tap (optional): A TensorAccessPattern is an alternative method of specifying offset/sizes/strides for determining an access pattern over the mem buffer.
        offsets (optional): Start points for data transfer in each dimension. There is a maximum of four offset dimensions.
        sizes: The extent of data to be transferred across each dimension. There is a maximum of four size dimensions.
        strides (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data.
        burst_length (optional): The configuration of the burst length for the DMA task. If 0, defaults to the highest available value.
        axcache (optional): The raw 4-bit AxCACHE value for the DMA's AXI-MM transfers. If
            omitted, the target model's default AxCACHE value is used.
        length_parameter (optional): Name of a scratchpad parameter n; the
            transfer then moves the static length plus n * length_unit elements
            (per iteration). Shim tiles only; see the length_parameter docs on
            aie.dma_bd.
        length_unit (optional): Elements added per unit of length_parameter; a
            multiple of 16 bytes. Required with length_parameter.

    Note:
        Contiguous row-major access patterns are automatically folded to canonical linear form
        by the compiler's canonicalization pass. For example, a 2D image access
        ``sizes=[1, 1, height, width], strides=[0, 0, width, 1]`` is equivalent to
        ``sizes=[1, 1, 1, height*width], strides=[0, 0, 0, 1]`` and will be canonicalized
        to the latter. This means the natural multidimensional form can always be used
        without concern for the hardware d0 dimension size limit.

    Example:

        npu_dma_memcpy_nd(of_in, 0, input_buffer, sizes=[1, 1, 1, 30])

        The example above describes a linear transfer of 30 data elements, or 120 Bytes, from the input_buffer in host memory into an object FIFO with matching
        metadata labeled "of_in".
        The size dimensions are expressed right to left where the right is dimension 0 and the left dimension 3. Higher dimensions not used should be set to 1.
    """

    def __init__(
        self,
        metadata: str | ObjectFifoCreateOp,
        bd_id,
        mem,
        tap: TensorAccessPattern | None = None,
        offsets: MixedValues | None = None,
        sizes: MixedValues | None = None,
        strides: MixedValues | None = None,
        issue_token: bool | None = None,
        burst_length: int = 0,
        axcache: int | None = None,
        packet: tuple[int] | None = None,
        offset_parameter: str | None = None,
        length_parameter: str | None = None,
        length_unit: int | None = None,
    ):
        if tap and not (offsets is None and sizes is None and strides is None):
            raise ValueError(
                "NpuDmaMemcpyNd can take either a TileAccessPattern OR (sizes and/or strides and/or offsets), but not both."
            )
        if tap:
            offset, sizes, strides = _shim_tap_dims(tap)
            offsets = [0] * (len(sizes) - 1) + [offset]
        else:
            if offsets is None:
                offsets = [0] * 4
            if sizes is None:
                sizes = [0] * 4
            if strides is None:
                strides = [0] * 3 + [1]
        dynamic_offsets, _packed_offsets, static_offsets = _dispatch_mixed_values(
            offsets
        )
        dynamic_sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
        dynamic_strides, _packed_strides, static_strides = _dispatch_mixed_values(
            strides
        )
        if isinstance(metadata, ObjectFifoCreateOp):
            metadata = metadata.sym_name.value
        super().__init__(
            mem,
            dynamic_offsets,
            dynamic_sizes,
            dynamic_strides,
            static_offsets,
            static_sizes,
            static_strides,
            metadata,
            bd_id,
            issue_token=issue_token,
            burst_length=burst_length,
            axcache=axcache,
            packet=packet,
            offset_parameter=offset_parameter,
            length_parameter=length_parameter,
            length_unit=length_unit,
        )


npu_dma_memcpy_nd = NpuDmaMemcpyNd


# Runtime sequence


def runtime_sequence(*inputs: Type, sym_name=None, context=None):
    def decorator(f):
        name = sym_name if sym_name else f.__name__
        seq_op = RuntimeSequenceOp(sym_name=name)
        my_inputs = []
        for input in inputs:
            my_inputs.append(try_convert_np_type_to_mlir_type(input))
        entry_block = seq_op.body.blocks.append(*my_inputs)
        args = entry_block.arguments
        with InsertionPoint(entry_block):
            f(*args)

    return decorator


_orig_dma_configure_task = dma_configure_task


def dma_configure_task(*args, **kwargs):
    return DMAConfigureTaskOp(T.index(), *args, **kwargs)


_orig_dma_configure_task_for = dma_configure_task_for


def dma_configure_task_for(alloc, *args, **kwargs):
    alloc_sym = alloc if isinstance(alloc, str) else alloc.sym_name.value
    return DMAConfigureTaskForOp(T.index(), alloc_sym, *args, **kwargs)


_orig_dma_start_bd_chain = dma_start_bd_chain


def dma_start_bd_chain(symbol, args, tile, direction, channel, *pyargs, **kwargs):
    chain_sym = symbol if isinstance(symbol, str) else symbol.sym_name.value
    return DMAStartBdChainOp(
        T.index(), chain_sym, args, tile, direction, channel, *pyargs, **kwargs
    )


_orig_dma_start_bd_chain_for = dma_start_bd_chain_for


def dma_start_bd_chain_for(symbol, args, alloc, *pyargs, **kwargs):
    chain_sym = symbol if isinstance(symbol, str) else symbol.sym_name.value
    alloc_sym = alloc if isinstance(alloc, str) else alloc.sym_name.value
    return DMAStartBdChainForOp(
        T.index(), chain_sym, args, alloc_sym, *pyargs, **kwargs
    )


def _is_aie4_target() -> bool:
    dev = find_parent_of_type(lambda op: isinstance(op.opview, aie.DeviceOp))
    arch = aie.get_target_model(dev.opview.device.value).get_target_arch()
    return arch == aie.AIEArch.AIE4


def _shim_has_d3(sizes) -> bool:
    """True when a 5-dim shim access pattern is [iteration, d3, d2, d1, d0].

    Only AIE4 shim BDs have a D3 addressing dimension; elsewhere every
    dimension before the last three iterates.
    """
    if sizes is None or len(sizes) != 5:
        return False
    return _is_aie4_target()


def _shim_tap_dims(tap: TensorAccessPattern):
    """`shim_dims` for the target: an AIE4 shim keeps a five-dimension walk
    as [iteration, d3, d2, d1, d0]; shallower ones keep the 4-entry form."""
    ndims = 5 if tap.rank >= 5 and _is_aie4_target() else 4
    return shim_dims(tap, ndims)


def shim_dma_bd(
    mem,
    tap: TensorAccessPattern | None = None,
    offset: int | None = None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len: int | None = None,
    burst_length: int = 0,
    axcache: int | None = None,
    packet: tuple[int] | None = None,
    out_of_order_id: int | None = None,
    offset_parameter: str | None = None,
    length_parameter: str | None = None,
    length_unit: int | None = None,
    iteration: BdIteration | None = None,
):
    if tap and not (offset is None and sizes is None and strides is None):
        raise ValueError(
            "shim_dma_bd can take either a TensorAccessPattern OR (sizes and/or strides and/or offsets), but not both."
        )

    if tap:
        offset, sizes, strides = _shim_tap_dims(tap)

    if offset is None:
        offset = 0
    if sizes is None:
        sizes = [0] * 4
    if strides is None:
        strides = [0] * (len(sizes) - 1) + [1]

    # The per-iteration transfer length is the product of the addressing
    # dimensions. For the historical <=4-dim access patterns that is the lowest
    # three dimensions [d2, d1, d0]; on AIE4 a 5-dim pattern additionally has a
    # real D3 addressing dimension that must be included.
    addr_sizes = sizes[1:] if _shim_has_d3(sizes) else sizes[-3:]
    if not any(is_sym(s) for s in addr_sizes):
        if transfer_len is None:
            transfer_len = sprod(addr_sizes)
        if length_parameter is not None and length_unit is None:
            length_unit = sprod(addr_sizes)

    dma_bd(
        mem,
        sizes=sizes,
        strides=strides,
        offset=offset,
        transfer_len=transfer_len,
        burst_length=burst_length,
        axcache=axcache,
        packet=packet,
        out_of_order_id=out_of_order_id,
        offset_parameter=offset_parameter,
        length_parameter=length_parameter,
        length_unit=length_unit,
        iteration=iteration,
    )


def shim_dma_single_bd_task(
    alloc,
    mem,
    tap: TensorAccessPattern | None = None,
    offset: int | None = None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len: int | None = None,
    issue_token: bool = False,
    burst_length: int = 0,
    axcache: int | None = None,
    packet: tuple[int] | None = None,
    out_of_order_id: int | None = None,
    offset_parameter: str | None = None,
    length_parameter: str | None = None,
    length_unit: int | None = None,
    iteration: BdIteration | None = None,
):
    """_summary_
    Enables data transfers between the AIE Engine array and external memory.
    DMA tasks operations do not require to specify a BD number and are capable of chaining BD operations.

    Args:
        alloc: The alloc argument associates the DMA task with an ObjectFIFO. This argument is called alloc because the shim-side end of a data transfer (specifically a channel on a shim tile) is referenced through a so-called "shim DMA allocation". When an ObjectFIFO is created with a Shim Tile endpoint, an allocation with the same name as the ObjectFIFO is automatically generated.
        mem: Reference to a host buffer, given as an argument to the sequence function, that this transfer will read from or write to.
        tap (optional): A TensorAccessPattern is an alternative method of specifying offset/sizes/strides for determining an access pattern over the mem buffer.
        offset (optional): Starting point for the data transfer. Default values is 0.
        sizes: The extent of data to be transferred across each dimension. The dimensions before the last three are iteration dimensions, one execution of the BD per index; past four in all, which must then be constant, the compiler splits the transfer into several BDs. On an
            AIE4 shim a five-entry pattern is [iteration, d3, d2, d1, d0].
        strides (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data.
        issue_token (optional): If a token is issued, one may call dma_await_task on the returned task. Default is False.
        burst_length (optional): The configuration of the burst length for the DMA task. If 0, defaults to the highest available value.
        axcache (optional): The raw 4-bit AxCACHE value for the DMA's AXI-MM transfers. If
            omitted, the target model's default AxCACHE value is used.
        packet (optional): The packet header information represented as a (packet_type, packet_id) tuple.
        length_parameter (optional): Name of a scratchpad parameter n; the
            transfer then moves the static length plus n * length_unit elements
            (per iteration). With `transfer_len=0` it moves exactly n units,
            each shaped by `sizes`. Shim tiles only; see the length_parameter
            docs on aie.dma_bd.
        length_unit (optional): Elements added per unit of length_parameter; a
            multiple of 16 bytes. Defaults to the elements one pass of the
            innermost three `sizes` moves.
        iteration (optional): The BD iteration state as a BdIteration(size, stride, current).

    Example:
        out_task = shim_dma_single_bd_task(of_out, C, sizes=[1, 1, 1, N], issue_token=True)

        The example above describes a linear transfer of N data elements from the C buffer in host memory into an object FIFO with matching metadata labeled "of_out".
        The sizes dimensions are expressed right to left where the right is dimension 0 and the left dimension 3.
        Higher dimensions not used should be set to 1.
    """
    if tap and not (offset is None and sizes is None and strides is None):
        raise ValueError(
            "shim_dma_single_bd_task can take either a TensorAccessPattern OR (sizes and/or strides and/or offsets), but not both."
        )

    if tap:
        offset, sizes, strides = _shim_tap_dims(tap)

    # The iteration attribute (when set) claims the shim's single iteration/repeat
    # slot, so the BD carries ONLY its access dimensions (<=3) and does NOT also
    # hoist an outer dimension.
    repeat_count = 0
    repeat_count_val = None
    if iteration is not None:
        if sizes is not None and len(sizes) > 3:
            raise ValueError(
                "shim_dma_single_bd_task: iteration= supports at most 3 access "
                "dimensions (the iteration attribute claims the 4th/iteration "
                f"slot); got {len(sizes)}."
            )
    else:
        # The shim DMA BD has 3 access dimensions plus a hardware repeat/iteration
        # dimension. The repeat_count below hoists sizes[0] into that iteration
        # dimension, but the transferred extent is prod(sizes[-3:]) (see shim_dma_bd),
        # so sizes[0] is left out of it only when there are 4 dimensions. With fewer
        # than 4 dims and sizes[0] > 1, sizes[0] is counted both as a real access dim
        # (in transfer_len and in the BD dimensions) and as repeat_count, so the shim
        # re-issues the whole transfer sizes[0] times, waits on the objectfifo for that
        # many objects, and dma_await_task never returns. Normalize to the canonical
        # 4-dim form (left-pad with unit dims) so sizes[0] is only ever the iteration
        # dimension. Past 4 dims, every dim before the last three iterates, and
        # aie-decompose-large-dma-bd splits the ones a BD cannot hold into separate
        # tasks, which it can only do for constant sizes and strides.
        if sizes is not None:
            if _shim_has_d3(sizes[:5]) and len(sizes) > 5:
                raise ValueError(
                    f"AIE4 shim DMA BD supports at most 5 dimensions, got {len(sizes)}"
                )
            while len(sizes) < 4:
                sizes = [1] + list(sizes)
                if strides is not None:
                    strides = [0] + list(strides)

        # The outermost size is the iteration/repeat dimension. The AIE4 shim itself
        # DOES support an advancing iteration -- the arch spec defines an
        # iteration-offset applied between transfers, the simulator applies
        # iteration_stepsize, and the raw dma_configure_task path uses it (covered
        # E2E by 16_shim_dma_4d). The limitation is specific to THIS objectfifo
        # helper: a strided outer dimension lowers to a self-repeating task
        # (repeat_count), which does not compose with the objectfifo's object/token
        # model -- empirically the repeats re-read the first block instead of
        # advancing (cert sim). So reject a strided outer here and ask the caller to
        # express the access as a single descriptor (outermost size 1, with the data
        # layout in d0..d3, letting the consuming objectfifo chunk it). A pure
        # (zero-stride) repeat is fine and lowers to a queue repeat_count; for a
        # genuinely advancing iteration, use the raw dma_configure_task path.
        outer_stride = strides[0] if strides else 0
        if sizes:
            # On AIE4 a 5-dim pattern's sizes[1] is the D3 addressing dimension,
            # so only sizes[0] repeats.
            outer = sizes[:1] if _shim_has_d3(sizes) else sizes[:-3]
            if all(isinstance(v, (int, np.integer)) for v in outer):
                runs = int(np.prod([int(v) for v in outer]))
                if runs > 1:
                    if outer_stride != 0 and len(outer) == 1:
                        raise ValueError(
                            "shim_dma_single_bd_task: a strided outermost (iteration) "
                            "dimension (sizes[0] > 1 with a nonzero stride) is not supported "
                            "through this objectfifo helper; the repeats do not advance "
                            "through the objectfifo and the transfer would silently re-read "
                            "the same block. Express the access as a single descriptor "
                            "(outermost size 1, data layout in d0..d3) and let the consuming "
                            "objectfifo chunk it, use a zero outer stride for a pure repeat, "
                            "or use the raw dma_configure_task path for an advancing iteration."
                        )
                    repeat_count = runs - 1
            else:
                # Runtime: a zero sizes[0] wraps to a huge count, which the
                # queue-push lowering refuses along with any other one past the
                # target's maximum.
                repeat_count_val = sizes[0] - 1
    task = dma_configure_task_for(
        alloc,
        repeat_count=repeat_count,
        repeat_count_val=repeat_count_val,
        issue_token=issue_token,
    )
    with bds(task) as bd:
        with bd[0]:
            shim_dma_bd(
                mem,
                offset=offset,
                sizes=sizes,
                strides=strides,
                transfer_len=transfer_len,
                burst_length=burst_length,
                axcache=axcache,
                packet=packet,
                out_of_order_id=out_of_order_id,
                offset_parameter=offset_parameter,
                length_parameter=length_parameter,
                length_unit=length_unit,
                iteration=iteration,
            )
            EndOp()
    return task


_orig_dma_await_task = dma_await_task


def dma_await_task(*args: DMAConfigureTaskForOp):
    if len(args) == 0:
        raise ValueError(
            "dma_await_task must receive at least one DMAConfigureTaskForOp to wait for"
        )
    for dma_task in args:
        _orig_dma_await_task(task=dma_task)


_orig_dma_free_task = dma_free_task


def dma_free_task(*args: DMAConfigureTaskForOp):
    if len(args) == 0:
        raise ValueError(
            "dma_free_task must receive at least one DMAConfigureTaskForOp to free"
        )
    for dma_task in args:
        _orig_dma_free_task(dma_task)


_orig_dma_start_task = dma_start_task


def dma_start_task(
    *args: DMAConfigureTaskForOp,
    repeat_count: int | Value | None = None,
    no_token: bool = False,
    repeat_parameter: str | None = None,
):
    """Push each task onto its channel's queue.

    ``repeat_count`` replaces the task's own count for these starts only, and
    ``no_token`` withholds the completion token the task would otherwise issue.
    A runtime ``repeat_count`` must fit one queue push. ``repeat_parameter``
    names a scratchpad parameter whose value, written per run, is the count
    instead, at most the task's own.
    """
    if len(args) == 0:
        raise ValueError(
            "dma_start_task must receive at least one DMAConfigureTaskForOp to start"
        )
    if repeat_parameter is not None and repeat_count is not None:
        raise ValueError(
            "dma_start_task takes a repeat_count or a repeat_parameter, not both"
        )
    runtime = isinstance(repeat_count, Value)
    repeat_count_val = repeat_count if runtime else None
    for dma_task in args:
        _orig_dma_start_task(
            dma_task,
            repeat_count=None if runtime else repeat_count,
            repeat_count_val=repeat_count_val,
            no_token=no_token or None,
            repeat_parameter=repeat_parameter,
        )


def set_lock_value(lock: aie.LockOp, value: int | Value):
    return set_lock(lock, _as_i32(si32(value)))


# Parameter ops

_orig_read_scratchpad_parameter = read_scratchpad_parameter


def read_scratchpad_parameter(
    name: str, result_type: Type
) -> _orig_read_scratchpad_parameter:
    """Read a scratchpad runtime parameter inside an `aie.core` body.

    Args:
        name: The `@sym_name` of the `aiex.scratchpad_parameter` declaration.
        result_type: The MLIR scalar type of the result (e.g. `T.bf16()`, `T.i32()`).

    Returns:
        An SSA value of the given type.

    For example:

    ```python
    val = aiex.read_scratchpad_parameter("foo", T.bf16())
    ```
    """
    return _orig_read_scratchpad_parameter(result_type, name)


# CERT control-code region ops
#
# The generated builders create the op but leave its region empty, and every
# cert region op carries an implicit `aie.end` terminator the caller would
# otherwise have to append by hand. These decorators do both, in the style of
# `runtime_sequence` above: the decorated function *is* the region body, so
# Python nesting mirrors the nesting in the emitted MLIR.
#
#     @cert_attach_to_group(2)
#     def _():
#         @cert_page()
#         def _():
#             @cert_job(0)
#             def _():
#                 cert_write32(0x600400, 0xA5A5A5A5)
#
# The raw builders remain available under their `Cert*Op` class names for
# callers that want to populate the region themselves.


def _cert_region(op):
    """Return a decorator that runs `f` inside `op`'s (single) region block."""

    def decorator(f):
        with InsertionPoint(op.body.blocks.append()):
            f()
            EndOp()
        return op

    return decorator


def cert_page(placement=None):
    """A page of cert control code.

    Args:
        placement: microcontroller (CERT group) id that runs this page, as
            returned by the target model's `getUcGroupId(col, half)`
            (AIE4: `2*col + half`). When omitted the page defaults to group 0.
    """
    return _cert_region(CertPageOp(placement=placement))


def cert_job(job_id):
    """A cert job. `job_id` is a label unique within the enclosing page, not a
    priority: execution order follows textual order."""
    return _cert_region(CertJobOp(job_id=job_id))


def cert_section(sym_name):
    """A named section of control code, referenced by `cert_load_pdi` or
    `cert_preempt`."""
    return _cert_region(CertSectionOp(sym_name=sym_name))


def cert_attach_to_group(group_id):
    """Attach the enclosed pages and jobs to microcontroller `group_id`."""
    return _cert_region(CertAttachToGroupOp(group_id=group_id))


def cert_uc_dma_chain(sym_name):
    """A named chain of `cert_uc_dma_bd` descriptors."""
    return _cert_region(CertUcDmaChainOp(sym_name=sym_name))


# CERT register operands
#
# `aiex.cert_reg` is a ConfinedAttr<I32Attr, [0, 23]>. ODS types such a
# constrained attribute as `Any` rather than `int`, so the generated builders
# reject a plain Python int with `std::bad_cast`. These wrappers accept an int
# (or an already-built attribute) and do the conversion, so a register operand
# reads as the register number it is: `cert_mov(3, 99)` targets `$r3`.


def _cert_reg(value):
    if isinstance(value, int):
        return IntegerAttr.get(IntegerType.get_signless(32), value)
    return value


_orig_cert_read32 = cert_read32


def cert_read32(value, address, **kwargs):
    """Read the constant address `address` into register `value`."""
    return _orig_cert_read32(_cert_reg(value), address, **kwargs)


_orig_cert_read32_d = cert_read32_d


def cert_read32_d(address, value, **kwargs):
    """Read the address held in register `address` into register `value`."""
    return _orig_cert_read32_d(_cert_reg(address), _cert_reg(value), **kwargs)


_orig_cert_add = cert_add


def cert_add(dest, value, **kwargs):
    """Add the constant `value` to register `dest`."""
    return _orig_cert_add(_cert_reg(dest), value, **kwargs)


_orig_cert_mov = cert_mov


def cert_mov(dest, value, **kwargs):
    """Move the constant `value` into register `dest`."""
    return _orig_cert_mov(_cert_reg(dest), value, **kwargs)


_orig_cert_uc_dma_write_des = cert_uc_dma_write_des


def cert_uc_dma_write_des(wait_handle, symbol, **kwargs):
    """Enqueue a uC-DMA chain asynchronously, returning a handle in
    `wait_handle`. Pair with `cert_wait_uc_dma`."""
    return _orig_cert_uc_dma_write_des(_cert_reg(wait_handle), symbol, **kwargs)


_orig_cert_wait_uc_dma = cert_wait_uc_dma


def cert_wait_uc_dma(wait_handle, **kwargs):
    """Wait for the uC-DMA transfer identified by `wait_handle`. Yields to
    another job rather than spinning."""
    return _orig_cert_wait_uc_dma(_cert_reg(wait_handle), **kwargs)
