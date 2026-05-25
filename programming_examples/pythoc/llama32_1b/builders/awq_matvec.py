# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the standalone packed-uint4 AWQ GEMV kernel.

Replaces the cached AIR-stitched
``awq_gemv_k{K}_m{M}_g{group_size}_{variant}.npu.air.mlir`` files with an
``aie/aiex``-dialect Python builder.  Parameterized on ``(k, m,
group_size, variant)``.  The kernel object linked into each device is
dim-specialised (one ``.o`` per shape -- the Stage-2 PythoC kernel bakes
``K``, ``M``, ``GROUP_SIZE`` as Python-source constants); the builder
itself is one function that takes the dims as kwargs.

Module layout (matches ``reference_mlir/awq_gemv_k*.npu.air.mlir``):

    module {
        aie.device(npu2) @awq_gemv_seg { ... }   # seg device (compute)
        aie.device(npu2) {                       # dispatcher
            aie.runtime_sequence @awq_gemv(...) {
                aiex.configure @awq_gemv_seg { aiex.run ... }
            }
        }
    }

Per seg device (single shim col 0 + single compute tile 0,2 -- no mem tile):

  * 8 locks on tile (0,2), pairs init=1/0 for {params, qweight, x, y}
  * 4 L1 buffers: ``buf3`` x (bf16[K]), ``buf2`` qweight (ui8[M*K/2]),
    ``buf1`` params (bf16[M*2*K/group_size]), ``buf0`` y (bf16[M])
  * 4 ``external_buffer`` declarations matching the L3 sides
  * 1 ``aie.mem`` block: MM2S0 chain (y out, 1 BD) + S2MM0 chain
    (x→qweight→params→x, 3 BDs cyclic)
  * 1 ``aie.core`` block: forever {acquire 4 ready locks → call
    awq_gemv_u4_bf16 → release 4 avail locks + y_full}
  * 3 ``aie.packet_flow`` (one per input: x, qweight, params),
    all from shim 0_0 DMA:0 → tile_0_2 DMA:0 (packet_id 0/1/2)
  * 1 ``aie.flow`` for output: tile_0_2 DMA:0 → shim 0_0 DMA:0
  * 4 ``aie.shim_dma_allocation``: 1 S2MM (output) + 3 MM2S (inputs).
    Three of the MM2S aliases (channel_1, channel_2) are unused
    AIR-allocator artifacts; kept for cached-MLIR op-count parity.
  * ``aie.runtime_sequence @awq_gemv_seg_sequence`` with 4 DMA tasks
    (x, qweight, params, y), all on @air_channel_0 except y on
    @air_channel_3.  Outputs get ``issue_token = true``.

References:
  * ``reference_mlir/awq_gemv_k2048_m32_g128_vecdeq.npu.air.mlir``
  * ``reference_mlir/awq_gemv_k8192_m8_g128_vecdeq.npu.air.mlir``
  * ``kernel_builder/awq_gemv_builder.py`` -- AIR-dialect reference
  * ``builders/lm_head_gemv.py`` -- closest placed-IRON sibling
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    buffer,
    core,
    device,
    dma_bd,
    dma_start,
    external_buffer,
    external_func,
    flow,
    lock,
    mem,
    next_bd,
    packetflow,
    shim_dma_allocation,
    tile,
    use_lock,
)
from aie.dialects.aiex import (
    EndOp,
    bds,
    dma_await_task,
    dma_configure_task_for,
    dma_free_task,
    dma_start_task,
    runtime_sequence,
)
from aie.extras.context import mlir_mod_ctx
from aie.ir import InsertionPoint
from ._emit import attach_loop_annotation_to_all_scf_for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_AWQ_GEMV_VARIANTS = {"scalar", "vecdeq"}


def _validate_dims(k: int, m: int, group_size: int) -> tuple[int, int, int]:
    k = int(k)
    m = int(m)
    group_size = int(group_size)
    if k <= 0 or m <= 0 or group_size <= 0:
        raise ValueError(
            f"AWQ GEMV dimensions must be positive, got "
            f"k={k}, m={m}, group_size={group_size}"
        )
    if k % 2 != 0:
        raise ValueError(
            f"AWQ GEMV K must be even for uint4 byte packing, got {k}"
        )
    if k % group_size != 0:
        raise ValueError(
            f"AWQ GEMV K={k} must be divisible by group_size={group_size}"
        )
    return k, m, group_size


def _validate_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in _AWQ_GEMV_VARIANTS:
        raise ValueError(
            f"Unsupported AWQ GEMV variant {variant!r}; "
            f"expected one of {sorted(_AWQ_GEMV_VARIANTS)}"
        )
    return variant


def awq_gemv_kernel_name(k: int, m: int, group_size: int,
                        *, variant: str = "vecdeq") -> str:
    """Cache-safe name for a specialized packed-AWQ GEMV kernel."""
    k, m, group_size = _validate_dims(k, m, group_size)
    variant = _validate_variant(variant)
    return f"awq_gemv_k{k}_m{m}_g{group_size}_{variant}"


def awq_gemv_object_name(k: int, m: int, group_size: int,
                         *, variant: str = "vecdeq") -> str:
    """Object filename linked by the specialized AWQ GEMV IR."""
    return f"{awq_gemv_kernel_name(k, m, group_size, variant=variant)}_pythoc.o"


def _bf16_memref(*shape, memory_space=None):
    """``MemRefType<...xbf16, memory_space>``.  Must be called inside an
    active ``mlir_mod_ctx()``."""
    from aie.extras import types as T
    from aie.ir import MemRefType, IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


def _ui8_memref(*shape, memory_space=None):
    """``MemRefType<...xui8, memory_space>``.  Must be called inside an
    active ``mlir_mod_ctx()``."""
    from aie.extras import types as T
    from aie.ir import MemRefType, IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.ui8(), None, ms)


def _bf16_np(*shape):
    return np.ndarray[shape, np.dtype[bfloat16]]


def _ui8_np(*shape):
    return np.ndarray[shape, np.dtype[np.uint8]]


def _shim_stride_dims(n: int) -> list[tuple[int, int]]:
    """Return ``[(n//512, 512), (512, 1)]`` -- AIR's 2D shim BD pattern
    for any DMA whose ``len >= 512`` and ``len`` is a multiple of 512.

    Lengths below 512 produce a single contiguous ``[(n, 1)]`` dim,
    matching the cached MLIR's ``aie.dma_bd(...) [<size = N, stride = 1>]``
    pattern for the small ``y`` output buffer.
    """
    if n < 512 or n % 512 != 0:
        return [(n, 1)]
    return [(n // 512, 512), (512, 1)]


# ---------------------------------------------------------------------------
# Seg device emitter.
# ---------------------------------------------------------------------------
def _emit_seg_device(k: int, m: int, group_size: int,
                     *, variant: str) -> None:
    """Emit one ``aie.device(npu2) @awq_gemv_seg { ... }`` block.

    Must be called inside an active ``mlir_mod_ctx()`` at module
    insertion point.  Side-effecting; returns nothing.

    Uses ``aie.helpers.dialects.scf._for(maxsize)`` for the forever-loop
    inside ``aie.core``, mirroring ``builders/lm_head_gemv.py``.  This
    yields an ``scf.for`` in the IR rather than the cached AIR's
    ``cf.br ^bb1`` style; aiecc handles both forms identically once the
    ``loop_annotation = #llvm.loop_annotation<mustProgress = true>`` is
    attached by ``attach_loop_annotation_to_all_scf_for``.
    """
    obj_name = awq_gemv_object_name(k, m, group_size, variant=variant)

    q_len = m * (k // 2)
    p_len = m * 2 * (k // group_size)

    from aie.ir import UnitAttr as _UnitAttr
    from aie.helpers.dialects.scf import _for as range_
    import sys as _sys

    @device(AIEDevice.npu2, sym_name="awq_gemv_seg")
    def _seg():
        shim_tile = tile(0, 0)
        ct = tile(0, 2)

        # --- Locks (8) ------------------------------------------------
        lk_p_avail = lock(ct, lock_id=7, init=1)
        lk_p_ready = lock(ct, lock_id=6, init=0)
        lk_q_avail = lock(ct, lock_id=5, init=1)
        lk_q_ready = lock(ct, lock_id=4, init=0)
        lk_x_avail = lock(ct, lock_id=3, init=1)
        lk_x_ready = lock(ct, lock_id=2, init=0)
        lk_y_done = lock(ct, lock_id=1, init=1)
        lk_y_full = lock(ct, lock_id=0, init=0)

        # --- Buffers --------------------------------------------------
        _x_l1_ty = _bf16_memref(k, memory_space=2)
        _q_l1_ty = _ui8_memref(q_len, memory_space=2)
        _p_l1_ty = _bf16_memref(p_len, memory_space=2)
        _y_l1_ty = _bf16_memref(m, memory_space=2)

        buf_x = buffer(ct, datatype=_x_l1_ty, name="buf3")
        buf_q = buffer(ct, datatype=_q_l1_ty, name="buf2")
        buf_p = buffer(ct, datatype=_p_l1_ty, name="buf1")
        buf_y = buffer(ct, datatype=_y_l1_ty, name="buf0")

        # External (L3) buffer declarations
        external_buffer(_bf16_np(k), name="__air_external_buffer")
        external_buffer(_ui8_np(q_len), name="__air_external_buffer_1")
        external_buffer(_bf16_np(p_len), name="__air_external_buffer_2")
        external_buffer(_bf16_np(m), name="__air_external_buffer_3")

        # --- aie.mem block --------------------------------------------
        @mem(ct)
        def _core_mem(block):
            # bb0
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(lk_y_full, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_y, offset=0, len=m)
                use_lock(lk_y_done, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                EndOp()
            with block[3]:
                dma_start(DMAChannelDir.S2MM, 0,
                          dest=block[4], chain=block[2])
            with block[4]:
                use_lock(lk_x_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_x, offset=0, len=k)
                use_lock(lk_x_ready, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(lk_q_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_q, offset=0, len=q_len)
                use_lock(lk_q_ready, LockAction.Release, value=1)
                next_bd(block[6])
            with block[6]:
                use_lock(lk_p_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_p, offset=0, len=p_len)
                use_lock(lk_p_ready, LockAction.Release, value=1)
                next_bd(block[4])

        # --- External func declaration --------------------------------
        awq_fn = external_func(
            "awq_gemv_u4_bf16",
            inputs=[_x_l1_ty, _q_l1_ty, _p_l1_ty, _y_l1_ty],
            link_with=obj_name,
        )
        awq_fn.operation.attributes["llvm.emit_c_interface"] = _UnitAttr.get()

        # --- aie.core block -------------------------------------------
        # NB: ``link_with`` lives on ``external_func`` (above), not on
        # ``aie.core`` -- the aie-assign-core-link-files pass aggregates
        # the per-func link files onto the cores it sees calls from.
        # The cached AIR MLIR shows ``link_with`` on the core; the
        # post-aie-assign-core-link-files IR is equivalent.
        @core(ct)
        def _core_body():
            # Acquire order matches AIR: y_done, x_ready, q_ready, p_ready.
            # Release order matches AIR: x_avail, q_avail, p_avail, y_full.
            for _ in range_(_sys.maxsize):
                use_lock(lk_y_done, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_x_ready, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_q_ready, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_p_ready, LockAction.AcquireGreaterEqual, value=1)
                awq_fn(buf_x, buf_q, buf_p, buf_y)
                use_lock(lk_x_avail, LockAction.Release, value=1)
                use_lock(lk_q_avail, LockAction.Release, value=1)
                use_lock(lk_p_avail, LockAction.Release, value=1)
                use_lock(lk_y_full, LockAction.Release, value=1)

        # --- Packet flows (3 inputs, shim_dma channel 0 multiplexed) --
        # Three packet_flow ops (pkt ids 0/1/2) all route shim 0_0 DMA:0
        # through the switch fabric to tile 0_2 DMA:0.  The cached MLIR
        # uses id-only packet_flow (no source/dest tile mention in
        # parens) -- AIR's packetflow Python wrapper emits the same
        # textual form because it constructs a body with one source op
        # and one dest op.
        for pkt_id in (0, 1, 2):
            packetflow(
                pkt_id=pkt_id,
                source=shim_tile,
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": ct, "port": WireBundle.DMA, "channel": 0},
            )

        # --- Output flow (single circuit-switched) --------------------
        flow(ct, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 0)

        # --- Shim DMA allocations -------------------------------------
        # @air_channel_3 (S2MM 0)  for y output.
        # @air_channel_0..2 (MM2S 0) for x, qweight, params.
        # AIR emits the S2MM alloc first, then the three MM2S aliases.
        # All three MM2S allocs sit on the same physical channel 0;
        # packet routing demuxes them by pkt_id.  The runtime_sequence
        # only references @air_channel_0 (for x, qweight, params) and
        # @air_channel_3 (for y) -- the _1 and _2 aliases are dead
        # symbols preserved for cached-MLIR op-count parity.
        shim_dma_allocation(
            "air_channel_3", shim_tile, DMAChannelDir.S2MM, 0,
        )
        shim_dma_allocation(
            "air_channel_0", shim_tile, DMAChannelDir.MM2S, 0,
        )
        shim_dma_allocation(
            "air_channel_1", shim_tile, DMAChannelDir.MM2S, 0,
        )
        shim_dma_allocation(
            "air_channel_2", shim_tile, DMAChannelDir.MM2S, 0,
        )

        # --- Runtime sequence ----------------------------------------
        @runtime_sequence(
            _bf16_np(k),
            _ui8_np(q_len),
            _bf16_np(p_len),
            _bf16_np(m),
            sym_name="awq_gemv_seg_sequence",
        )
        def _seq(arg_x, arg_q, arg_p, arg_y):
            # Input tasks (MM2S, packet-routed).
            x_task = dma_configure_task_for("air_channel_0")
            with bds(x_task) as bd:
                with bd[0]:
                    dma_bd(
                        arg_x,
                        offset=0,
                        len=k,
                        dimensions=_shim_stride_dims(k),
                        packet=(0, 0),
                    )
                    EndOp()
            dma_start_task(x_task)

            q_task = dma_configure_task_for("air_channel_0")
            with bds(q_task) as bd:
                with bd[0]:
                    dma_bd(
                        arg_q,
                        offset=0,
                        len=q_len,
                        dimensions=_shim_stride_dims(q_len),
                        packet=(0, 1),
                    )
                    EndOp()
            dma_start_task(q_task)

            p_task = dma_configure_task_for("air_channel_0")
            with bds(p_task) as bd:
                with bd[0]:
                    dma_bd(
                        arg_p,
                        offset=0,
                        len=p_len,
                        dimensions=_shim_stride_dims(p_len),
                        packet=(0, 2),
                    )
                    EndOp()
            dma_start_task(p_task)

            # Output task (S2MM, issue_token=true so host can await).
            y_task = dma_configure_task_for(
                "air_channel_3", issue_token=True,
            )
            with bds(y_task) as bd:
                with bd[0]:
                    dma_bd(
                        arg_y,
                        offset=0,
                        len=m,
                        dimensions=_shim_stride_dims(m),
                    )
                    EndOp()
            dma_start_task(y_task)

            # Free inputs (don't wait), await output, then free remaining
            # inputs.  AIR's ordering: free x; await y; free q; free p.
            dma_free_task(x_task)
            dma_await_task(y_task)
            dma_free_task(q_task)
            dma_free_task(p_task)


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
# ---------------------------------------------------------------------------
def _emit_dispatcher_device(k: int, m: int, group_size: int) -> None:
    """Emit the unnamed top-level dispatcher device.

    Carries the outer ``aie.runtime_sequence @awq_gemv`` that hands the
    4 host args to the seg device's ``awq_gemv_seg_sequence`` via
    ``aiex.configure`` + ``aiex.run``.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    q_len = m * (k // 2)
    p_len = m * 2 * (k // group_size)

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            _bf16_np(k),
            _ui8_np(q_len),
            _bf16_np(p_len),
            _bf16_np(m),
            sym_name="awq_gemv",
        )
        def _outer(arg_x, arg_q, arg_p, arg_y):
            cfg = ConfigureOp(symbol="awq_gemv_seg")
            blk = cfg.body.blocks.append()
            with InsertionPoint(blk):
                RunOp(
                    runtime_sequence_symbol="awq_gemv_seg_sequence",
                    args=[arg_x, arg_q, arg_p, arg_y],
                )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_awq_matvec_module(
    k: int,
    m: int,
    group_size: int = 128,
    variant: str = "vecdeq",
    *,
    verbose: bool = False,
) -> str:
    """Build the standalone AWQ GEMV ``aie/aiex``-dialect module.

    Args:
        k: input vector dimension (must be even and divisible by
            ``group_size``).
        m: number of output elements (= rows of the packed weight).
        group_size: AWQ quantization group size.  Default 128 matches
            the model shapes.
        variant: kernel variant -- ``"scalar"`` or ``"vecdeq"``.  Both
            produce identical placed-IRON output; only the linked
            ``.o`` differs.
        verbose: print a one-line trace.

    Returns:
        The MLIR module as text -- ready to hand to
        ``kernel_builder/aie_compile.compile_aie_to_elf``.
    """
    k, m, group_size = _validate_dims(k, m, group_size)
    variant = _validate_variant(variant)

    if verbose:
        print(
            f"  [awq_matvec] building placed-IRON module "
            f"k={k} m={m} g={group_size} variant={variant}"
        )

    with mlir_mod_ctx() as ctx:
        _emit_seg_device(k, m, group_size, variant=variant)
        _emit_dispatcher_device(k, m, group_size)
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)

    return str(module)


# ---------------------------------------------------------------------------
# CLI -- emit the module to stdout (useful for diffing vs cached MLIR).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=2048, help="K dimension")
    parser.add_argument("-m", type=int, default=32, help="M dimension")
    parser.add_argument("-g", "--group-size", type=int, default=128,
                        help="AWQ group size")
    parser.add_argument("--variant", choices=sorted(_AWQ_GEMV_VARIANTS),
                        default="vecdeq", help="Kernel variant")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: stdout)")
    args = parser.parse_args()
    text = build_awq_matvec_module(
        k=args.k, m=args.m, group_size=args.group_size,
        variant=args.variant,
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
