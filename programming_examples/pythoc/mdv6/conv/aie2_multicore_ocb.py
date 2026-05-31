"""Phase E prototype — multicore Conv+BN+SiLU with OCB-unroll.

Variant of aie2_multicore.py that collapses the host-Python OCB loop
into the runtime sequence at compile time. One xrt.run processes
`n_ocb` consecutive output channel blocks, with the memtile DMA TAP
striding into a concatenated weight BO and a strided output BO.

The kernel `core_fn` wraps its body in `range_(n_ocb)`; each iteration
acquires a fresh weight slot, processes ppc patches, releases. The
runtime sequence emits per-OCB fill/drain descriptors via the same
compile-time unroll mechanism the existing `range_(patches_per_core)`
already uses for the spatial dimension.

Host contract (input/weight/output BO sizes):
- I: host_input_size bf16 elements   (same as aie2_multicore.py)
- W: n_ocb × weight_block_size bf16   (concatenated per-OCB weight slots)
- O: n_ocb × host_output_size bf16    (concatenated per-OCB outputs)

Input is currently re-filled from DDR per OCB (input-invariant across
OCBs but cheaper-to-redmaa than wider memtile L2 staging). The redundant
DMA is small (~5 µs/OCB on re8) vs the per-dispatch overhead saved.
"""
import argparse
import os
import numpy as np
import sys

from aie.iron import (
    Buffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.pythoc import PythocKernel
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


KERNELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "kernels", "build")
)


def multicore_conv_ocb(dev, tile_h=8, tile_w=8, ic=16, oc_block=16,
                       n_ocb=2,
                       kernel_size=3, stride_val=1, padding_val=1,
                       n_cores=32, patches_per_core=1, input_depth=1,
                       active_tile_h=None, active_tile_w=None,
                       active_ic=None, active_oc=None,
                       active_stride=None, active_padding=None):
    """N-core tiled fused Conv+BN+SiLU with OCB unrolled in runtime sequence.

    `oc_block` is the PER-OCB output channel block; total output channels =
    n_ocb * oc_block.
    """
    if kernel_size == 1:
        padding_val = 0
    elif kernel_size == 3 and padding_val < 0:
        padding_val = 1

    active_tile_h = tile_h if active_tile_h is None else active_tile_h
    active_tile_w = tile_w if active_tile_w is None else active_tile_w
    active_ic = ic if active_ic is None else active_ic
    active_oc = oc_block if active_oc is None else active_oc
    active_stride = stride_val if active_stride is None else active_stride
    active_padding = padding_val if active_padding is None else active_padding

    patch_h = (tile_h - 1) * stride_val + kernel_size
    patch_w = (tile_w - 1) * stride_val + kernel_size
    patch_size_raw = patch_h * patch_w * ic
    patch_size = patch_size_raw + (patch_size_raw % 2)
    conv_weight_size = oc_block * ic * kernel_size * kernel_size
    bn_size = oc_block
    weight_block_size = conv_weight_size + 2 * bn_size
    output_tile_size = tile_h * tile_w * oc_block

    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col
    total_patches = n_cores * patches_per_core

    print(f"OCB-unroll conv{kernel_size}x{kernel_size} ({ic}->{oc_block*n_ocb}, "
          f"oc_block={oc_block}, n_ocb={n_ocb}), tile {tile_h}x{tile_w}, "
          f"stride={stride_val}, {n_cores} cores, {patches_per_core} ppc",
          file=sys.stderr)
    print(f"  patch={patch_h}x{patch_w}x{ic}={patch_size}, "
          f"wt_slot={weight_block_size}, out_tile={output_tile_size}",
          file=sys.stderr)

    # Per-tile types
    patch_ty = np.ndarray[(patch_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]

    # Per-core types
    core_input_size = patches_per_core * patch_size
    core_output_size = patches_per_core * output_tile_size
    core_input_ty = np.ndarray[(core_input_size,), np.dtype[np.uint16]]
    core_output_ty = np.ndarray[(core_output_size,), np.dtype[np.uint16]]

    # Host buffer types — input unchanged, weight × n_ocb, output × n_ocb
    host_input_size = n_cores * core_input_size
    host_output_size = n_cores * core_output_size
    big_weight_size = n_ocb * weight_block_size
    big_output_size = n_ocb * host_output_size
    host_input_ty = np.ndarray[(host_input_size,), np.dtype[np.uint16]]
    big_weight_ty = np.ndarray[(big_weight_size,), np.dtype[np.uint16]]
    big_output_ty = np.ndarray[(big_output_size,), np.dtype[np.uint16]]

    kern_name = (
        "gemm_conv1x1_fused_packed_bf16" if kernel_size == 1
        else "conv3x3_fused_packed_bf16"
    )
    kernel = PythocKernel(
        kern_name,
        os.path.join(KERNELS_DIR, f"{kern_name}.o"),
        [
            patch_ty, weight_ty, output_tile_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
    )

    stride = active_stride
    padding = active_padding

    RTP_LEN = 6
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    init_rtp = np.array([active_tile_h, active_tile_w, active_ic, active_oc,
                         stride, padding], dtype=np.int32)
    rtps = [
        Buffer(rtp_ty, name=f"rtp_{i}", initial_value=init_rtp, use_write_rtp=True)
        for i in range(n_cores)
    ]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    def core_fn(of_in, of_wt, of_out, kern, my_rtp, barrier):
        # NOTE: barrier handshake stays outside the OCB loop — one wait at
        # entry, one release at exit. The kernel body runs n_ocb times,
        # each time acquiring a fresh weight slot from the memtile.
        #
        # Outer OCB loop uses Python `range(n_ocb)` (not IRON `range_()`) so
        # each iteration unrolls at build time into a separate acquire/release
        # block, matching the unroll behavior the runtime sequence uses for
        # per-OCB DMA descriptors. IRON `range_()` keeps a scf.for around
        # objectfifo ops which doesn't lower cleanly with the existing kernel
        # pattern.
        barrier.wait_for_value(1)
        t_h = my_rtp[0]
        t_w = my_rtp[1]
        ic_v = my_rtp[2]
        oc_v = my_rtp[3]
        str_v = my_rtp[4]
        pad_v = my_rtp[5]
        for _ in range(n_ocb):
            elem_wt = of_wt.acquire(1)
            for _ in range_(patches_per_core):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                kern(elem_in, elem_wt, elem_out,
                     t_h, t_w, ic_v, oc_v, str_v, pad_v)
                of_in.release(1)
                of_out.release(1)
            of_wt.release(1)
        barrier.release_with_value(1)

    # Per-column infrastructure
    col_in_fifos = []
    col_out_fifos = []
    wt_fifos = []
    workers = []

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)

        col_in_size = cores_this_col * core_input_size
        col_out_size = cores_this_col * core_output_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[core_input_size * i for i in range(cores_this_col)],
            obj_types=[patch_ty] * cores_this_col,
            depths=[input_depth] * cores_this_col,
            names=[f"input_{col}_{i}" for i in range(cores_this_col)],
        )

        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[core_output_size * i for i in range(cores_this_col)],
            obj_types=[output_tile_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"output_{col}_{i}" for i in range(cores_this_col)],
        )

        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"weights_{col}")

        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        for i in range(cores_this_col):
            global_core_idx = col * cores_per_col + i
            w = Worker(core_fn, [
                in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(), kernel,
                rtps[global_core_idx], barriers[global_core_idx],
            ], stack_size=4096)
            workers.append(w)

    rt = Runtime()
    with rt.sequence(host_input_ty, big_weight_ty, big_output_ty) as (I, W, O):
        rt.start(*workers)

        t_h, t_w = active_tile_h, active_tile_w
        ic_c, oc_c, s_c, p_c = active_ic, active_oc, stride, padding
        def set_rtps(*rtp_bufs):
            for rb in rtp_bufs:
                rb[0] = t_h; rb[1] = t_w
                rb[2] = ic_c; rb[3] = oc_c
                rb[4] = s_c;  rb[5] = p_c
        rt.inline_ops(set_rtps, rtps)
        for b in barriers:
            rt.set_barrier(b, 1)

        # OCB loop — compile-time unrolled. Each iteration emits its own
        # weight fill (strided into W) + input fill (same offset, reused
        # data) + output drain (strided into O).
        #
        # Weight fill uses a 4D TAP to match the descriptor shape the
        # original aie2_multicore.py emits when calling rt.fill(..., W) with
        # no TAP (the default lowers to a 4D descriptor on weight broadcasts).
        # Input/output fills use 2D TAPs matching the original's explicit
        # TAPs on those fifos.
        for ocb in range(n_ocb):
            tap_wt = TensorAccessPattern(
                (big_weight_size,),
                offset=ocb * weight_block_size,
                sizes=[1, 1, 1, weight_block_size],
                strides=[0, 0, 0, 1],
            )
            for wf in wt_fifos:
                rt.fill(wf.prod(), W, tap_wt)

            for col in range(n_cols):
                cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
                col_in_size = cores_this_col * core_input_size
                col_out_size = cores_this_col * core_output_size

                tap_in = TensorAccessPattern(
                    (host_input_size,),
                    offset=col * cores_per_col * core_input_size,
                    sizes=[1, col_in_size],
                    strides=[0, 1],
                )
                # Output TAP: stride into O by (ocb * host_output_size) +
                # (col * cores_per_col * core_output_size). Per-column slice
                # stays the same size; ocb stride places this OCB's outputs
                # after all earlier OCBs' outputs in the host BO.
                tap_out = TensorAccessPattern(
                    (big_output_size,),
                    offset=(ocb * host_output_size +
                            col * cores_per_col * core_output_size),
                    sizes=[1, col_out_size],
                    strides=[0, 1],
                )
                rt.fill(col_in_fifos[col].prod(), I, tap_in)
                # Wait only on the final OCB's drain — earlier drains can
                # overlap with subsequent fills/compute.
                rt.drain(col_out_fifos[col].cons(), O, tap_out,
                         wait=(ocb == n_ocb - 1))

    return Program(dev, rt).resolve_program()


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate OCB-unrolled multicore Conv+BN+SiLU MLIR."
    )
    parser.add_argument("n_cores", nargs="?", type=int, default=32)
    parser.add_argument("tile_h", nargs="?", type=int, default=4)
    parser.add_argument("tile_w", nargs="?", type=int, default=4)
    parser.add_argument("ic", nargs="?", type=int, default=64)
    parser.add_argument("oc_block", nargs="?", type=int, default=16)
    parser.add_argument("n_ocb", nargs="?", type=int, default=4)
    parser.add_argument("kernel_size", nargs="?", type=int, default=3)
    parser.add_argument("stride", nargs="?", type=int, default=1)
    parser.add_argument("patches_per_core", nargs="?", type=int, default=1)
    parser.add_argument("--active-tile-h", type=int)
    parser.add_argument("--active-tile-w", type=int)
    parser.add_argument("--active-ic", type=int)
    parser.add_argument("--active-oc", type=int)
    parser.add_argument("--active-stride", type=int)
    parser.add_argument("--active-padding", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    dev = NPU2()
    args = _parse_args(sys.argv[1:])
    module = multicore_conv_ocb(
        dev,
        args.tile_h, args.tile_w, args.ic, args.oc_block,
        args.n_ocb,
        args.kernel_size, args.stride,
        1 if args.kernel_size == 3 else 0,
        args.n_cores, args.patches_per_core, 1,
        active_tile_h=args.active_tile_h,
        active_tile_w=args.active_tile_w,
        active_ic=args.active_ic,
        active_oc=args.active_oc,
        active_stride=args.active_stride,
        active_padding=args.active_padding,
    )
    print(module)
