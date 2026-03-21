"""Multi-core weight broadcast conv1x1+BN+SiLU. N cores process N spatial patches
in parallel, sharing weights via per-column broadcast.

Architecture:
- Per column: 1 bulk input FIFO → split at memtile → 4 core FIFOs
              4 core output FIFOs → join at memtile → 1 bulk output FIFO
              1 weight FIFO → broadcast to 4 cores
- Shim DMA: 1 input + 1 output + 1 weight = 3/4 channels per column
- Supports 1-32 cores (1-8 columns × 1-4 tiles/col)
"""
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def multicore_broadcast_conv1x1(dev, tile_h=8, tile_w=8, ic=16, oc=16, n_cores=2):
    """N-core broadcast conv1x1+BN+SiLU."""

    patch_size = tile_h * tile_w * ic
    conv_weight_size = oc * ic
    bn_size = oc
    weight_block_size = conv_weight_size + 2 * bn_size
    output_tile_size = tile_h * tile_w * oc
    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col

    print(f"Multicore broadcast conv1x1 ({ic}->{oc}), tile {tile_h}x{tile_w}, "
          f"{n_cores} cores ({n_cols} cols)", file=sys.stderr)

    # Types
    patch_ty = np.ndarray[(patch_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]

    host_input_size = n_cores * patch_size
    host_output_size = n_cores * output_tile_size
    host_input_ty = np.ndarray[(host_input_size,), np.dtype[np.uint16]]
    host_output_ty = np.ndarray[(host_output_size,), np.dtype[np.uint16]]

    kernel = Kernel("conv1x1_fused_packed_bf16", "conv_bf16.o", [
        patch_ty, weight_ty, output_tile_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    stride = 1
    padding = 0

    def core_fn(of_in, of_wt, of_out, kern):
        elem_wt = of_wt.acquire(1)
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kern(elem_in, elem_wt, elem_out,
             tile_h, tile_w, ic, oc, stride, padding)
        of_in.release(1)
        of_out.release(1)
        of_wt.release(1)

    # Build per-column infrastructure
    col_in_fifos = []   # Bulk input FIFOs (host → memtile)
    col_out_fifos = []  # Bulk output FIFOs (memtile → host)
    wt_fifos = []       # Weight FIFOs (host → broadcast)
    workers = []

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)

        # Column-level memtile types
        col_in_size = cores_this_col * patch_size
        col_out_size = cores_this_col * output_tile_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        # Bulk input FIFO → split at memtile to per-core
        col_in_fifo = ObjectFifo(col_in_ty, name=f"col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[patch_size * i for i in range(cores_this_col)],
            obj_types=[patch_ty] * cores_this_col,
            names=[f"input_{col}_{i}" for i in range(cores_this_col)],
        )

        # Per-core output → join at memtile to bulk output FIFO
        col_out_fifo = ObjectFifo(col_out_ty, name=f"col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[output_tile_size * i for i in range(cores_this_col)],
            obj_types=[output_tile_ty] * cores_this_col,
            names=[f"output_{col}_{i}" for i in range(cores_this_col)],
        )

        # Weight FIFO for this column (broadcast to cores_this_col consumers)
        wt_fifo = ObjectFifo(weight_ty, name=f"weights_{col}")

        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        # Create workers for this column
        for i in range(cores_this_col):
            w = Worker(core_fn, [
                in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(), kernel,
            ], stack_size=4096)
            workers.append(w)

    # Runtime: distribute across columns with TensorAccessPattern
    rt = Runtime()
    with rt.sequence(host_input_ty, weight_ty, host_output_ty) as (I, W, O):
        rt.start(*workers)

        # Fill weights for each column (all read same host buffer)
        for wf in wt_fifos:
            rt.fill(wf.prod(), W)

        # Fill/drain per column
        for col in range(n_cols):
            cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
            col_in_size = cores_this_col * patch_size
            col_out_size = cores_this_col * output_tile_size

            tap_in = TensorAccessPattern(
                (host_input_size,),
                offset=col * cores_per_col * patch_size,
                sizes=[1, col_in_size],
                strides=[0, 1],
            )
            tap_out = TensorAccessPattern(
                (host_output_size,),
                offset=col * cores_per_col * output_tile_size,
                sizes=[1, col_out_size],
                strides=[0, 1],
            )
            rt.fill(col_in_fifos[col].prod(), I, tap_in)
            rt.drain(col_out_fifos[col].cons(), O, tap_out,
                     wait=(col == n_cols - 1))

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    dev = NPU2()
    n_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tile_h = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tile_w = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ic = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    oc = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    module = multicore_broadcast_conv1x1(dev, tile_h, tile_w, ic, oc, n_cores)
    print(module)
