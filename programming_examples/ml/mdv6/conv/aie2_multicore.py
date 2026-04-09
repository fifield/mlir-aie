"""Generalized multi-core tiled fused Conv+BN+SiLU for AIE2P.

Supports:
- Conv1x1 and Conv3x3 (with stride and padding)
- 1-32 cores (1-8 columns × up to 4 tiles/col)
- patches_per_core batching (each core processes multiple tiles per invocation)
- Per-column weight broadcast
- Hierarchical split/join at memtile for per-column data distribution

Usage:
  python3 aie2_multicore.py n_cores tile_h tile_w ic oc kernel_size [stride] [patches_per_core]
"""
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def multicore_conv(dev, tile_h=8, tile_w=8, ic=16, oc=16,
                   kernel_size=1, stride_val=1, padding_val=0,
                   n_cores=32, patches_per_core=1):
    """N-core tiled fused Conv+BN+SiLU."""

    if kernel_size == 1:
        padding_val = 0
    elif kernel_size == 3 and padding_val < 0:
        padding_val = 1

    patch_h = (tile_h - 1) * stride_val + kernel_size
    patch_w = (tile_w - 1) * stride_val + kernel_size
    patch_size_raw = patch_h * patch_w * ic
    patch_size = patch_size_raw + (patch_size_raw % 2)  # DMA alignment
    conv_weight_size = oc * ic * kernel_size * kernel_size
    bn_size = oc
    weight_block_size = conv_weight_size + 2 * bn_size
    output_tile_size = tile_h * tile_w * oc

    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col
    total_patches = n_cores * patches_per_core

    print(f"Multicore conv{kernel_size}x{kernel_size} ({ic}->{oc}), "
          f"tile {tile_h}x{tile_w}, stride={stride_val}, "
          f"{n_cores} cores, {patches_per_core} patches/core, "
          f"{total_patches} total", file=sys.stderr)
    print(f"  patch={patch_h}x{patch_w}x{ic}={patch_size}, "
          f"wt={weight_block_size}, out_tile={output_tile_size}", file=sys.stderr)

    # Per-tile types
    patch_ty = np.ndarray[(patch_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]

    # Per-core types (for batching)
    core_input_size = patches_per_core * patch_size
    core_output_size = patches_per_core * output_tile_size
    core_input_ty = np.ndarray[(core_input_size,), np.dtype[np.uint16]]
    core_output_ty = np.ndarray[(core_output_size,), np.dtype[np.uint16]]

    # Host buffer types
    host_input_size = n_cores * core_input_size
    host_output_size = n_cores * core_output_size
    host_input_ty = np.ndarray[(host_input_size,), np.dtype[np.uint16]]
    host_output_ty = np.ndarray[(host_output_size,), np.dtype[np.uint16]]

    # Kernel
    kern_name = "conv1x1_fused_packed_bf16" if kernel_size == 1 else "conv3x3_fused_packed_bf16"
    kernel = Kernel(kern_name, "conv_bf16.o", [
        patch_ty, weight_ty, output_tile_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    stride = stride_val
    padding = padding_val

    def core_fn(of_in, of_wt, of_out, kern):
        elem_wt = of_wt.acquire(1)
        for _ in range_(patches_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kern(elem_in, elem_wt, elem_out,
                 tile_h, tile_w, ic, oc, stride, padding)
            of_in.release(1)
            of_out.release(1)
        of_wt.release(1)

    # Build per-column infrastructure
    col_in_fifos = []
    col_out_fifos = []
    wt_fifos = []
    workers = []

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)

        # Column-level types
        col_in_size = cores_this_col * core_input_size
        col_out_size = cores_this_col * core_output_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        # Bulk input → split at memtile (depth=1 to save L1 memory)
        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[core_input_size * i for i in range(cores_this_col)],
            obj_types=[patch_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"input_{col}_{i}" for i in range(cores_this_col)],
        )

        # Per-core output → join at memtile (depth=1)
        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[core_output_size * i for i in range(cores_this_col)],
            obj_types=[output_tile_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"output_{col}_{i}" for i in range(cores_this_col)],
        )

        # Weight broadcast (depth=1)
        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"weights_{col}")

        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        for i in range(cores_this_col):
            w = Worker(core_fn, [
                in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(), kernel,
            ], stack_size=4096)
            workers.append(w)

    # Runtime
    rt = Runtime()
    with rt.sequence(host_input_ty, weight_ty, host_output_ty) as (I, W, O):
        rt.start(*workers)

        for wf in wt_fifos:
            rt.fill(wf.prod(), W)

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
            tap_out = TensorAccessPattern(
                (host_output_size,),
                offset=col * cores_per_col * core_output_size,
                sizes=[1, col_out_size],
                strides=[0, 1],
            )
            rt.fill(col_in_fifos[col].prod(), I, tap_in)
            rt.drain(col_out_fifos[col].cons(), O, tap_out,
                     wait=(col == n_cols - 1))

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    dev = NPU2()
    n_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    tile_h = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tile_w = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ic = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    oc = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    ks = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    stride = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    ppc = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    module = multicore_conv(dev, tile_h, tile_w, ic, oc, ks, stride,
                            1 if ks == 3 else 0, n_cores, ppc)
    print(module)
