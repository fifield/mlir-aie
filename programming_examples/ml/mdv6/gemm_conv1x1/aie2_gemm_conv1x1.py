"""GEMM-based Conv1x1 for AIE2P using 8x8x8 bf16 mmul.

Maps 1x1 convolution to matrix multiply: Out[M,N] = In[M,K] x W[K,N]
where M = spatial (H*W), K = input_channels, N = output_channels.

Input is in HWC layout = row-major [M,K] -- no reshape needed.
Weights pre-packed offline in [N/8, K/8, 8, 8] blocked layout.

Multicore: up to 32 cores (8 columns x 4 tiles), spatial parallelism over M.
Each core processes tile_m pixels. Weight broadcast to all cores.

Usage:
  python3 aie2_gemm_conv1x1.py n_cores tile_m ic oc [patches_per_core]

Examples:
  # 32-core, 64 pixels/core, 128->64 channels
  python3 aie2_gemm_conv1x1.py 32 64 128 64

  # 8-core, 128 pixels/core, 64->128 channels, 4 patches each
  python3 aie2_gemm_conv1x1.py 8 128 64 128 4
"""
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def gemm_conv1x1(dev, tile_m=64, ic=128, oc=64, n_cores=32,
                 patches_per_core=1, fused=True):
    """N-core GEMM-based Conv1x1 [+ BN + SiLU].

    Args:
        tile_m: spatial pixels per core per patch (must be %16==0)
        ic: input channels (must be %8==0)
        oc: output channels (must be %16==0)
        n_cores: number of compute cores (1-32)
        patches_per_core: patches each core processes per invocation
        fused: if True, use BN+SiLU fused kernel; else pure GEMM
    """
    assert tile_m % 4 == 0, f"tile_m={tile_m} must be divisible by 4 (mmul<4,8,8>)"
    assert ic % 8 == 0, f"ic={ic} must be divisible by 8"
    assert oc % 8 == 0, f"oc={oc} must be divisible by 8"

    # Buffer sizes (in bf16 elements, passed as uint16)
    input_tile_size = tile_m * ic
    if fused:
        weight_size = oc * ic + 2 * oc  # conv_weights + bn_w + bn_b
    else:
        weight_size = oc * ic
    output_tile_size = tile_m * oc

    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col
    total_patches = n_cores * patches_per_core

    # Memory estimate per core
    mem_per_core = (input_tile_size + weight_size + output_tile_size) * 2 + 2048
    print(f"GEMM Conv1x1: tile_m={tile_m}, {ic}->{oc}, "
          f"{n_cores} cores, {patches_per_core} patches/core", file=sys.stderr)
    print(f"  input={input_tile_size} wt={weight_size} out={output_tile_size} "
          f"mem/core={mem_per_core/1024:.1f}KB", file=sys.stderr)

    # Per-tile types
    input_ty = np.ndarray[(input_tile_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]

    # Per-core types (for batched patches)
    core_in_size = patches_per_core * input_tile_size
    core_out_size = patches_per_core * output_tile_size
    core_in_ty = np.ndarray[(core_in_size,), np.dtype[np.uint16]]
    core_out_ty = np.ndarray[(core_out_size,), np.dtype[np.uint16]]

    # Host buffer types
    host_in_size = n_cores * core_in_size
    host_out_size = n_cores * core_out_size
    host_in_ty = np.ndarray[(host_in_size,), np.dtype[np.uint16]]
    host_out_ty = np.ndarray[(host_out_size,), np.dtype[np.uint16]]

    # Kernel -- same extern "C" interface as existing conv kernels
    kern_name = "gemm_conv1x1_fused_packed_bf16" if fused else "gemm_conv1x1_bf16"
    kernel = Kernel(kern_name, "gemm_conv1x1_bf16.o", [
        input_ty, weight_ty, output_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    # tile_h and tile_w are passed so that tile_h * tile_w = tile_m
    # Use tile_h=tile_m, tile_w=1 since the kernel only uses the product
    kern_tile_h = tile_m
    kern_tile_w = 1
    stride = 1
    padding = 0

    def core_fn(of_in, of_wt, of_out, kern):
        elem_wt = of_wt.acquire(1)
        for _ in range_(patches_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kern(elem_in, elem_wt, elem_out,
                 kern_tile_h, kern_tile_w, ic, oc, stride, padding)
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

        col_in_size = cores_this_col * core_in_size
        col_out_size = cores_this_col * core_out_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        # Bulk input -> split at memtile -> per-core
        # depth=1: no double-buffering to fit in 64KB L1
        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[core_in_size * i for i in range(cores_this_col)],
            obj_types=[input_ty] * cores_this_col,
            names=[f"input_{col}_{i}" for i in range(cores_this_col)],
        )

        # Per-core output -> join at memtile -> bulk output
        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[core_out_size * i for i in range(cores_this_col)],
            obj_types=[output_ty] * cores_this_col,
            names=[f"output_{col}_{i}" for i in range(cores_this_col)],
        )

        # Weight broadcast: depth=1 (acquired once, held for all patches)
        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"weights_{col}")

        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        for i in range(cores_this_col):
            w = Worker(core_fn, [
                in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(), kernel,
            ], stack_size=8192)
            workers.append(w)

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(host_in_ty, weight_ty, host_out_ty) as (I, W, O):
        rt.start(*workers)

        # Broadcast weights to all columns
        for wf in wt_fifos:
            rt.fill(wf.prod(), W)

        # Distribute input and collect output per column
        for col in range(n_cols):
            cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
            col_in_sz = cores_this_col * core_in_size
            col_out_sz = cores_this_col * core_out_size

            tap_in = TensorAccessPattern(
                (host_in_size,),
                offset=col * cores_per_col * core_in_size,
                sizes=[1, col_in_sz],
                strides=[0, 1],
            )
            tap_out = TensorAccessPattern(
                (host_out_size,),
                offset=col * cores_per_col * core_out_size,
                sizes=[1, col_out_sz],
                strides=[0, 1],
            )
            rt.fill(col_in_fifos[col].prod(), I, tap_in)
            rt.drain(col_out_fifos[col].cons(), O, tap_out,
                     wait=(col == n_cols - 1))

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    dev = NPU2()
    n_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    tile_m = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    ic = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    oc = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    ppc = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    fused = "--no-fuse" not in sys.argv

    module = gemm_conv1x1(dev, tile_m, ic, oc, n_cores, ppc, fused)
    print(module)
