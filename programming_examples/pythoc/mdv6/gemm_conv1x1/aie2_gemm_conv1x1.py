"""GEMM-based Conv1x1 for AIE2P using 8x8x8 bf16 mmul.

Maps 1x1 convolution to matrix multiply: Out[M,N] = In[M,K] x W[K,N]
where M = spatial (H*W), K = input_channels, N = output_channels.

Input is in HWC layout = row-major [M,K] -- no reshape needed.
Weights pre-packed offline in [N/8, K/8, 8, 8] blocked layout.

Multicore: up to 32 cores (8 columns x 4 tiles), spatial parallelism over M.
Each core processes tile_m pixels. Weight broadcast to all cores.

K-blocking: when full IC×OC weights don't fit in L1, weights are split into
k_block-sized IC chunks. Each core loops over K-blocks, accumulating partial
results. BN+SiLU applied after the last K-block. This eliminates OC blocking
at the host level — full OC is processed in a single NPU call.

Usage:
  python3 aie2_gemm_conv1x1.py n_cores tile_m ic oc [patches_per_core] [k_block]

Examples:
  # 32-core, 64 pixels/core, 128->64 channels (no K-blocking needed)
  python3 aie2_gemm_conv1x1.py 32 64 128 64

  # 32-core, 24 pixels/core, 512->256, k_block=32 (16 K-blocks)
  python3 aie2_gemm_conv1x1.py 32 24 512 256 1 32
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
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

# Pre-built PythoC kernel .o files live under ../kernels/build/. The PythocKernel
# wrapper takes a name + absolute .o path; aiecc resolves link_with relative to
# its --tmpdir, so build_gemm_conv1x1.py also stages the .o into the build dir.
KERNELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "kernels", "build")
)


def gemm_conv1x1(dev, tile_m=64, ic=128, oc=64, n_cores=32,
                 patches_per_core=1, k_block=0, fused=True,
                 active_tile_m=None, active_ic=None, active_oc=None,
                 active_k_block=None):
    """N-core GEMM-based Conv1x1 [+ BN + SiLU] with optional K-blocking.

    Args:
        tile_m: spatial pixels per core per patch (must be %4==0)
        ic: input channels (must be %8==0)
        oc: output channels (must be %8==0)
        n_cores: number of compute cores (1-32)
        patches_per_core: patches each core processes per invocation
        k_block: IC channels per K-block (0 = no K-blocking, use full ic)
        fused: if True, use BN+SiLU fused kernel; else pure GEMM
    """
    assert tile_m % 4 == 0, f"tile_m={tile_m} must be divisible by 4 (mmul<4,8,8>)"
    assert ic % 8 == 0, f"ic={ic} must be divisible by 8"
    assert oc % 8 == 0, f"oc={oc} must be divisible by 8"
    active_tile_m = tile_m if active_tile_m is None else active_tile_m
    active_ic = ic if active_ic is None else active_ic
    active_oc = oc if active_oc is None else active_oc

    # K-blocking setup
    if k_block <= 0 or k_block >= ic:
        k_block = ic  # No K-blocking
    assert k_block % 8 == 0, f"k_block={k_block} must be divisible by 8"
    assert ic % k_block == 0, f"ic={ic} must be divisible by k_block={k_block}"
    n_k_blocks = ic // k_block
    use_kblocking = n_k_blocks > 1
    if active_k_block is None:
        active_k_block = active_ic if not use_kblocking else k_block
    active_n_k_blocks = active_ic // active_k_block if use_kblocking else 1

    # Buffer sizes (in bf16 elements, passed as uint16)
    input_tile_size = tile_m * ic
    # Each weight chunk: k_block*oc conv weights + 2*oc BN params
    wt_chunk_size = k_block * oc + 2 * oc
    output_tile_size = tile_m * oc

    if not use_kblocking:
        # Original behavior: weight_size includes full IC weights + BN
        if fused:
            weight_size = oc * ic + 2 * oc
        else:
            weight_size = oc * ic
    else:
        weight_size = wt_chunk_size  # per-chunk size for ObjectFifo type

    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col

    # Memory estimate per core
    if use_kblocking:
        mem_per_core = (input_tile_size + output_tile_size + wt_chunk_size) * 2 + 2048
    else:
        mem_per_core = (input_tile_size + weight_size + output_tile_size) * 2 + 2048
    print(f"GEMM Conv1x1: tile_m={tile_m}, {ic}->{oc}, "
          f"{n_cores} cores, {patches_per_core} patches/core"
          + (f", k_block={k_block} ({n_k_blocks} K-blocks)" if use_kblocking else ""),
          file=sys.stderr)
    print(f"  input={input_tile_size} wt_chunk={weight_size} out={output_tile_size} "
          f"mem/core={mem_per_core/1024:.1f}KB", file=sys.stderr)
    if (active_tile_m, active_ic, active_oc, active_k_block) != (
            tile_m, ic, oc, k_block):
        print(f"  active RTP: tile_m={active_tile_m}, {active_ic}->{active_oc}, "
              f"k_block={active_k_block}", file=sys.stderr)

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

    # Host weight buffer: for K-blocking, contains n_k_blocks chunks
    # (sent repeatedly for each patch via TAP)
    if use_kblocking:
        host_wt_size = n_k_blocks * wt_chunk_size
        host_wt_ty = np.ndarray[(host_wt_size,), np.dtype[np.uint16]]
    else:
        host_wt_ty = weight_ty

    # Kernel selection — PythoC-built .o files staged into KERNELS_DIR.
    # Non-K-blocked path uses the fused (matmul + BN + SiLU) kernel.
    # K-blocked path uses the kblocked kernel (BN+SiLU on the last K-block).
    if use_kblocking:
        kern_name = "gemm_conv1x1_kblocked_bf16"
    elif fused:
        kern_name = "gemm_conv1x1_fused_packed_bf16"
    else:
        # Pure GEMM (no BN+SiLU) — not built by the PythoC kernel set; fall back
        # to the fused kernel and let the host pass identity BN params.
        kern_name = "gemm_conv1x1_fused_packed_bf16"

    obj_path = os.path.join(KERNELS_DIR, f"{kern_name}.o")
    kernel = PythocKernel(kern_name, obj_path, [
        input_ty, weight_ty, output_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    # Runtime parameters (Tier 2 prereq): six int32 scalars per core.
    #
    # K-blocked: [tile_m, full_ic, oc, k_block, n_k_blocks, unused].
    #   k_start is still Python-unrolled (kb*k_block) because the kernel call
    #   sits inside a Python `for kb in range(...)` loop, so each iteration's
    #   k_start is a compile-time immediate — safe and fast. k_block and
    #   n_k_blocks are in RTP anyway so the kernel body can trust them.
    # Non-K-blocked: [tile_m, 1, ic, oc, 1, 0]  (tile_w/stride/padding fixed).
    RTP_LEN = 6
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    if use_kblocking:
        init_rtp = np.array([active_tile_m, active_ic, active_oc,
                             active_k_block, active_n_k_blocks, 0],
                            dtype=np.int32)
    else:
        init_rtp = np.array([active_tile_m, 1, active_ic, active_oc, 1, 0],
                            dtype=np.int32)
    rtps = [
        Buffer(rtp_ty, name=f"rtp_{i}", initial_value=init_rtp, use_write_rtp=True)
        for i in range(n_cores)
    ]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    if use_kblocking:
        # K-blocked kernel args: tile_m, full_ic, oc, k_start, k_block, n_k_blocks
        # k_start varies per K-block (Python-unrolled)
        def core_fn(of_in, of_wt, of_out, kern, my_rtp, barrier):
            barrier.wait_for_value(1)
            tm_v = my_rtp[0]
            fic_v = my_rtp[1]
            oc_v = my_rtp[2]
            kb_v = my_rtp[3]
            nkb_v = my_rtp[4]
            for _ in range_(patches_per_core):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                # Python range = unrolled in MLIR, each iteration has constant k_start
                for kb in range(n_k_blocks):
                    elem_wt = of_wt.acquire(1)
                    kern(elem_in, elem_wt, elem_out,
                         tm_v, fic_v, oc_v, kb * k_block, kb_v, nkb_v)
                    of_wt.release(1)
                of_in.release(1)
                of_out.release(1)
            barrier.release_with_value(1)
    else:
        # Original non-K-blocked path
        kern_tile_h = tile_m
        kern_tile_w = 1
        stride = 1
        padding = 0

        def core_fn(of_in, of_wt, of_out, kern, my_rtp, barrier):
            barrier.wait_for_value(1)
            t_h = my_rtp[0]
            t_w = my_rtp[1]
            ic_v = my_rtp[2]
            oc_v = my_rtp[3]
            s_v = my_rtp[4]
            p_v = my_rtp[5]
            elem_wt = of_wt.acquire(1)
            for _ in range_(patches_per_core):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                kern(elem_in, elem_wt, elem_out,
                     t_h, t_w, ic_v, oc_v, s_v, p_v)
                of_in.release(1)
                of_out.release(1)
            of_wt.release(1)
            barrier.release_with_value(1)

    # Build per-column infrastructure.
    #
    # Super-FIFO sizing: one patch per core per super-FIFO cycle. Split/join
    # semantics require sum(sub-FIFO sizes) == super-FIFO size. The runtime
    # TAPs below produce `patches_per_core` super-FIFO cycles per call,
    # gathering the p-th patch from each core into cycle p.
    col_in_fifos = []
    col_out_fifos = []
    wt_fifos = []
    workers = []

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)

        col_in_size = cores_this_col * input_tile_size
        col_out_size = cores_this_col * output_tile_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        # Input: shim → memtile(split) → compute, depth=1
        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[input_tile_size * i for i in range(cores_this_col)],
            obj_types=[input_ty] * cores_this_col,
            names=[f"input_{col}_{i}" for i in range(cores_this_col)],
        )

        # Output: compute → memtile(join) → shim, depth=1
        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[output_tile_size * i for i in range(cores_this_col)],
            obj_types=[output_ty] * cores_this_col,
            names=[f"output_{col}_{i}" for i in range(cores_this_col)],
        )

        # Weights: broadcast to all cores in column
        # For K-blocking: depth=1, acquired/released per K-block
        # For non-K-blocking: depth=1, acquired once for all patches
        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"weights_{col}")

        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        for i in range(cores_this_col):
            global_core_idx = col * cores_per_col + i
            w = Worker(core_fn, [
                in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(), kernel,
                rtps[global_core_idx], barriers[global_core_idx],
            ], stack_size=8192)
            workers.append(w)

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(host_in_ty, host_wt_ty, host_out_ty) as (I, W, O):
        rt.start(*workers)

        # Runtime parameter write. Values may differ by generated .bin while the
        # xclbin/ObjectFifo envelope remains shared across a regime.
        _rtp_vals = [int(v) for v in init_rtp]
        def set_rtps(*rtp_bufs):
            for rb in rtp_bufs:
                rb[0] = _rtp_vals[0]; rb[1] = _rtp_vals[1]
                rb[2] = _rtp_vals[2]; rb[3] = _rtp_vals[3]
                rb[4] = _rtp_vals[4]; rb[5] = _rtp_vals[5]
        rt.inline_ops(set_rtps, rtps)
        for b in barriers:
            rt.set_barrier(b, 1)

        if use_kblocking:
            # K-blocked: send weight chunks repeatedly for each patch.
            # Host weight buffer: [chunk_0, chunk_1, ..., chunk_{n_k_blocks-1}]
            # Each patch needs all n_k_blocks chunks → repeat ppc times.
            #
            # DMA BD d0 size is limited to 1023 for multi-dim patterns,
            # so factor wt_chunk_size into (d1, d0) where d0 ≤ 1023.
            def _factor_for_dma(n, max_inner=1023):
                for inner in range(max_inner, 0, -1):
                    if n % inner == 0:
                        return n // inner, inner
                return n, 1

            wt_d1, wt_d0 = _factor_for_dma(wt_chunk_size)
            print(f"  Weight TAP: [{patches_per_core}, {n_k_blocks}, {wt_d1}, {wt_d0}]",
                  file=sys.stderr)

            for wf in wt_fifos:
                tap_wt = TensorAccessPattern(
                    (host_wt_size,),
                    offset=0,
                    sizes=[patches_per_core, n_k_blocks, wt_d1, wt_d0],
                    strides=[0, wt_chunk_size, wt_d0, 1],
                )
                rt.fill(wf.prod(), W, tap_wt)
        else:
            # Original: broadcast full weights once
            for wf in wt_fifos:
                rt.fill(wf.prod(), W)

        # Distribute input / collect output per column.
        #
        # Host layout: patches are packed per core (core_0's ppc patches,
        # then core_1's ppc patches, ...). Super-FIFO element holds one
        # patch per core, so we emit `patches_per_core` cycles per call.
        # Cycle p gathers patch p from each core: offset = c*core_in_size +
        # p*input_tile_size within the column.
        #
        # DMA BD inner dim is limited to 1023 for multi-dim transfers;
        # factor input_tile_size / output_tile_size as needed.
        def _factor_for_dma(n, max_inner=1023):
            for inner in range(max_inner, 0, -1):
                if n % inner == 0:
                    return n // inner, inner
            return n, 1

        in_d1, in_d0 = _factor_for_dma(input_tile_size)
        out_d1, out_d0 = _factor_for_dma(output_tile_size)

        for col in range(n_cols):
            cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)

            tap_in = TensorAccessPattern(
                (host_in_size,),
                offset=col * cores_per_col * core_in_size,
                sizes=[patches_per_core, cores_this_col, in_d1, in_d0],
                strides=[input_tile_size, core_in_size, in_d0, 1],
            )
            tap_out = TensorAccessPattern(
                (host_out_size,),
                offset=col * cores_per_col * core_out_size,
                sizes=[patches_per_core, cores_this_col, out_d1, out_d0],
                strides=[output_tile_size, core_out_size, out_d0, 1],
            )
            rt.fill(col_in_fifos[col].prod(), I, tap_in)
            rt.drain(col_out_fifos[col].cons(), O, tap_out,
                     wait=(col == n_cols - 1))

    return Program(dev, rt).resolve_program()


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate multicore GEMM Conv1x1 MLIR."
    )
    parser.add_argument("n_cores", nargs="?", type=int, default=32)
    parser.add_argument("tile_m", nargs="?", type=int, default=64)
    parser.add_argument("ic", nargs="?", type=int, default=128)
    parser.add_argument("oc", nargs="?", type=int, default=64)
    parser.add_argument("patches_per_core", nargs="?", type=int, default=1)
    parser.add_argument("k_block", nargs="?", type=int, default=0)
    parser.add_argument("--active-tile-m", type=int)
    parser.add_argument("--active-ic", type=int)
    parser.add_argument("--active-oc", type=int)
    parser.add_argument("--active-k-block", type=int)
    parser.add_argument("--no-fuse", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    dev = NPU2()
    args = _parse_args(sys.argv[1:])
    module = gemm_conv1x1(
        dev,
        args.tile_m,
        args.ic,
        args.oc,
        args.n_cores,
        args.patches_per_core,
        args.k_block,
        not args.no_fuse,
        active_tile_m=args.active_tile_m,
        active_ic=args.active_ic,
        active_oc=args.active_oc,
        active_k_block=args.active_k_block,
    )
    print(module)
