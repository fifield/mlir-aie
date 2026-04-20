"""Multi-core tiled Conv+BN+SiLU. N cores process N patches in parallel."""
import numpy as np
import sys

from aie.iron import (
    Kernel, ObjectFifo, Program, Runtime, Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_


def conv_tiled_fused_multicore_bf16(dev, input_height, input_width, input_channels,
                                     output_channels, kernel_size=3, stride=1, padding=1,
                                     tile_h=16, tile_w=16, out_chan_block=16,
                                     n_cores=4, patches_per_core=1):
    """Multi-core tiled Conv+BN+SiLU. N cores each process patches_per_core tiles."""
    if kernel_size not in [1, 3]:
        raise ValueError("Only 1x1 and 3x3 kernels supported")
    if kernel_size == 1:
        padding = 0

    out_h = (input_height + 2 * padding - kernel_size) // stride + 1
    out_w = (input_width + 2 * padding - kernel_size) // stride + 1

    patch_h = (tile_h - 1) * stride + kernel_size
    patch_w = (tile_w - 1) * stride + kernel_size

    patch_size_raw = patch_h * patch_w * input_channels
    patch_size = patch_size_raw + (patch_size_raw % 2)
    conv_weight_size = out_chan_block * input_channels * kernel_size * kernel_size
    bn_size = out_chan_block
    weight_block_size = conv_weight_size + 2 * bn_size
    output_tile_size = tile_h * tile_w * out_chan_block

    total_patches = patches_per_core * n_cores

    print(f"Generating multi-core tiled fused conv ({n_cores} cores):", file=sys.stderr)
    print(f"  {input_height}×{input_width}×{input_channels} → {out_h}×{out_w}×{output_channels}", file=sys.stderr)
    print(f"  tile={tile_h}×{tile_w}, oc={out_chan_block}, patches/core={patches_per_core}", file=sys.stderr)
    print(f"  total patches per invocation: {total_patches}", file=sys.stderr)

    patch_ty = np.ndarray[(patch_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]

    # Each core's input/output buffer
    core_input_ty = np.ndarray[(patches_per_core * patch_size,), np.dtype[np.uint16]]
    core_output_ty = np.ndarray[(patches_per_core * output_tile_size,), np.dtype[np.uint16]]

    # Host buffers: all cores' data concatenated
    host_input_ty = np.ndarray[(total_patches * patch_size,), np.dtype[np.uint16]]
    host_output_ty = np.ndarray[(total_patches * output_tile_size,), np.dtype[np.uint16]]

    if kernel_size == 3:
        kern_name = "conv3x3_fused_packed_bf16"
    else:
        kern_name = "conv1x1_fused_packed_bf16"

    kernel = Kernel(kern_name, "rep_elan_bf16.o", [
        patch_ty, weight_ty, output_tile_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    # Per-core FIFOs (separate weight FIFO per core for placement)
    workers = []
    of_inputs = []
    of_weights_list = []
    of_outputs = []

    for c in range(n_cores):
        of_in = ObjectFifo(patch_ty, depth=1, name=f"input_{c}")
        of_wt = ObjectFifo(weight_ty, depth=1, name=f"weights_{c}")
        of_out = ObjectFifo(output_tile_ty, depth=1, name=f"output_{c}")
        of_inputs.append(of_in)
        of_weights_list.append(of_wt)
        of_outputs.append(of_out)

        def make_core_fn(ppc):
            def core_fn(of_in_h, of_wts_h, of_out_h, kern):
                elem_wts = of_wts_h.acquire(1)
                for _ in range_(ppc):
                    elem_in = of_in_h.acquire(1)
                    elem_out = of_out_h.acquire(1)
                    kern(elem_in, elem_wts, elem_out,
                         tile_h, tile_w, input_channels, out_chan_block, stride, padding)
                    of_in_h.release(1)
                    of_out_h.release(1)
                of_wts_h.release(1)
            return core_fn

        w = Worker(make_core_fn(patches_per_core), [
            of_in.cons(), of_wt.cons(), of_out.prod(), kernel,
        ], stack_size=4096)
        workers.append(w)

    # Runtime: send patches to each core, drain outputs
    rt = Runtime()
    with rt.sequence(host_input_ty, weight_ty, host_output_ty) as (I, W, O):
        for w in workers:
            rt.start(w)

        # Fill each core's weights and input patches
        for c in range(n_cores):
            rt.fill(of_weights_list[c].prod(), W)
            rt.fill(of_inputs[c].prod(), I)

        # Drain each core's output
        for c in range(n_cores):
            rt.drain(of_outputs[c].cons(), O, wait=(c == n_cores - 1))

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    dev = NPU2()  # Full 8-column 6-row array
    n_cores = int(sys.argv[10]) if len(sys.argv) > 10 else 4
    H = int(sys.argv[2]); W = int(sys.argv[3])
    ic = int(sys.argv[4]); oc = int(sys.argv[5])
    ks = int(sys.argv[6]); th = int(sys.argv[7]); tw = int(sys.argv[8])
    ocb = int(sys.argv[9])
    n_cores = int(sys.argv[10]) if len(sys.argv) > 10 else 4
    ppc = int(sys.argv[11]) if len(sys.argv) > 11 else 1
    stride = int(sys.argv[12]) if len(sys.argv) > 12 else 1

    module = conv_tiled_fused_multicore_bf16(dev, H, W, ic, oc, ks, stride,
                                              1 if ks == 3 else 0,
                                              th, tw, ocb, n_cores, ppc)
    print(module)
