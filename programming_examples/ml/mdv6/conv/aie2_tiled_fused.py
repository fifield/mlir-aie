"""Tiled Conv+BN+SiLU fused kernel for large images. Supports stride=1 and stride=2."""
import numpy as np
import sys

from aie.iron import (
    Kernel, ObjectFifo, Program, Runtime, Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_


def conv_tiled_fused_bf16(dev, input_height, input_width, input_channels,
                           output_channels, kernel_size=3, stride=1, padding=1,
                           tile_h=16, tile_w=16, out_chan_block=16, n_patches=1):
    """Tiled Conv+BN+SiLU with packed weights [conv_wts, bn_w, bn_b]."""
    if kernel_size not in [1, 3]:
        raise ValueError("Only 1x1 and 3x3 kernels supported")
    if kernel_size == 1:
        padding = 0

    out_h = (input_height + 2 * padding - kernel_size) // stride + 1
    out_w = (input_width + 2 * padding - kernel_size) // stride + 1

    tiles_h = (out_h + tile_h - 1) // tile_h
    tiles_w = (out_w + tile_w - 1) // tile_w

    patch_h = (tile_h - 1) * stride + kernel_size
    patch_w = (tile_w - 1) * stride + kernel_size

    patch_size_raw = patch_h * patch_w * input_channels
    patch_size = patch_size_raw + (patch_size_raw % 2)  # 4-byte align
    conv_weight_size = out_chan_block * input_channels * kernel_size * kernel_size
    bn_size = out_chan_block  # fused bn_w and bn_b
    weight_block_size = conv_weight_size + 2 * bn_size
    output_tile_size = tile_h * tile_w * out_chan_block

    print(f"Generating tiled fused conv:", file=sys.stderr)
    print(f"  {input_height}×{input_width}×{input_channels} → {out_h}×{out_w}×{output_channels} (s={stride}, k={kernel_size})", file=sys.stderr)
    print(f"  tile={tile_h}×{tile_w}, oc_block={out_chan_block}, n_patches={n_patches}", file=sys.stderr)
    print(f"  patch={patch_h}×{patch_w}={patch_size} elems, wt={weight_block_size}, out_tile={output_tile_size}", file=sys.stderr)
    mem = (n_patches * patch_size + weight_block_size + n_patches * output_tile_size + 2048) * 2
    print(f"  est memory: {mem/1024:.1f} KB", file=sys.stderr)

    patch_ty = np.ndarray[(patch_size,), np.dtype[np.uint16]]
    input_buffer_ty = np.ndarray[(n_patches * patch_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]
    output_buffer_ty = np.ndarray[(n_patches * output_tile_size,), np.dtype[np.uint16]]

    if kernel_size == 3:
        kern_name = "conv3x3_fused_packed_bf16"
    else:
        kern_name = "conv1x1_fused_packed_bf16"

    # The fused packed kernel: (input, packed_weights, output, h, w, ic, oc, stride, padding)
    kernel = Kernel(kern_name, "rep_elan_bf16.o", [
        patch_ty, weight_ty, output_tile_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    of_input = ObjectFifo(patch_ty, depth=1, name="input_patch")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights")
    of_output = ObjectFifo(output_tile_ty, depth=1, name="output_tile")

    def core_fn(of_in, of_wts, of_out, kern):
        elem_wts = of_wts.acquire(1)
        for _ in range_(n_patches):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kern(elem_in, elem_wts, elem_out,
                 tile_h, tile_w, input_channels, out_chan_block, stride, padding)
            of_in.release(1)
            of_out.release(1)
        of_wts.release(1)

    worker = Worker(core_fn, [
        of_input.cons(), of_weights.cons(), of_output.prod(), kernel,
    ], stack_size=4096)

    rt = Runtime()
    with rt.sequence(input_buffer_ty, weight_ty, output_buffer_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_input.prod(), I)
        rt.fill(of_weights.prod(), W)
        rt.drain(of_output.cons(), O, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    dev = NPU2Col1()
    H = int(sys.argv[2]); W = int(sys.argv[3])
    ic = int(sys.argv[4]); oc = int(sys.argv[5])
    ks = int(sys.argv[6]); th = int(sys.argv[7]); tw = int(sys.argv[8])
    ocb = int(sys.argv[9]); np_ = int(sys.argv[10]) if len(sys.argv) > 10 else 1
    stride = int(sys.argv[11]) if len(sys.argv) > 11 else 1

    module = conv_tiled_fused_bf16(dev, H, W, ic, oc, ks, stride, 1 if ks == 3 else 0,
                                    th, tw, ocb, np_)
    print(module)
