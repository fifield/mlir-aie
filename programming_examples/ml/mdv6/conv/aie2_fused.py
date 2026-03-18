"""Conv+BN+SiLU fused kernel for IRON. Supports 1x1 and 3x3."""
import numpy as np
import sys

from aie.iron import (
    Kernel, Buffer, ObjectFifo, Program, Runtime, Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


def conv_fused_bf16(dev, height, width, in_ch, out_ch, kernel_size, stride=1, padding=1):
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    input_size = height * width * in_ch
    weight_size = out_ch * in_ch * kernel_size * kernel_size
    bn_size = out_ch  # fused bn_weight and bn_bias each
    output_size = out_h * out_w * out_ch

    # Total weights: conv weights + bn_weight + bn_bias
    total_weight_size = weight_size + 2 * bn_size

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    conv_weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    bn_ty = np.ndarray[(bn_size,), np.dtype[np.uint16]]

    if kernel_size == 3:
        kern_name = "conv3x3_fused_bf16"
    elif kernel_size == 1:
        kern_name = "conv1x1_fused_bf16"
    else:
        raise ValueError(f"Kernel size {kernel_size} not supported")

    # Kernel signature: input, weights, bn_weight, bn_bias, output, dims...
    if kernel_size == 3:
        kernel_args = [input_ty, conv_weight_ty, bn_ty, bn_ty, output_ty,
                       np.int32, np.int32, np.int32, np.int32, np.int32, np.int32]
    else:
        kernel_args = [input_ty, conv_weight_ty, bn_ty, bn_ty, output_ty,
                       np.int32, np.int32, np.int32, np.int32]

    conv_kernel = Kernel(kern_name, "conv_bf16.o", kernel_args)

    of_input = ObjectFifo(input_ty, depth=1, name="input")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights")
    of_output = ObjectFifo(output_ty, depth=1, name="output")

    # BN params are extracted from the weight buffer into local Buffers
    bn_w_buf = Buffer(bn_ty, name="bn_weight")
    bn_b_buf = Buffer(bn_ty, name="bn_bias")

    def core_fn(of_in, of_wts, of_out, kernel, bn_w, bn_b):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        # The weight FIFO contains [conv_weights, bn_weight, bn_bias]
        # The kernel takes them separately, but we pass the full weight buffer
        # and let pointer arithmetic in the kernel handle offsets.
        # Actually, the kernel expects separate pointers. We need to split.
        # For now, pass the full weight buffer as conv weights (kernel reads
        # only weight_size elements) and BN buffers separately.
        # The BN params need to be copied from the weight FIFO to local buffers.
        # This requires a memcpy kernel or restructuring.

        # Simpler: pack conv_weights, bn_weight, bn_bias sequentially in weight FIFO.
        # Kernel: we can't split a memref in IRON.
        # Alternative: pass entire weight buffer to kernel and have kernel offset internally.

        # ACTUALLY: Let's just pass the whole weight buffer as "weights" and make the
        # C kernel extract bn params via pointer arithmetic from the same buffer.
        # This requires modifying the C kernel... or just use 3 separate ObjectFifos.

        # Let's use the simplest approach: 3 input FIFOs (data, conv_weights, bn_params)
        pass

    # REDESIGN: Use separate ObjectFifos for conv weights and BN params
    # This avoids needing to split a memref in IRON.
    pass


# Actually, simpler approach: modify the C kernel to take a single packed weight buffer
# and extract conv weights, bn_weight, bn_bias via pointer offsets internally.
# This is what repconv/repncsp already do.

def conv_fused_packed_bf16(dev, height, width, in_ch, out_ch, kernel_size, stride=1, padding=1):
    """Conv+BN+SiLU with all weights packed in a single buffer."""
    if kernel_size == 1:
        padding = 0

    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    input_size = height * width * in_ch
    weight_size = out_ch * in_ch * kernel_size * kernel_size
    bn_size = out_ch
    total_weight_size = weight_size + 2 * bn_size  # [conv_wts, bn_w, bn_b]
    output_size = out_h * out_w * out_ch

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    # Use the existing fused kernel but with a wrapper that unpacks
    if kernel_size == 3:
        kern_name = "conv3x3_fused_packed_bf16"
    else:
        kern_name = "conv1x1_fused_packed_bf16"

    conv_kernel = Kernel(
        kern_name, "conv_bf16.o",
        [input_ty, weight_ty, output_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    of_input = ObjectFifo(input_ty, depth=1, name="input")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights")
    of_output = ObjectFifo(output_ty, depth=1, name="output")

    def core_fn(of_in, of_wts, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        if kernel_size == 3:
            kernel(elem_in, elem_wts, elem_out,
                   height, width, in_ch, out_ch, stride, padding)
        else:
            kernel(elem_in, elem_wts, elem_out,
                   height, width, in_ch, out_ch, 1, 0)

        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [
        of_input.cons(), of_weights.cons(), of_output.prod(), conv_kernel,
    ], stack_size=4096)

    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_input.prod(), I)
        rt.fill(of_weights.prod(), W)
        rt.drain(of_output.cons(), O, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    dev = NPU2Col1()
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    in_ch = int(sys.argv[4])
    out_ch = int(sys.argv[5])
    kernel_size = int(sys.argv[6])
    stride = int(sys.argv[7]) if len(sys.argv) > 7 else (2 if kernel_size == 3 else 1)
    padding = int(sys.argv[8]) if len(sys.argv) > 8 else (1 if kernel_size == 3 else 0)

    module = conv_fused_packed_bf16(dev, height, width, in_ch, out_ch, kernel_size, stride, padding)
    print(module)
