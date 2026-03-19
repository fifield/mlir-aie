# repncsp_elan/aie2_streamed.py
#
# RepNCSPELAN with streamed weights — loads weights in 6 chunks
# to fit within 64KB tile memory.
#
# Architecture:
#   Input → Conv1 → split [x1,x2]
#           x2 → RepNCSP#1 → Conv3x3 → x3
#           x3 → RepNCSP#2 → Conv3x3 → x4
#           Concat [x1,x2,x3,x4] → Conv4 → Output
#
# Weight chunks (loaded sequentially):
#   1. Conv1 weights (~2KB)
#   2. RepNCSP#1 weights (~4KB)
#   3. Conv3x3#1 weights (~5KB)
#   4. RepNCSP#2 weights (~4KB)
#   5. Conv3x3#2 weights (~5KB)
#   6. Conv4 weights (~4KB)
#
# Max single chunk: ~5KB vs monolithic 24KB

import numpy as np
import sys

from aie.iron import (
    Kernel,
    Buffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_


def repncsp_elan_bf16_streamed(
    dev,
    height=8,
    width=8,
    in_channels=32,
    out_channels=32,
    part_channels=32,
    process_channels=None,
):
    if process_channels is None:
        process_channels = part_channels // 2

    half_part = part_channels // 2
    concat_channels = part_channels + 2 * process_channels
    rn_neck = process_channels // 2

    # Buffer sizes
    input_size = height * width * in_channels
    output_size = height * width * out_channels
    conv1_size = height * width * part_channels
    repncsp_size = height * width * process_channels
    concat_size = height * width * concat_channels
    rn_neck_size = height * width * rn_neck
    rn_concat_size = height * width * 2 * rn_neck

    # Weight chunk sizes
    conv1_wt_size = part_channels * in_channels + 4 * part_channels
    rn_wt_size = (rn_neck * half_part + 4 * rn_neck +          # conv1
                  rn_neck * rn_neck * 9 + 4 * rn_neck +        # bn conv3x3
                  rn_neck * rn_neck + 4 * rn_neck +             # bn conv1x1
                  rn_neck * rn_neck * 9 + 4 * rn_neck +        # bn conv2
                  rn_neck * half_part + 4 * rn_neck +           # conv2
                  process_channels * 2 * rn_neck + 4 * process_channels)  # conv3
    conv3x3_wt_size = process_channels * process_channels * 9 + 4 * process_channels
    # RepNCSP#2 has process_channels input instead of half_part
    rn2_wt_size = (rn_neck * process_channels + 4 * rn_neck +  # conv1
                   rn_neck * rn_neck * 9 + 4 * rn_neck +       # bn conv3x3
                   rn_neck * rn_neck + 4 * rn_neck +            # bn conv1x1
                   rn_neck * rn_neck * 9 + 4 * rn_neck +       # bn conv2
                   rn_neck * process_channels + 4 * rn_neck +   # conv2
                   process_channels * 2 * rn_neck + 4 * process_channels)  # conv3
    conv4_wt_size = out_channels * concat_channels + 4 * out_channels

    # Max weight chunk size (determines weight FIFO size)
    max_wt_size = max(conv1_wt_size, rn_wt_size, conv3x3_wt_size,
                      rn2_wt_size, conv4_wt_size)

    print(f"Generating streamed RepNCSPELAN:", file=sys.stderr)
    print(f"  Weight chunk sizes: Conv1={conv1_wt_size}, RN1={rn_wt_size}, "
          f"C3x3={conv3x3_wt_size}, RN2={rn2_wt_size}, Conv4={conv4_wt_size}", file=sys.stderr)
    print(f"  Max chunk: {max_wt_size} elements ({max_wt_size*2} bytes)", file=sys.stderr)

    # Type definitions
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    weight_chunk_ty = np.ndarray[(max_wt_size,), np.dtype[np.uint16]]
    conv1_ty = np.ndarray[(conv1_size,), np.dtype[np.uint16]]
    repncsp_ty = np.ndarray[(repncsp_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]
    rn_neck_ty = np.ndarray[(rn_neck_size,), np.dtype[np.uint16]]
    rn_concat_ty = np.ndarray[(rn_concat_size,), np.dtype[np.uint16]]

    # Host sends all 6 weight chunks concatenated, padded to max_wt_size each
    total_weight_padded_size = 6 * max_wt_size
    total_weight_ty = np.ndarray[(total_weight_padded_size,), np.dtype[np.uint16]]

    # Single monolithic kernel that processes all 6 stages sequentially
    # Each stage acquires a new weight chunk from the FIFO
    repncsp_elan_kernel = Kernel(
        "repncsp_elan_bf16",
        "repncsp_elan_bf16.o",
        [
            input_ty,        # input
            weight_chunk_ty, # weights (one chunk at a time)
            output_ty,       # output
            conv1_ty,        # conv1_output
            repncsp_ty,      # x3x4_repncsp
            repncsp_ty,      # x3x4_conv
            concat_ty,       # concat_buffer
            rn_neck_ty,      # rn_conv1
            rn_neck_ty,      # rn_bottleneck
            rn_neck_ty,      # rn_conv2
            rn_concat_ty,    # rn_concat
            rn_neck_ty,      # rn_bn_input_copy
            rn_neck_ty,      # rn_bn_temp1
            rn_neck_ty,      # rn_bn_temp2
            rn_neck_ty,      # rn_bn_temp3
            rn_neck_ty,      # rn_bn_temp4
            np.int32,        # height
            np.int32,        # width
            np.int32,        # in_channels
            np.int32,        # out_channels
            np.int32,        # part_channels
            np.int32,        # process_channels
            np.int32,        # stage (0-5)
        ],
    )

    # ObjectFifos
    of_input = ObjectFifo(input_ty, depth=1, name="input")
    of_weights = ObjectFifo(weight_chunk_ty, depth=1, name="weights")
    of_output = ObjectFifo(output_ty, depth=1, name="output")

    # Local buffers (shared between stages)
    conv1_buf = Buffer(conv1_ty, name="conv1_out")
    x3x4_repncsp_buf = Buffer(repncsp_ty, name="x3x4_repncsp")
    x3x4_conv_buf = Buffer(repncsp_ty, name="x3x4_conv")
    concat_buf = Buffer(concat_ty, name="concat")
    rn_conv1_buf = Buffer(rn_neck_ty, name="rn_conv1")
    rn_bottleneck_buf = Buffer(rn_neck_ty, name="rn_bottleneck")
    rn_conv2_buf = Buffer(rn_neck_ty, name="rn_conv2")
    rn_concat_buf = Buffer(rn_concat_ty, name="rn_concat")
    rn_bn_icopy_buf = Buffer(rn_neck_ty, name="rn_bn_icopy")
    rn_bn_t1_buf = Buffer(rn_neck_ty, name="rn_bn_t1")
    rn_bn_t2_buf = Buffer(rn_neck_ty, name="rn_bn_t2")
    rn_bn_t3_buf = Buffer(rn_neck_ty, name="rn_bn_t3")
    rn_bn_t4_buf = Buffer(rn_neck_ty, name="rn_bn_t4")

    def core_fn(of_in, of_wts, of_out, kernel,
                conv1, x3x4_rn, x3x4_c, concat,
                rn_c1, rn_bn, rn_c2, rn_cat,
                rn_icopy, rn_t1, rn_t2, rn_t3, rn_t4):
        # Acquire input once (keep for all stages)
        elem_in = of_in.acquire(1)

        # Process 6 weight chunks (one per stage)
        for stage in range_(6):
            elem_wts = of_wts.acquire(1)

            kernel(
                elem_in, elem_wts,
                # Output (only written in stage 5)
                of_out.acquire(1) if False else elem_in,  # placeholder
                conv1, x3x4_rn, x3x4_c, concat,
                rn_c1, rn_bn, rn_c2, rn_cat,
                rn_icopy, rn_t1, rn_t2, rn_t3, rn_t4,
                height, width, in_channels, out_channels,
                part_channels, process_channels,
                stage,
            )

            of_wts.release(1)

        # Release input and output
        of_in.release(1)

    # Hmm, the above won't work well because we can't conditionally acquire output.
    # Let me simplify: the kernel gets called once per weight chunk, and it knows
    # which stage it's in via the stage parameter. Input stays acquired.
    # Output: we acquire output before stage 5, or just keep it acquired the whole time.

    # Simpler approach: acquire everything upfront
    def core_fn_v2(of_in, of_wts, of_out, kernel,
                   conv1, x3x4_rn, x3x4_c, concat,
                   rn_c1, rn_bn, rn_c2, rn_cat,
                   rn_icopy, rn_t1, rn_t2, rn_t3, rn_t4):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)

        for stage in range_(6):
            elem_wts = of_wts.acquire(1)
            kernel(
                elem_in, elem_wts, elem_out,
                conv1, x3x4_rn, x3x4_c, concat,
                rn_c1, rn_bn, rn_c2, rn_cat,
                rn_icopy, rn_t1, rn_t2, rn_t3, rn_t4,
                height, width, in_channels, out_channels,
                part_channels, process_channels,
                stage,
            )
            of_wts.release(1)

        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn_v2,
        [
            of_input.cons(),
            of_weights.cons(),
            of_output.prod(),
            repncsp_elan_kernel,
            conv1_buf,
            x3x4_repncsp_buf,
            x3x4_conv_buf,
            concat_buf,
            rn_conv1_buf,
            rn_bottleneck_buf,
            rn_conv2_buf,
            rn_concat_buf,
            rn_bn_icopy_buf,
            rn_bn_t1_buf,
            rn_bn_t2_buf,
            rn_bn_t3_buf,
            rn_bn_t4_buf,
        ],
        stack_size=4096,
    )

    # Runtime: send input once, send 6 weight chunks, drain output once
    rt = Runtime()
    with rt.sequence(input_ty, total_weight_ty, output_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_input.prod(), I)
        # Send 6 weight chunks from the concatenated weight buffer
        for i in range_(6):
            rt.fill(of_weights.prod(), W)  # TODO: need offset-based fill
        rt.drain(of_output.cons(), O, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


# Parse args
try:
    dev = NPU2Col1()
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    in_channels = int(sys.argv[4])
    out_channels = int(sys.argv[5])
    part_channels = int(sys.argv[6])
    process_channels = int(sys.argv[7]) if len(sys.argv) > 7 else None
except (IndexError, ValueError) as e:
    print(f"Usage: python aie2_streamed.py npu2 H W C_in C_out part_ch [proc_ch]")
    sys.exit(1)

module = repncsp_elan_bf16_streamed(dev, height, width, in_channels, out_channels,
                                      part_channels, process_channels)
print(module)
