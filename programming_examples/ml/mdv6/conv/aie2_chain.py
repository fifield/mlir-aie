"""Level 4: 2-operator chain through memtile. Conv1x1→Conv1x1 pipeline,
intermediate data stays on-chip (no external round-trip).

Both convs use same channel dims (ch→ch) so they share the same weight type.
Both get the same weight data from the host buffer (functional test of chaining)."""
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_


def chain_conv1x1(dev, tile_h=8, tile_w=8, ch=16):
    """2-op chain: Conv1x1(ch→ch) → Conv1x1(ch→ch), intermediate through memtile."""

    spatial = tile_h * tile_w
    data_size = spatial * ch
    wt_size = ch * ch + 2 * ch  # conv weights + BN params

    print(f"Level 4: chain Conv1x1({ch}→{ch}) → Conv1x1({ch}→{ch}), "
          f"tile {tile_h}x{tile_w}", file=sys.stderr)
    print(f"  data_size={data_size}, wt_size={wt_size}", file=sys.stderr)

    # Types
    data_ty = np.ndarray[(data_size,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(wt_size,), np.dtype[np.uint16]]

    # Kernel
    kernel = Kernel("conv1x1_fused_packed_bf16", "conv_bf16.o", [
        data_ty, wt_ty, data_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    # FIFOs
    in_fifo = ObjectFifo(data_ty, name="input")
    inter_fifo = ObjectFifo(data_ty, name="inter")  # stays on-chip
    out_fifo = ObjectFifo(data_ty, name="output")
    wt1_fifo = ObjectFifo(wt_ty, name="weights1")
    wt2_fifo = ObjectFifo(wt_ty, name="weights2")

    stride = 1
    padding = 0

    # Worker 1: input → intermediate
    def core_fn1(of_in, of_wt, of_out, kern):
        elem_wt = of_wt.acquire(1)
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kern(elem_in, elem_wt, elem_out,
             tile_h, tile_w, ch, ch, stride, padding)
        of_in.release(1)
        of_out.release(1)
        of_wt.release(1)

    # Worker 2: intermediate → output
    def core_fn2(of_in, of_wt, of_out, kern):
        elem_wt = of_wt.acquire(1)
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kern(elem_in, elem_wt, elem_out,
             tile_h, tile_w, ch, ch, stride, padding)
        of_in.release(1)
        of_out.release(1)
        of_wt.release(1)

    w1 = Worker(core_fn1, [
        in_fifo.cons(), wt1_fifo.cons(), inter_fifo.prod(), kernel,
    ], stack_size=4096)
    w2 = Worker(core_fn2, [
        inter_fifo.cons(), wt2_fifo.cons(), out_fifo.prod(), kernel,
    ], stack_size=4096)

    # Runtime: both weight FIFOs filled from same host buffer
    rt = Runtime()
    with rt.sequence(data_ty, wt_ty, data_ty) as (I, W, O):
        rt.start(w1, w2)
        rt.fill(wt1_fifo.prod(), W)
        rt.fill(wt2_fifo.prod(), W)
        rt.fill(in_fifo.prod(), I)
        rt.drain(out_fifo.cons(), O, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    dev = NPU2()
    tile_h = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tile_w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ch = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    module = chain_conv1x1(dev, tile_h, tile_w, ch)
    print(module)
