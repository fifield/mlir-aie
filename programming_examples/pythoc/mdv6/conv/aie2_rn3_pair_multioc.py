#!/usr/bin/env python3
"""One-dispatch multi-OC-block rn3 pair prototype.

This proves the dispatch-reduction shape: one runtime_sequence/host dispatch
computes multiple OC blocks. OC blocks are grouped in sets of four to fit the
available memtile DMA capacity. Each group receives the same input patch,
splits its packed weight slice through a memtile, and joins four ocb-sized
outputs into a block-major output region.

Output layout is block-major for bring-up:
  [oc_block][tile_h][tile_w][ocb]
not final HWC interleaved. Integration can either add a reorder or make the
join/kernel write HWC offsets later.
"""
import argparse
import os
import sys
import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker, WorkerRuntimeBarrier
from aie.iron.pythoc import PythocKernel
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern

KERNELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "build"))


def rn3_pair_multioc(dev, tile_h=8, tile_w=8, ic=48, mid=48, ocb=4, n_ocb=12, n_patches=1, group_ocb=4, single_output_join=False, output_group_ocb=None, repeat_output_drain=False, repeat_input_fill=False, finish_per_patch=False):
    input_size = (tile_h + 4) * (tile_w + 4) * ic
    w1_size = mid * ic * 9
    w2_size = ocb * mid * 9
    weight_block_size = w1_size + 2 * mid + w2_size + 2 * ocb
    all_weight_size = n_ocb * weight_block_size
    output_block_size = tile_h * tile_w * ocb
    full_output_size = n_ocb * output_block_size
    input_batch_size = n_patches * input_size
    output_batch_size = n_patches * full_output_size

    patch_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_block_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    input_batch_ty = np.ndarray[(input_batch_size,), np.dtype[np.uint16]]
    all_weight_ty = np.ndarray[(all_weight_size,), np.dtype[np.uint16]]
    output_block_ty = np.ndarray[(output_block_size,), np.dtype[np.uint16]]
    full_output_ty = np.ndarray[(output_batch_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        "rn3_pair_fused_bf16",
        os.path.join(KERNELS_DIR, "rn3_pair_fused_bf16.o"),
        [patch_ty, weight_block_ty, output_block_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    workers = []
    barriers = [WorkerRuntimeBarrier() for _ in range(n_ocb)]
    group_specs = []

    global_out_fifo = None
    global_out_joins = None
    output_joins_by_ocb = None
    output_specs = []
    if single_output_join:
        global_out_obj_ty = np.ndarray[(full_output_size,), np.dtype[np.uint16]]
        global_out_fifo = ObjectFifo(global_out_obj_ty, depth=1, name="rn3p_multi_out_all_full")
        global_out_joins = global_out_fifo.prod().join(
            offsets=[output_block_size * i for i in range(n_ocb)],
            obj_types=[output_block_ty] * n_ocb,
            depths=[1] * n_ocb,
            names=[f"rn3p_multi_out_all_{i}" for i in range(n_ocb)],
        )
    elif output_group_ocb is not None:
        output_joins_by_ocb = [None] * n_ocb
        for og, out_first_ocb in enumerate(range(0, n_ocb, output_group_ocb)):
            out_n = min(output_group_ocb, n_ocb - out_first_ocb)
            out_group_size = out_n * output_block_size
            out_group_ty = np.ndarray[(out_group_size,), np.dtype[np.uint16]]
            out_fifo = ObjectFifo(out_group_ty, depth=1, name=f"rn3p_multi_out_og{og}_full")
            joins = out_fifo.prod().join(
                offsets=[output_block_size * i for i in range(out_n)],
                obj_types=[output_block_ty] * out_n,
                depths=[1] * out_n,
                names=[f"rn3p_multi_out_og{og}_{i}" for i in range(out_n)],
            )
            for i in range(out_n):
                output_joins_by_ocb[out_first_ocb + i] = joins[i]
            output_specs.append((out_first_ocb, out_group_size, out_fifo))

    def core_fn(of_in, of_wt, of_out, kern, barrier):
        barrier.wait_for_value(1)
        elem_wt = of_wt.acquire(1)
        for _ in range(n_patches):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kern(elem_in, elem_wt, elem_out, tile_h, tile_w, ic, mid, ocb)
            of_in.release(1)
            of_out.release(1)
        of_wt.release(1)
        barrier.release_with_value(1)

    for g, first_ocb in enumerate(range(0, n_ocb, group_ocb)):
        this_n = min(group_ocb, n_ocb - first_ocb)
        group_weight_size = this_n * weight_block_size
        group_output_size = this_n * output_block_size
        group_weight_ty = np.ndarray[(group_weight_size,), np.dtype[np.uint16]]
        group_output_ty = np.ndarray[(group_output_size,), np.dtype[np.uint16]]

        in_fifo = ObjectFifo(patch_ty, depth=1, name=f"rn3p_multi_in_g{g}")

        group_w_fifo = ObjectFifo(group_weight_ty, depth=1, name=f"rn3p_multi_w_g{g}_all")
        w_splits = group_w_fifo.cons().split(
            offsets=[weight_block_size * i for i in range(this_n)],
            obj_types=[weight_block_ty] * this_n,
            depths=[1] * this_n,
            names=[f"rn3p_multi_w_g{g}_{i}" for i in range(this_n)],
        )

        if single_output_join:
            assert global_out_joins is not None
            group_out_fifo = None
            out_joins = global_out_joins[first_ocb:first_ocb + this_n]
        elif output_group_ocb is not None:
            assert output_joins_by_ocb is not None
            group_out_fifo = None
            out_joins = output_joins_by_ocb[first_ocb:first_ocb + this_n]
        else:
            group_out_fifo = ObjectFifo(group_output_ty, depth=1, name=f"rn3p_multi_out_g{g}_full")
            out_joins = group_out_fifo.prod().join(
                offsets=[output_block_size * i for i in range(this_n)],
                obj_types=[output_block_ty] * this_n,
                depths=[1] * this_n,
                names=[f"rn3p_multi_out_g{g}_{i}" for i in range(this_n)],
            )

        for i in range(this_n):
            global_i = first_ocb + i
            workers.append(Worker(
                core_fn,
                [in_fifo.cons(), w_splits[i].cons(), out_joins[i].prod(), kernel, barriers[global_i]],
                stack_size=4096,
            ))

        group_specs.append((first_ocb, group_weight_size, group_output_size, in_fifo, group_w_fifo, group_out_fifo))

    # Runtime handles are fn_args now (#3387), so the per-group fifo endpoints
    # are hoisted into lists parallel to group_specs / output_specs and indexed
    # inside the sequence body.
    group_w_prods = [spec[4].prod() for spec in group_specs]
    group_in_prods = [spec[3].prod() for spec in group_specs]
    group_out_conss = [
        spec[5].cons() if spec[5] is not None else None for spec in group_specs
    ]
    output_specs_conss = [spec[2].cons() for spec in output_specs]
    global_out_cons = (
        global_out_fifo.cons() if global_out_fifo is not None else None
    )

    def sequence(
        I,
        W,
        O,
        group_w_prods,
        group_in_prods,
        group_out_conss,
        output_specs_conss,
        global_out_cons,
    ):
        for b in barriers:
            b.set(1)
        weight_tg = TaskGroup() if finish_per_patch else None
        for gi, (first_ocb, group_weight_size, group_output_size, in_fifo, group_w_fifo, group_out_fifo) in enumerate(group_specs):
            tap_w = TensorAccessPattern(
                (all_weight_size,),
                offset=first_ocb * weight_block_size,
                sizes=[1, group_weight_size],
                strides=[0, 1],
            )
            group_w_prods[gi].fill(W, tap_w, group=weight_tg)
        if weight_tg is not None:
            weight_tg.finish()
        if repeat_output_drain:
            if single_output_join:
                assert global_out_cons is not None
                tap_o = TensorAccessPattern(
                    (output_batch_size,),
                    offset=0,
                    sizes=[n_patches, full_output_size],
                    strides=[full_output_size, 1],
                )
                global_out_cons.drain(O, tap_o, wait=True)
            elif output_group_ocb is not None:
                for oi, (out_first_ocb, out_group_size, out_fifo) in enumerate(output_specs):
                    tap_o = TensorAccessPattern(
                        (output_batch_size,),
                        offset=out_first_ocb * output_block_size,
                        sizes=[n_patches, out_group_size],
                        strides=[full_output_size, 1],
                    )
                    output_specs_conss[oi].drain(O, tap_o, wait=True)
            else:
                for gi, (first_ocb, _group_weight_size, group_output_size, _in_fifo, _group_w_fifo, group_out_fifo) in enumerate(group_specs):
                    assert group_out_conss[gi] is not None
                    tap_o = TensorAccessPattern(
                        (output_batch_size,),
                        offset=first_ocb * output_block_size,
                        sizes=[n_patches, group_output_size],
                        strides=[full_output_size, 1],
                    )
                    group_out_conss[gi].drain(O, tap_o, wait=True)
        if repeat_input_fill:
            for gi, (_first_ocb, _group_weight_size, _group_output_size, in_fifo, _group_w_fifo, _group_out_fifo) in enumerate(group_specs):
                tap_i = TensorAccessPattern(
                    (input_batch_size,),
                    offset=0,
                    sizes=[n_patches, input_size],
                    strides=[input_size, 1],
                )
                group_in_prods[gi].fill(I, tap_i)
        for p in range(n_patches):
            patch_tg = TaskGroup() if finish_per_patch else None
            for gi, (first_ocb, _group_weight_size, group_output_size, in_fifo, _group_w_fifo, group_out_fifo) in enumerate(group_specs):
                tap_i = TensorAccessPattern(
                    (input_batch_size,),
                    offset=p * input_size,
                    sizes=[1, input_size],
                    strides=[0, 1],
                )
                if not repeat_input_fill:
                    group_in_prods[gi].fill(I, tap_i, group=patch_tg)
                if not repeat_output_drain and not single_output_join and output_group_ocb is None:
                    assert group_out_conss[gi] is not None
                    tap_o = TensorAccessPattern(
                        (output_batch_size,),
                        offset=p * full_output_size + first_ocb * output_block_size,
                        sizes=[1, group_output_size],
                        strides=[0, 1],
                    )
                    group_out_conss[gi].drain(O, tap_o, group=patch_tg, wait=True)
            if not repeat_output_drain and output_group_ocb is not None:
                for oi, (out_first_ocb, out_group_size, out_fifo) in enumerate(output_specs):
                    tap_o = TensorAccessPattern(
                        (output_batch_size,),
                        offset=p * full_output_size + out_first_ocb * output_block_size,
                        sizes=[1, out_group_size],
                        strides=[0, 1],
                    )
                    output_specs_conss[oi].drain(O, tap_o, group=patch_tg, wait=True)
            if not repeat_output_drain and single_output_join:
                assert global_out_cons is not None
                tap_o = TensorAccessPattern(
                    (output_batch_size,),
                    offset=p * full_output_size,
                    sizes=[1, full_output_size],
                    strides=[0, 1],
                )
                global_out_cons.drain(O, tap_o, group=patch_tg, wait=True)
            if patch_tg is not None:
                patch_tg.finish()

    rt = Runtime(
        sequence,
        [
            input_batch_ty,
            all_weight_ty,
            full_output_ty,
            group_w_prods,
            group_in_prods,
            group_out_conss,
            output_specs_conss,
            global_out_cons,
        ],
    )

    return Program(dev, rt, workers=[*workers]).resolve_program()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("tile_h", nargs="?", type=int, default=8)
    p.add_argument("tile_w", nargs="?", type=int, default=8)
    p.add_argument("ic", nargs="?", type=int, default=48)
    p.add_argument("mid", nargs="?", type=int, default=48)
    p.add_argument("ocb", nargs="?", type=int, default=4)
    p.add_argument("n_ocb", nargs="?", type=int, default=12)
    p.add_argument("n_patches", nargs="?", type=int, default=1)
    p.add_argument("--single-output-join", action="store_true")
    p.add_argument("--output-group-ocb", type=int, default=None)
    p.add_argument("--repeat-output-drain", action="store_true")
    p.add_argument("--repeat-input-fill", action="store_true")
    p.add_argument("--finish-per-patch", action="store_true")
    args = p.parse_args(argv)
    print(rn3_pair_multioc(NPU2(), args.tile_h, args.tile_w, args.ic, args.mid, args.ocb, args.n_ocb, args.n_patches, single_output_join=args.single_output_join, output_group_ocb=args.output_group_ocb, repeat_output_drain=args.repeat_output_drain, repeat_input_fill=args.repeat_input_fill, finish_per_patch=args.finish_per_patch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
