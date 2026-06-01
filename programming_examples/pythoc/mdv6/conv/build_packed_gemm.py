#!/usr/bin/env python3
"""Build packed single-dispatch GEMM spatial-fanout ELFs.

Unlike the older cloned fanout ABI `[wt, in0, out0, in1, out1, ...]`, these
ELFs keep the kernel arg count fixed at three:

  arg0 = shared weights
  arg1 = packed input  = concat(old_x1_input_batch_0, old_x1_input_batch_1, ...)
  arg2 = packed output = concat(old_x1_output_batch_0, old_x1_output_batch_1, ...)

The GEMM generator's runtime sequence handles all per-batch DMA traffic with
batch-offset TensorAccessPatterns.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_merged import build_merged, _resolve_build_dir
from gemm_configs import (
    MODEL_LAYERS_1x1,
    choose_k_block,
    compute_ppc_kblocked,
    compute_ppc,
)

N_CORES = 32
_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "gemm_conv1x1", "aie2_gemm_conv1x1.py")
)


def packed_gemm_elf_name(tile_m, ic, oc, k_block, ppc, n_batches):
    kb_str = f"kb{k_block}_" if k_block > 0 else ""
    return f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_{kb_str}p{ppc}_x{n_batches}_packed"


def packed_host_sizes(n_batches, host_in_size, host_out_size):
    return n_batches * host_in_size, n_batches * host_out_size


def derive_layer_shape(name, H, W, IC, OC):
    """Return (tile_m, k_block, ppc, n_batches) mirroring runtime dispatch."""
    M = H * W
    k_block, tile_m = choose_k_block(IC, OC, M)
    if tile_m < 4:
        return None
    tile_m = min(tile_m, 256)
    if k_block > 0:
        ppc = compute_ppc_kblocked(M, tile_m, IC, OC, k_block)
    else:
        ppc = compute_ppc(M, tile_m, IC, OC)
    pixels_per_call = N_CORES * tile_m * ppc
    n_batches = int(math.ceil(M / pixels_per_call))
    return tile_m, k_block, ppc, n_batches


def iter_layer_shapes(selected_layer=None):
    for name, H, W, IC, OC in MODEL_LAYERS_1x1:
        if selected_layer is not None and name != selected_layer:
            continue
        shape = derive_layer_shape(name, H, W, IC, OC)
        if shape is None:
            print(f"SKIP {name}: does not fit in L1")
            continue
        tile_m, k_block, ppc, n_batches = shape
        yield name, H, W, IC, OC, tile_m, k_block, ppc, n_batches


def build_one(tile_m, ic, oc, k_block, ppc, n_batches, force=False):
    elf_base = packed_gemm_elf_name(tile_m, ic, oc, k_block, ppc, n_batches)
    elf_path = os.path.join(_resolve_build_dir(), f"{elf_base}.elf")
    if os.path.exists(elf_path) and not force:
        print(f"  {elf_base}: already built, skipping")
        return elf_path

    kb_arg = str(k_block)
    sub_label = elf_base[len("merged_"):]  # human-readable label
    sub_args = [
        str(N_CORES), str(tile_m), str(ic), str(oc), str(ppc), kb_arg,
        "--spatial-batches", str(n_batches),
    ]
    return build_merged(
        elf_base,
        [sub_label],
        share_arg_idxs={1},  # dispatcher ABI: [W, packed_in, packed_out]
        kind="gemm",
        sub_spec_overrides={sub_label: (_GEMM_SCRIPT, sub_args)},
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layer", help="build only one MODEL_LAYERS_1x1 layer name")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true", help="list packed GEMM targets without building")
    args = p.parse_args(argv)

    # Dedup by exact physical packed shape.
    shape_to_layers = {}
    for row in iter_layer_shapes(args.layer):
        name, _H, _W, IC, OC, tile_m, k_block, ppc, n_batches = row
        key = (tile_m, IC, OC, k_block, ppc, n_batches)
        shape_to_layers.setdefault(key, []).append(name)

    print(f"Packed GEMM unique shapes: {len(shape_to_layers)}")
    for (tile_m, ic, oc, k_block, ppc, n_batches), names in shape_to_layers.items():
        print(f"  {packed_gemm_elf_name(tile_m, ic, oc, k_block, ppc, n_batches)}: {names}")

    if args.list:
        return 0

    t0 = time.time()
    ok = fail = 0
    for key, names in shape_to_layers.items():
        tile_m, ic, oc, k_block, ppc, n_batches = key
        name = packed_gemm_elf_name(tile_m, ic, oc, k_block, ppc, n_batches)
        print(f"=== Building {name} ({len(names)} layers: {names}) ===")
        try:
            path = build_one(tile_m, ic, oc, k_block, ppc, n_batches, force=args.force)
        except Exception as e:
            print(f"  {name}: build failed: {e}")
            path = None
        if path is None:
            fail += 1
        else:
            ok += 1
    print(f"\nDone: {ok} OK, {fail} FAIL in {time.time() - t0:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
