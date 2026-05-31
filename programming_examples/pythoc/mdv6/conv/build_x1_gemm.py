#!/usr/bin/env python3
"""Build merged_<layer>_gemm_x1.elf for every active GEMM 1x1 conv layer.

Phase A.2 (mlir-aie-mi7 Phase A): convert the GEMM standalone xclbin path to
single-sub-device ELFs so every GEMM dispatch can route through xrt.elf/xrt.run.

The shape per layer is derived from the model's MODEL_LAYERS_1x1 list, using
the same `choose_k_block` / `compute_ppc_kblocked` logic the runtime would.
Each ELF is built with `--share 1` so the dispatcher arg order is
(wt, in, out) — matching the host-side `set_arg(0, wt_bo)` convention shared
with MC merged ELFs (see feedback_merged_elf_arg_order.md).

Output naming: merged_<layer>_gemm_x1.elf
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gemm_conv1x1")))

from build_merged import build_merged
from build_gemm_conv1x1 import (  # noqa: E402
    MODEL_LAYERS_1x1,
    choose_k_block,
    compute_ppc_kblocked,
    compute_ppc,
)

_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "gemm_conv1x1", "aie2_gemm_conv1x1.py")
)


def _derive_layer_shape(name, H, W, IC, OC):
    """Returns (tile_m, k_block, ppc) for this layer, mirroring runtime dispatch."""
    M = H * W
    k_block, tile_m = choose_k_block(IC, OC, M)
    if tile_m < 4:
        return None
    tile_m = min(tile_m, 256)
    if k_block > 0:
        ppc = compute_ppc_kblocked(M, tile_m, IC, OC, k_block)
    else:
        ppc = compute_ppc(M, tile_m, IC, OC)
    return (tile_m, k_block, ppc)


def main():
    layers = []
    for name, H, W, IC, OC in MODEL_LAYERS_1x1:
        shape = _derive_layer_shape(name, H, W, IC, OC)
        if shape is None:
            print(f"SKIP {name}: does not fit in L1")
            continue
        tile_m, k_block, ppc = shape
        layers.append((name, IC, OC, tile_m, k_block, ppc))

    # Dedup by shape so cousin layers (re6_c1 + re18_c1 are not equivalent;
    # they have different IC) build only the configs they need. We key by
    # (tile_m, ic, oc, k_block, ppc) — same shape → same ELF can serve.
    print(f"Total layers: {len(layers)}")
    shape_keys = {}
    for name, IC, OC, tile_m, k_block, ppc in layers:
        key = (tile_m, IC, OC, k_block, ppc)
        shape_keys.setdefault(key, []).append(name)

    print(f"Unique GEMM shapes: {len(shape_keys)}")
    for key, names in shape_keys.items():
        tile_m, ic, oc, k_block, ppc = key
        kb_str = f"kb{k_block}_" if k_block > 0 else ""
        print(f"  gemm_t{tile_m}_ic{ic}_oc{oc}_{kb_str}p{ppc}: {names}")

    bd = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_merged"))

    t0 = time.time()
    ok = fail = 0
    for (tile_m, ic, oc, k_block, ppc), names in shape_keys.items():
        # Build name follows the standalone xclbin convention so it's recognisable.
        kb_str = f"kb{k_block}_" if k_block > 0 else ""
        elf_base = f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_{kb_str}p{ppc}_x1"
        elf_path = os.path.join(bd, f"{elf_base}.elf")
        if os.path.exists(elf_path):
            print(f"  {elf_base}: already built, skipping")
            ok += 1
            continue
        print(f"=== Building {elf_base} ({len(names)} layers: {names}) ===")
        # GEMM uses the gemm_conv1x1 generator. Pass cmd-line args directly via
        # sub_spec_overrides since GEMM shapes aren't in a fixed CONFIGS list.
        sub_label = elf_base[len("merged_"):-len("_x1")]  # human-readable label
        sub_args = [str(32), str(tile_m), str(ic), str(oc), str(ppc), str(k_block)]
        try:
            path = build_merged(
                elf_base,
                [sub_label],
                share_arg_idxs={1},
                kind="gemm",
                sub_spec_overrides={sub_label: (_GEMM_SCRIPT, sub_args)},
            )
        except Exception as e:
            print(f"  {elf_base}: build failed: {e}")
            path = None
        if path is not None:
            ok += 1
        else:
            fail += 1

    dt = time.time() - t0
    print(f"\nDone: {ok} OK, {fail} FAIL in {dt:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
