#!/usr/bin/env python3
"""Phase E prototype build — OCB-unrolled rn3 ELF.

Wraps aie2_multicore_ocb.py output in a single-sub merged dispatcher
(via build_merged.build_merged) so the resulting ELF has a `@main`
entry point compatible with xrt.elf + xrt.ext.kernel(ctx, "main").
That matches how every other merged-x1 ELF in conv/build_merged is
loaded by run_tiled_mc.py.

Usage:
  python3 build_ocb.py --layer re8_rn3
  python3 build_ocb.py --layer re8_rn3_ref
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from build_merged import build_merged

_OCB_SCRIPT = os.path.join(_HERE, "aie2_multicore_ocb.py")

# (label, n_cores, tile_h, tile_w, ic, oc_block, n_ocb, ks, stride, ppc,
#  active_tile, active_ic, active_oc)
# Tile/ic/oc match the regime-active config; n_ocb covers the full OC budget.
# ppc is the EFFECTIVE patches_per_core inside the OCB-unroll ELF:
# regime_ppc × n_spatial_batches. The kernel's range_(ppc) unroll absorbs
# both the original ppc and the spatial-batch dimension into one xrt.run.
_LAYERS = {
    # ---- rn3 layers (Phase E) ----
    # re8_rn3: regime_ppc=1, 25 spatial patches fit in 32×1=32 cores per call,
    # n_spatial_batches=1 → effective ppc=1.
    "re8_rn3":     (32, 4, 4, 64, 16, 4, 3, 1, 1,  4, 64, 16),
    "re8_rn3_ref": (32, 4, 4, 64, 16, 1, 3, 1, 1,  4, 64, 16),
    # re6_rn3: regime_ppc=1, 100 spatial patches need 32×4=128 cores per call,
    # n_spatial_batches=4 → effective ppc=4. n_ocb=3 (OC=48 / oc_block=16).
    "re6_rn3":     (32, 4, 4, 48, 16, 3, 3, 1, 4,  4, 48, 16),
    "re6_rn3_ref": (32, 4, 4, 48, 16, 1, 3, 1, 4,  4, 48, 16),
    # re4_rn3: regime_ppc=4, 400 spatial patches need 32×16=512 cores per call,
    # n_spatial_batches=4 → effective ppc=16. n_ocb=1 (OC=32 = oc_block=32).
    # active_oc=32 must match the ELF's built oc_block (the dispatch will
    # use this oc_block, overriding the regime active_oc=16). Phase E shipped
    # with active_oc=16 here which produced a BO size mismatch and silent
    # numerical error (~+0.06 max_class_diff).
    "re4_rn3":     (32, 4, 4, 32, 32, 1, 3, 1, 16,  4, 32, 32),

    # ---- c3 layers (Phase F) ----
    # re8_c3: spatial 20×20 → 5×5=25 patches @ tile=4. n_spatial_batches=1,
    # effective ppc=1. OC=128 → n_ocb=8. IC=128.
    "re8_c3":      (32, 4, 4, 128, 16, 8, 3, 1, 1,  4, 128, 16),
    # re6_c3: spatial 40×40 → 10×10=100 patches @ tile=4.
    # n_spatial_batches=4 → effective ppc=4. OC=96 → n_ocb=6. IC=96.
    "re6_c3":      (32, 4, 4, 96, 16, 6, 3, 1, 4,  4, 96, 16),
    # re4_c3: spatial 80×80 → 20×20=400 patches @ tile=4.
    # n_spatial_batches=13 (=ceil(400/32)) → round to ppc=16 (covers 512).
    # OC=64 → n_ocb=4. IC=64.
    "re4_c3":      (32, 4, 4, 64, 16, 4, 3, 1, 16,  4, 64, 16),
    # elan_c3: spatial 160×160 → 40×40=1600 patches @ tile=4. Too many for
    # single-batch absorption — would need ppc=50. Skip OCB for now (already
    # uses merged_elan_c3_p4_x4 fanout dispatch).

    # ---- stride-2 aconv layers (Phase G). Regime envelope is
    # regime_r5_stride2_conv3x3 (active oc_block=8) but the test_full_model
    # callers pass larger oc_block: aconv3 caller=16, others caller=8.
    # Building at oc_block=16 keeps n_ocb manageable (avoids the iter-count
    # compilation limit hit at >40 unrolled iterations for stride=2 patches).
    # Stride=2 means patch_h = (tile-1)*2 + 3 = 9 (vs 6 for stride=1).
    #
    # aconv3: out 80×80, ic=64, oc=128. Caller oc_block=16, n_ocb=8.
    # 32×4=128 covers 100 spatial. ppc=4. 32 inner iter.
    "aconv3":      (32, 4, 4, 64, 16, 8, 3, 2, 4,  4, 64, 16),
    # aconv7: out 20×20, ic=128, oc=256. Caller oc_block=8, but build at 16
    # (override): n_ocb=16. 32×1 covers 25 spatial. ppc=1. 16 inner iter.
    "aconv7":      (32, 4, 4, 128, 16, 16, 3, 2, 1,  4, 128, 16),
    # aconv16: out 40×40, ic=64, oc=96. Caller oc_block=8, build at 16:
    # n_ocb=6. 32×4=128 covers 100 spatial. ppc=4. 24 inner iter.
    "aconv16":     (32, 4, 4, 64, 16, 6, 3, 2, 4,  4, 64, 16),
    # aconv19: out 20×20, ic=96, oc=128. Build at oc_block=8 matches regime:
    # n_ocb=16, ppc=1, 16 inner iter (built earlier, kept for reference).
    "aconv19":     (32, 4, 4, 96, 8, 16, 3, 2, 1,  4, 96, 8),
    # aconv5 (out 40×40, ic=96, oc=192) at oc_block=8 has 24 OCBs × ppc=4
    # = 96 iter → compile fails. oc_block=16 would still be 12×4=48 iter.
    # oc_block=24 doesn't align to 8-wide SIMD. Skip OCB for aconv5.
}


def _build_layer(label, cfg):
    (n_cores, tile_h, tile_w, ic, oc_block, n_ocb, ks, stride, ppc,
     active_tile, active_ic, active_oc) = cfg

    # Build the args list aie2_multicore_ocb.py expects:
    # positional: n_cores tile_h tile_w ic oc_block n_ocb ks stride ppc
    # then --active-* keyword args.
    sub_args = [
        str(n_cores), str(tile_h), str(tile_w), str(ic), str(oc_block),
        str(n_ocb), str(ks), str(stride), str(ppc),
        "--active-tile-h", str(active_tile),
        "--active-tile-w", str(active_tile),
        "--active-ic", str(active_ic),
        "--active-oc", str(active_oc),
        "--active-stride", str(stride),
        "--active-padding", "1" if ks == 3 else "0",
    ]

    sub_name = f"ocb_{label}"
    out_name = f"ocb_{label}_x1"
    return build_merged(
        out_name, [sub_name],
        kind="mc",
        sub_spec_overrides={sub_name: (_OCB_SCRIPT, sub_args)},
        share_arg_idxs={1},  # matches existing merged-x1 convention (W at arg0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=list(_LAYERS) + ["all"], default="re8_rn3")
    args = parser.parse_args()

    layers = list(_LAYERS) if args.layer == "all" else [args.layer]
    ok = fail = 0
    for label in layers:
        print(f"=== {label} ===")
        if _build_layer(label, _LAYERS[label]) is not None:
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} OK, {fail} FAIL")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
