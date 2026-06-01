#!/usr/bin/env python3
"""Build merged ELFs for every active MC 3x3 conv variant.

Phase A.1 (mlir-aie-mi7 Phase A) target: every MC dispatch takes the
xrt.elf+xrt.run path. This script produces:

  1. Single-clone wrappers (merged_<variant>_x1.elf) for layers that
     don't benefit from batch fanout — most variants.
  2. Multi-clone batch-fanout ELFs (merged_<variant>_xN.elf) for the
     3 large-tile / large-fanout layers in _FANOUT below.

The Phase E/F/G OCB-unrolled ELFs (ocb_*) are built separately by
build_ocb.py and take precedence over the x1 ELFs at dispatch time.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_merged import build_merged, _resolve_build_dir

# Single-clone (x1) variants — every 3x3 MC variant the model dispatches
# at steady state, derived from call sites in test_full_model_mc.py +
# _MC_PPC variant resolution in run_tiled_mc.py. Most are also covered
# by OCB-unrolled ELFs (Phase E/F/G) which take precedence; the x1 ELFs
# remain as the fallback path for MERGED_OCB=0 testing.
_X1_VARIANTS = [
    "mc_aconv3",
    "mc_aconv5_p4",
    "mc_aconv7",
    "mc_aconv16",
    "mc_aconv19",
    "mc_re4_c3_p2",
    "mc_re4_rn3_p4",
    "mc_re6_c3",
    "mc_re6_rn3",
    "mc_re8_c3",
    "mc_re8_rn3",
]

# Multi-clone batch-fanout ELFs. `(variant, n_clones)` — N sub-clones of
# the same kernel in one dispatcher collapse N spatial batches per
# layer-call into one xrt.run.
_FANOUT = [
    ("mc_ftconv0",     8),   # merged_ftconv0_x8
    ("mc_ftconv1_p2",  4),   # merged_ftconv1_p2_x4
    ("mc_elan_c3_p4",  4),   # merged_elan_c3_p4_x4
]


def _build_one(variant, n_clones):
    if n_clones == 1:
        out = f"merged_{variant.replace('mc_', '')}_x1"
        sub_names = [variant]
    else:
        out = f"merged_{variant.replace('mc_', '')}_x{n_clones}"
        sub_names = [variant] * n_clones
    # Skip check must look at the actual build target (which honors
    # MDV6_BUILD_DIR), not the source tree — otherwise CI/lit runs that
    # set MDV6_BUILD_DIR=. silently skip builds that the test then can't
    # find in the per-test working directory.
    elf_path = os.path.join(_resolve_build_dir(), f"{out}.elf")
    if os.path.exists(elf_path):
        print(f"  {out}: already built, skipping")
        return True
    print(f"=== Building {out} ({n_clones} clone{'s' if n_clones > 1 else ''}) ===")
    # share_arg_idxs={1} promotes the per-sub wt arg (index 1 in the sub's
    # (in, wt, out) tuple) to a single shared dispatcher arg. All clones
    # then receive the same wt BO; per-clone in/out args follow. The runtime
    # dispatcher in _run_tiled_mc_inner_merged matches this convention.
    path = build_merged(out, sub_names, share_arg_idxs={1})
    return path is not None


def main():
    targets = [(v, 1) for v in _X1_VARIANTS] + _FANOUT
    print(f"Building {len(targets)} MC merged ELFs...")
    t0 = time.time()
    ok = fail = 0
    for variant, n_clones in targets:
        if _build_one(variant, n_clones):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} OK, {fail} FAIL in {time.time() - t0:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
