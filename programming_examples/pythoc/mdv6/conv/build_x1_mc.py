#!/usr/bin/env python3
"""Build merged_<variant>_x1.elf for every active MC 3x3 conv variant.

Phase A.1 (mlir-aie-mi7 Phase A) target: convert standalone MC xclbins to
single-sub-device ELFs so every MC dispatch can take the xrt.elf+xrt.run
path. The 5 layers already batch-merged in run_tiled_mc._MERGED_LAYERS_ALL
keep their multi-clone ELFs — only the *remaining* variants get a 1-clone
wrapper here.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_merged import build_merged

# Variants that already have a multi-clone batch ELF in run_tiled_mc.py
# (mlir-aie-mi7.5 Phase 3). We don't override those — the larger fan-out
# is a strict win.
_ALREADY_MERGED = {
    "mc_ftconv0",
    "mc_ftconv1_p2",
    "mc_elan_c3_p4",
    "mc_aconv3",
    "mc_aconv16",
}

# Every other 3x3 MC variant that the model dispatches at steady state.
# Derived from the call sites in test_full_model_mc.py + _MC_PPC variant
# resolution in run_tiled_mc.py.
_X1_VARIANTS = [
    "mc_aconv5_p4",
    "mc_aconv7",
    "mc_aconv19",
    "mc_re4_c3_p2",
    "mc_re4_rn3_p4",
    "mc_re6_c3",
    "mc_re6_rn3",
    "mc_re8_c3",
    "mc_re8_rn3",
]


def main():
    print(f"Building {len(_X1_VARIANTS)} MC _x1 ELFs...")
    t0 = time.time()
    ok = fail = 0
    for variant in _X1_VARIANTS:
        out = f"merged_{variant.replace('mc_', '')}_x1"
        # Skip if already built (build_merged itself doesn't have a cache).
        bd = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "build_merged"))
        elf_path = os.path.join(bd, f"{out}.elf")
        if os.path.exists(elf_path):
            print(f"  {out}: already built, skipping")
            ok += 1
            continue
        print(f"=== Building {out} ===")
        # share={1} puts the wt arg first in @main, matching the multi-clone
        # ELF convention (wt, then per-clone in/out pairs). The runtime
        # dispatcher in run_tiled_mc._run_tiled_mc_inner_merged sets
        # arg0=wt regardless of n_batches, so all merged ELFs must agree on
        # this order. (For n_clones=1, --share is purely an arg reordering.)
        path = build_merged(out, [variant], share_arg_idxs={1})
        if path is not None:
            ok += 1
        else:
            fail += 1
    dt = time.time() - t0
    print(f"\nDone: {ok} OK, {fail} FAIL in {dt:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
