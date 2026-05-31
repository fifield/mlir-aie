#!/usr/bin/env python3
"""Phase C step A — rn1 pair ELFs for the three rep_elan blocks.

For each rn1 shape used by RepNCSP (re4, re6, re8), builds a TWO-sub-device
ELF where both subs share one input BO via chain_links and have independent
weights/outputs. One xrt.run replaces the two consecutive mc_*_rn1 calls
inside run_rn_mc.

Dispatcher @main signature (after chain_link aliasing):
  arg0: shared in
  arg1: sub0 wt
  arg2: sub0 out
  arg3: sub1 wt
  arg4: sub1 out

The pair ELF is named merged_gemm_t<tile_m>_ic<ic>_oc<oc>_p<ppc>_pair_x1
so run_gemm_pair_mc (in run_tiled_mc.py) can find it by shape.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gemm_conv1x1")))

from build_merged import build_merged

_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "gemm_conv1x1", "aie2_gemm_conv1x1.py"))


# (label, tile_m, ic, oc, ppc) — matches the existing _x1 ELF names so the
# pair counterpart sits next to its single-sub sibling.
_RN1_PAIRS = [
    ("re4_rn1", 256,  64, 32, 1),  # 80×80,  64→32
    ("re6_rn1", 164,  96, 48, 1),  # 40×40,  96→48
    ("re8_rn1", 104, 128, 64, 1),  # 20×20, 128→64
]


def _build_one(tile_m, ic, oc, ppc):
    sub_args = ["32", str(tile_m), str(ic), str(oc), str(ppc), "0"]
    out_name = f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_pair_x1"
    bd = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "build_merged"))
    elf_path = os.path.join(bd, f"{out_name}.elf")
    if os.path.exists(elf_path):
        print(f"  {out_name}: already built, skipping")
        return True
    sub_names = [f"gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_a",
                 f"gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_b"]
    path = build_merged(
        out_name, sub_names,
        kind="gemm",
        sub_spec_overrides={
            sub_names[0]: (_GEMM_SCRIPT, sub_args),
            sub_names[1]: (_GEMM_SCRIPT, sub_args),
        },
        chain_links=[(0, 0, 1, 0)],  # sub1.arg0 (in) == sub0.arg0 (in)
    )
    return path is not None


def main():
    print(f"Building {len(_RN1_PAIRS)} rn1 pair ELFs...")
    t0 = time.time()
    ok = fail = 0
    for label, tile_m, ic, oc, ppc in _RN1_PAIRS:
        print(f"=== {label}: tile_m={tile_m} ic={ic} oc={oc} ppc={ppc} ===")
        if _build_one(tile_m, ic, oc, ppc):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} OK, {fail} FAIL in {time.time() - t0:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
