#!/usr/bin/env python3
"""Phase C step A — proof-of-concept rn1 pair ELF.

Builds merged_gemm_re6_rn1_pair_x1.elf with TWO sub-devices of the
gemm_t164_ic96_oc48_p1 (mc_re6_rn1) kernel sharing one input BO via
chain_links. Different weights and outputs per sub.

Dispatcher @main signature (after chain_link aliasing):
  arg0: shared in     (memref<154880xui16>  = 32 cores × 164 tile_m × 96 IC, half-word u16 == bf16)
  arg1: sub0 wt       (memref<4704xui16>    = 96*48 + 2*48 == 96 oc-blocked weights + BN)
  arg2: sub0 out      (memref<252352xui16>  = 32 cores × 164 tile_m × 48 OC, padded)
  arg3: sub1 wt       (same shape as arg1)
  arg4: sub1 out      (same shape as arg2)

One xrt.run replaces the two consecutive mc_re6_rn1 dispatches inside
run_rn_mc — re6 alone has 3 calls to run_rn_mc per frame × 2 rn1 each
= 6 launches → 3 launches.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gemm_conv1x1")))

from build_merged import build_merged

_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "gemm_conv1x1", "aie2_gemm_conv1x1.py"))


def main():
    # gemm_re6_rn1: n_cores=32, tile_m=164, ic=96, oc=48, ppc=1, k_block=0
    sub_args = ["32", "164", "96", "48", "1", "0"]
    out_name = "merged_gemm_re6_rn1_pair_x1"
    sub_names = ["gemm_t164_ic96_oc48_p1_a", "gemm_t164_ic96_oc48_p1_b"]
    # Both subs use the same generator + args. chain_links shares sub1.in
    # (arg 0) with sub0.in (arg 0). No share_arg_idxs — we want unique
    # weights & outputs per sub.
    path = build_merged(
        out_name,
        sub_names,
        kind="gemm",
        sub_spec_overrides={
            sub_names[0]: (_GEMM_SCRIPT, sub_args),
            sub_names[1]: (_GEMM_SCRIPT, sub_args),
        },
        chain_links=[(0, 0, 1, 0)],  # sub1.arg0 (in) == sub0.arg0 (in)
    )
    return 0 if path is not None else 1


if __name__ == "__main__":
    sys.exit(main())
