#!/usr/bin/env python3
"""B2a — re8-shape device-resident GEMM->GEMM dispatcher merge.

Demonstrates the CONTEXT-NEGATIVE dispatcher merge on a REAL re8 GEMM shape
(the rnm GEMM: 1x1 conv, IC=128 -> OC=128, tile_m=44, ppc=1, 32 cores). Two
back-to-back 1x1 GEMMs of this shape are packed as TWO sub-devices inside ONE
merged ELF, wired producer->consumer via a chain_link so the intermediate stays
DEVICE-RESIDENT across an on-device PDI swap:

    chain_links=[(0, 2, 1, 0)]   # sub1.arg0 (in)  <-  sub0.arg2 (out)

For this shape producer.out and consumer.in are the SAME MLIR memref type
(memref<180224xui16>, = 32 slots * tile_m=44 * 128 ch), so build_merged's
chain_link type-check passes with no on-device reformat.

CONTEXT MODEL (the headline B2a proof):
  - BEFORE: two separate single-sub merged ELFs => 2 xrt.hw_context, and the
    host issues 2 xrt.run dispatches (one per ELF), bouncing the intermediate
    through host DDR between them.
  - AFTER:  one merged ELF with 2 aiex.configure sub-devices => 1 xrt.hw_context
    (PoC-1 proved N sub-devices = 1 context), and the host issues 1 xrt.run
    (the on-device PDI swap, ~39 us, replaces the 2nd host dispatch + the host
    bounce of the intermediate).

So fusing two back-to-back operators by making them sub-devices inside ONE
merged ELF dispatcher REDUCES the live hw_context count by one and removes one
host dispatch/frame. This is the construct the milestone wants demonstrated on
a real re8 shape; the build helper is shared with the wired re8 path
(fuse_re8_runner) behind MDV6_FUSE_RE8.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gemm_conv1x1")))

from build_merged import build_merged, _resolve_build_dir

_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "gemm_conv1x1", "aie2_gemm_conv1x1.py"))

# re8 rnm GEMM shape: 1x1 conv IC=128 -> OC=128, tile_m=44, ppc=1, 32 cores.
# (matches the deployed merged_gemm_t44_ic128_oc128_p1_x1 ELF.)
N_CORES = 32
TILE_M = 44
IC = 128
OC = 128
PPC = 1
K_BLOCK = 0  # full IC in L1; matches the deployed _x1 ELF kernel

# Per-sub GEMM CLI args: [n_cores, tile_m, ic, oc, ppc, k_block]
_SUB_ARGS = [str(N_CORES), str(TILE_M), str(IC), str(OC), str(PPC), str(K_BLOCK)]

# Merged producer->consumer ELF (one context, device-resident intermediate).
STITCH_ELF = "re8_gemm_stitch_t44_ic128_oc128_p1_merged"
# Standalone single-sub ELFs used to build the 2-dispatch / 2-context baseline.
PROD_ELF = "re8_gemm_solo_t44_ic128_oc128_p1_a"
CONS_ELF = "re8_gemm_solo_t44_ic128_oc128_p1_b"


def build_stitch_elf(build_dir=None):
    """Merged producer->consumer ELF (1 context, device-resident intermediate)."""
    bd = build_dir or _resolve_build_dir()
    elf = os.path.join(bd, f"{STITCH_ELF}.elf")
    if os.path.exists(elf):
        return elf
    sub_names = ["re8_stitch_prod", "re8_stitch_cons"]
    return build_merged(
        STITCH_ELF, sub_names, kind="gemm", build_dir=bd,
        sub_spec_overrides={
            sub_names[0]: (_GEMM_SCRIPT, _SUB_ARGS),
            sub_names[1]: (_GEMM_SCRIPT, _SUB_ARGS),
        },
        # PRODUCER->CONSUMER: sub1.arg0 (in) <- sub0.arg2 (out).
        chain_links=[(0, 2, 1, 0)],
    )


def build_single(out_name, sub_label, build_dir=None):
    bd = build_dir or _resolve_build_dir()
    elf = os.path.join(bd, f"{out_name}.elf")
    if os.path.exists(elf):
        return elf
    return build_merged(
        out_name, [sub_label], kind="gemm", build_dir=bd,
        sub_spec_overrides={sub_label: (_GEMM_SCRIPT, _SUB_ARGS)},
    )


def build_all(build_dir=None):
    print("=== B2a re8 GEMM stitch: building merged + baseline ELFs ===")
    s = build_stitch_elf(build_dir)
    p = build_single(PROD_ELF, "re8_solo_a", build_dir)
    c = build_single(CONS_ELF, "re8_solo_b", build_dir)
    ok = all(x is not None for x in (s, p, c))
    print(f"=== build {'OK' if ok else 'FAILED'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if build_all() else 1)
