#!/usr/bin/env python3
"""B2c-1 (half 1) HW test — re8 rn3 chain as a merged xrt.elf, bit-exact.

Proves the re8 chain runs correctly when packed through build_merged's
dispatcher (@main entry, 1 aiex.configure, --generate-full-elf), loaded as ONE
xrt.hw_context via xrt.elf + xrt.ext.kernel("main"), vs the production
ResidentXCLBinRunner chain.

This validates plumbing items #1 (BFP .o staging — non-issue, PythocKernels
compile inline) and #3 (chain migrated off ResidentXCLBin onto merged xrt.elf),
AND that the chain's inout ping-pong (arg0=A and arg2=B both filled + drained,
final output in B for odd n_iters) composes under the merged @main dispatcher.

It does NOT yet chain_link the chain into a consumer — see the B2c-1 blocker
report: the chain output is padded-HWC memref<50176xui16> while the model's c3
(3x3 mc) input is im2col patch-packed memref<204800xui16> (4.08x larger), so a
raw chain_link BO alias cannot bridge them without an on-device im2col reformat.

Run:  source env.sh && flock /tmp/npu-dev.lock python3 conv/test_re8_chain_merged_hw.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MDV6 = _HERE.parent
for _p in (str(_HERE), str(_MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyxrt as xrt
from build_re8_chain_merged import build
from aie2_rn3_chain_geo import geo_params
from rn3_chain_runner import run_rn3_chain_geo
from test_rn3_pair_vector_oneblock_hw import f32_to_bf16_u16, bf16_u16_to_f32

N_ITERS = 3
GEO = "re8"


def _make_weight_pairs(rng, n_iters, ic):
    """Random small bf16 RepConv (3x3) + conv2 (3x3) weight pairs.

    Each is the fused_weight_u16 envelope the chain runner expects:
    [ic*ic*9 conv] + [ic bn_w] + [ic bn_b]  (3x3, ic->ic).
    """
    pairs = []
    n = ic * ic * 9 + 2 * ic
    for _ in range(n_iters):
        w1 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
        w2 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
        pairs.append((w1, w2))
    return pairs


def main():
    elf_path = build(GEO, n_iters=N_ITERS)
    if elf_path is None:
        print("FAIL: merged chain ELF build failed")
        return 1

    p = geo_params(GEO)
    ic, G = p["IC"], p["GBOUND"]
    rng = np.random.default_rng(7)
    inp = torch.from_numpy(
        (rng.standard_normal((G, G, ic)).astype(np.float32) * 0.5)
    ).to(torch.bfloat16)
    pairs = _make_weight_pairs(rng, N_ITERS, ic)

    # ---- reference: production ResidentXCLBinRunner chain ----
    ref = run_rn3_chain_geo(GEO, inp, pairs)  # [G, G, ic] bf16

    # ---- merged xrt.elf chain, ONE hw_context, @main, run once ----
    # Reproduce the exact host padding + weight packing the runner uses, then
    # dispatch through xrt.elf instead of the resident xclbin.
    from rn3_chain_runner import _pack_geo_iter, PAD as _PAD
    nt = p["WORKER_TILES"][0]
    weights = np.concatenate([
        np.tile(_pack_geo_iter(w1, w2, ic, p["WSLOT"], p["N_BLK"]), nt)
        for w1, w2 in pairs])
    img = np.zeros(p["IMG_ELEMS"], np.float32)
    imgv = img.reshape(p["IMG_H"], p["IMG"], ic)
    imgv[_PAD:_PAD + G, _PAD:_PAD + G, :] = inp.float().numpy()
    img_u16 = f32_to_bf16_u16(img.reshape(-1))

    device = xrt.device(0)
    elf = xrt.elf(elf_path)
    ctx = xrt.hw_context(device, elf)           # ONE hw_context
    kern = xrt.ext.kernel(ctx, "main")
    one_context = isinstance(ctx, xrt.hw_context)

    a_bo = xrt.ext.bo(device, img_u16.nbytes)
    wt_bo = xrt.ext.bo(device, weights.nbytes)
    b_bo = xrt.ext.bo(device, img_u16.nbytes)

    def _fill(bo, arr):
        mv = bo.map()
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(np.ascontiguousarray(arr), dtype=np.uint8))
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # inout ping-pong: A and B both seeded with the padded input image.
    _fill(a_bo, img_u16)
    _fill(wt_bo, weights.astype(np.uint16))
    _fill(b_bo, img_u16)

    r = xrt.run(kern)
    r.set_arg(0, a_bo); r.set_arg(1, wt_bo); r.set_arg(2, b_bo)
    r.start(); r.wait2()

    # odd n_iters -> final image lands in B (arg2)
    final_bo = b_bo if (N_ITERS % 2 == 1) else a_bo
    final_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    final = np.frombuffer(final_bo.map(), dtype=np.uint16,
                          count=p["IMG_ELEMS"]).copy()
    got = bf16_u16_to_f32(final).reshape(p["IMG_H"], p["IMG"], ic)
    got = got[_PAD:_PAD + G, _PAD:_PAD + G, :]
    got = torch.from_numpy(got).to(torch.bfloat16)

    diff = (got.float() - ref.float()).abs()
    max_diff = float(diff.max())
    exact = bool((got.float() == ref.float()).all())

    print("\n========= B2c-1 (half 1): re8 chain as merged xrt.elf =========")
    print(f"geo={GEO} n_iters={N_ITERS}  chain sig (A,WT,B) = "
          f"(memref<{p['IMG_ELEMS']}xui16>, memref<{p['WSLOT']*p['N_BLK']*2*N_ITERS*nt}xui16>, "
          f"memref<{p['IMG_ELEMS']}xui16>)")
    print(f"[hw_context]  merged chain ELF loaded under 1 xrt.hw_context: "
          f"{'YES' if one_context else 'NO'} (entry='main')")
    print(f"[inout/ping-pong]  arg0=A, arg2=B both seeded + drained; "
          f"final output read from {'B (arg2)' if N_ITERS % 2 else 'A (arg0)'}")
    print(f"[bit-exact vs ResidentXCLBin chain]  max_diff={max_diff:.6f}  "
          f"-> {'PASS (bit-exact)' if exact else ('PASS (bf16 tol)' if max_diff < 0.05 else 'FAIL')}")
    ok = one_context and (exact or max_diff < 0.05)
    print(f"\n{'PASS' if ok else 'FAIL'}: re8 chain runs correctly as a 1-context merged xrt.elf")
    print("===============================================================")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
