#!/usr/bin/env python3
"""HW gate: rn3 chain with fused rnm epilogue vs torch reference.

rnm = SiLU(BN(W3 . concat(chain_out, x2))) computed inside the chain launch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conv.aie2_rn3_chain_geo import geo_params
from conv.rn3_chain_runner import run_rn3_chain_geo
from conv.test_rn3_pair_vector_oneblock_hw import f32_to_bf16_u16
from conv.test_rn3_chain_geo_hw import torch_pair

GEO = os.environ.get("GEO", "re6")
N_ITERS = int(os.environ.get("N_ITERS", "3"))


def main():
    p = geo_params(GEO)
    ic, G = p["IC"], p["GBOUND"]
    oc = 2 * ic

    rng = np.random.default_rng(0)
    x1 = torch.from_numpy(rng.standard_normal((G, G, ic)).astype(np.float32) * 0.5).to(torch.bfloat16)
    x2 = torch.from_numpy(rng.standard_normal((G, G, ic)).astype(np.float32) * 0.5).to(torch.bfloat16)
    mk = lambda: (rng.integers(0, 60, size=ic*ic*9 + 2*ic).astype(np.uint16) + 15000).astype(np.uint16)
    pairs = [(mk(), mk()) for _ in range(N_ITERS)]

    w3 = (rng.standard_normal((oc, oc)).astype(np.float32) * 0.1)
    bnw = np.ones(oc, np.float32)
    bnb = np.zeros(oc, np.float32)
    rnm_w_u16 = np.concatenate([f32_to_bf16_u16(w3.reshape(-1)),
                                f32_to_bf16_u16(bnw), f32_to_bf16_u16(bnb)])

    out = run_rn3_chain_geo(GEO, x1, pairs, x2_hwc=x2, rnm_w_u16=rnm_w_u16)

    ref = x1
    for w1, w2 in pairs:
        ref = (torch_pair(torch_pair(ref, w1, ic), w2, ic).float() + ref.float()).to(torch.bfloat16)
    cat = torch.cat([ref, x2], dim=2).float()  # [G, G, oc]
    y = cat.reshape(-1, oc) @ torch.from_numpy(w3).T.float()
    y = y * torch.from_numpy(bnw) + torch.from_numpy(bnb)
    rref = F.silu(y).reshape(G, G, oc).to(torch.bfloat16)

    d = (out.float() - rref.float()).abs().numpy()
    print(f"out mean|x|={out.float().abs().mean():.4f} ref mean|x|={rref.float().abs().mean():.4f}")
    print(f"chain_rnm_{GEO}({N_ITERS}): max={d.max():.6f} mean={d.mean():.6f}")
    print("PASS" if d.max() < 0.06 else "FAIL")


if __name__ == "__main__":
    main()
