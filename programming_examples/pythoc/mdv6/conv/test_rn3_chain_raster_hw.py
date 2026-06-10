#!/usr/bin/env python3
"""HW gate + bench: raster chain (1 tile/core) vs torch and vs column chain."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conv.rn3_chain_runner import run_rn3_chain_raster, run_rn3_chain_geo
from conv.test_rn3_chain_geo_hw import torch_pair

GEO = os.environ.get("GEO", "re6w")
BASE = GEO[:-1]
N_ITERS = int(os.environ.get("N_ITERS", "3"))


def main():
    from conv.aie2_rn3_chain_geo import raster_params
    p = raster_params(GEO)
    ic, G = p["IC"], p["GBOUND"]
    rng = np.random.default_rng(0)
    x0 = torch.from_numpy(rng.standard_normal((G, G, ic)).astype(np.float32) * 0.5).to(torch.bfloat16)
    mk = lambda: (rng.integers(0, 60, size=ic*ic*9 + 2*ic).astype(np.uint16) + 15000).astype(np.uint16)
    pairs = [(mk(), mk()) for _ in range(N_ITERS)]

    out = run_rn3_chain_raster(GEO, x0, pairs)
    ref = x0
    for w1, w2 in pairs:
        ref = (torch_pair(torch_pair(ref, w1, ic), w2, ic).float() + ref.float()).to(torch.bfloat16)
    d = (out.float() - ref.float()).abs()
    print(f"raster_{GEO}({N_ITERS}): max={d.max():.6f} mean={d.mean():.6f}",
          "PASS" if d.max() < 0.05 else "FAIL")

    # warm bench raster vs column chain
    for fn, name in ((run_rn3_chain_raster, "raster"), (run_rn3_chain_geo, "column")):
        g = GEO if name == "raster" else BASE
        fn(g, x0, pairs)
        t0 = time.perf_counter()
        for _ in range(10):
            fn(g, x0, pairs)
        print(f"{name}: {(time.perf_counter()-t0)/10*1e3:.2f} ms/launch ({N_ITERS} iters)")


if __name__ == "__main__":
    main()
