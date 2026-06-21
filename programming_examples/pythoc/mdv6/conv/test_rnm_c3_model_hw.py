#!/usr/bin/env python3
"""A1 bit-exact probe: run_rnm_c3 (fused rnm->halo_c3) vs the model's separate
rnm GEMM + mc_c3 3x3, on REAL re8 model weights.

Builds MDV6, pulls rep_elan8.conv2[0].conv3 (rnm) and rep_elan8.conv2[1] (c3),
feeds a random concat(chain_out, x2) [20,20,128], and compares:
  - model path : rt(gemm_rnm) -> rt(mc_c3, 3x3)   (2 launches, BN+SiLU in kernel)
  - fused path : run_rnm_c3 (1 launch, device-resident seam, host BN-bias+SiLU)

Reports max/mean diff so we can judge against the detection tolerance.

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_rnm_c3_model_hw.py
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MDV6 = _HERE.parent
_PYROOT = _MDV6.parent.parent.parent / "python"
for _p in (str(_HERE), str(_MDV6), str(_PYROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mdv6.model import MDV6MITYOLOv9c

spec1 = importlib.util.spec_from_file_location(
    'ett', os.path.join(_MDV6, '_full_model_helpers', 'elan_test_tiled.py'))
ett = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ett)
fuse_bn = ett.fuse_bn

spec3 = importlib.util.spec_from_file_location('mcr', os.path.join(_MDV6, 'run_tiled_mc.py'))
mcr = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(mcr)

from conv.rnm_halo_runner import run_rnm_c3


def _rt_gemm(name, inp, w, H, W_, oc):
    return mcr.run_gemm_conv1x1_mc(name, name, inp, w, H, W_, oc)


def main():
    model = MDV6MITYOLOv9c(num_classes=3).eval()
    wp = os.path.join(_MDV6, 'mdv6_bf16_weights.pt')
    if os.path.exists(wp):
        model.load_state_dict(torch.load(wp, map_location='cpu', weights_only=True))
    model = model.to(torch.bfloat16)

    layer = model.rep_elan8
    repncsp = layer.conv2[0]           # first RepNCSP of re8
    rnm_mod = repncsp.conv3            # rnm: 1x1 128->128
    c3_mod = layer.conv2[1]            # c3 : 3x3 128->128

    H = W = 20; oc = 128; ic = 128
    rng = np.random.default_rng(13)
    concat = torch.from_numpy(
        (rng.standard_normal((H, W, ic)).astype(np.float32) * 0.25)).to(torch.bfloat16)

    rnm_w = fuse_bn(rnm_mod)
    c3_w = fuse_bn(c3_mod)

    # ---- model path: separate rnm GEMM + mc_c3 3x3 ----
    x3rn = _rt_gemm('gemm_re8_rnm', concat, rnm_w, H, W, oc)
    x3_model = mcr.run_tiled_fused_conv_mc('mc_re8_c3', 're8_conv3x3', x3rn, c3_w,
                                           H, W, oc, 4, 4, 16, 1, 3, 1)

    # ---- fused path ----
    x3_fused = run_rnm_c3(concat, rnm_w, c3_w, H, W, oc, mcr_mod=mcr)

    d = (x3_fused.float() - x3_model.float()).abs()
    print("\n===== A1: fused rnm->c3 vs model rnm GEMM + mc_c3 =====")
    print(f"  shape x3: {tuple(x3_model.shape)}")
    print(f"  max_diff={float(d.max()):.5f} mean_diff={float(d.mean()):.6f}")
    print(f"  x3_model[0,0,:4]={x3_model[0,0,:4].float().numpy()}")
    print(f"  x3_fused[0,0,:4]={x3_fused[0,0,:4].float().numpy()}")
    print(f"  model std={x3_model.float().std():.4f} fused std={x3_fused.float().std():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
