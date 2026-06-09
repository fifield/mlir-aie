#!/usr/bin/env python3
"""Direct site comparator: current two mc_re6_rn3 calls vs fused memtile rn3-pair.

Runs the model up to the first/selected re6 rn3 pair site, then compares:

    baseline = rt(mc_re6_rn3, current, w_rep) -> rt(mc_re6_rn3, repconv_out, w_c2)
    fused    = run_re6_rn3_pair(current, w_rep, w_c2)

This returns detailed max locations so numerical deltas can be fixed by evidence.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import test_full_model_mc as tfm  # noqa: E402
from mdv6.model import MDV6MITYOLOv9c  # noqa: E402
from conv.rn3_pair_vector_memtile_runner import run_re6_rn3_pair_debug, last_stats, close_runner, pack_vector_weight_slots_from_fused  # noqa: E402
from conv.aie2_rn3_pair_vector_ocb import N_MID_BLOCKS, MID_BLOCK, IC, W1_SIZE, WEIGHT_SLOT_SIZE  # noqa: E402
from conv.test_rn3_pair_vector_oneblock_hw import unpack_packed_3x3_weights_f32  # noqa: E402


def _load_model():
    model = MDV6MITYOLOv9c().eval().bfloat16()
    wt_path = ROOT / "mdv6_bf16_weights.pt"
    if wt_path.exists():
        sd = torch.load(wt_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # Some checkpoints are wrapped modules; keep same forgiving behavior as main script.
            pass
    return model


def _input(seed: int):
    rng = np.random.default_rng(seed)
    # Same shape as model input; deterministic random is enough for site localization.
    x = rng.normal(0.0, 0.35, size=(640, 640, 3)).astype(np.float32)
    return torch.from_numpy(x).to(torch.bfloat16)


def _reach_rep_elan6_input(model, x_hwc):
    # Mirror test_full_model_mc forward through aconv5, using existing wrappers.
    inp_padded = torch.zeros(640, 640, 8, dtype=torch.bfloat16)
    inp_padded[:, :, :3] = x_hwc
    conv0_wt = tfm.fuse_bn(model.conv0)
    oc0, ic0, ks0 = 32, 3, 3
    wt_conv = conv0_wt[:oc0 * ic0 * ks0 * ks0]
    wt_bn = conv0_wt[oc0 * ic0 * ks0 * ks0:]
    w_orig = torch.from_numpy(wt_conv.copy()).view(torch.bfloat16).reshape(oc0, ic0, ks0, ks0)
    w_pad = torch.zeros(oc0, 8, ks0, ks0, dtype=torch.bfloat16)
    w_pad[:, :ic0, :, :] = w_orig
    conv0_wt_padded = np.concatenate([w_pad.flatten().view(torch.uint16).numpy(), wt_bn])
    conv0 = tfm.rt('mc_ftconv0', 'ftconv0', inp_padded, conv0_wt_padded, 320, 320, 32, 20, 20, 32, 2, 3, 1)
    conv1 = tfm.rt('mc_ftconv1', 'ftconv1', conv0, tfm.fuse_bn(model.conv1), 160, 160, 64, 12, 12, 16, 2, 3, 1)
    elan2 = tfm.run_elan_mc(model.elan2, conv1, 160, 160, 64, 64,
                            'mc_elan_c1', 'tf_elan_conv1', 'mc_elan_c3', 'tf_elan_conv3x3', 'mc_elan_c4', 'tf_elan_conv4',
                            8, 64, 8, 32, 8, 64, 64, 32)
    aconv3 = tfm.run_aconv_mc('mc_aconv3', 'tf_aconv3', model.aconv3, tfm.to_nchw(elan2), 80, 80, 128, 8, 16)
    rep4 = tfm.run_re_mc(model.rep_elan4, aconv3, 80, 80, 128, 128, 128, 64,
                         'mc_re4_c1', 're4_conv1', 'mc_re4_c3', 're4_conv3x3', 'mc_re4_c4', 're4_conv4',
                         'mc_re4_rn1', 're4_rn_conv1x1_64_32', 'mc_re4_rn3', 're4_rn_conv3x3_32_32',
                         'mc_elan_c1', 'tf_elan_conv1',
                         10, 64, 12, 16, 8, 32, 16, 32, 8, 32, 8, 64)
    aconv5 = tfm.run_aconv_mc('mc_aconv5', 'aconv5', model.aconv5, tfm.to_nchw(rep4), 40, 40, 192, 4, 8)
    return aconv5


def _cpu_conv1_oracle(current: torch.Tensor, w_rep: np.ndarray) -> torch.Tensor:
    """Whole-image conv1 CPU oracle using the same vector-packed RepConv slots."""
    slots = pack_vector_weight_slots_from_fused(w_rep, w_rep).reshape(-1, WEIGHT_SLOT_SIZE)
    parts = []
    x = current.detach().float().permute(2, 0, 1).unsqueeze(0).contiguous()
    for mb in range(N_MID_BLOCKS):
        w, bn_w, bn_b = unpack_packed_3x3_weights_f32(slots[mb, :W1_SIZE], MID_BLOCK, IC)
        wt = torch.from_numpy(w).float()
        y = torch.nn.functional.conv2d(x, wt, bias=None, stride=1, padding=1)
        bw = torch.from_numpy(bn_w).float().view(1, -1, 1, 1)
        bb = torch.from_numpy(bn_b).float().view(1, -1, 1, 1)
        # Match the project/vector approximation more closely than torch SiLU.
        z = y * bw + bb
        sig = 0.5 + z / (2.0 + 2.0 * torch.abs(z))
        parts.append((z * sig).to(torch.bfloat16).float())
    return torch.cat(parts, dim=1).squeeze(0).permute(1, 2, 0).contiguous().to(torch.bfloat16)


def _compare_one(current, block, site_idx: int):
    w_rep = tfm.fuse_repconv(block.conv1)
    w_c2 = tfm.fuse_bn(block.conv2)
    repconv_out = tfm.rt('mc_re6_rn3', 're6_rn_c3', current, w_rep, 40, 40, 48, 8, 8, 16, 1, 3, 1)
    baseline = tfm.rt('mc_re6_rn3', 're6_rn_c3', repconv_out, w_c2, 40, 40, 48, 8, 8, 16, 1, 3, 1)
    fused, fused_conv1, _raw = run_re6_rn3_pair_debug(current, w_rep, w_c2, bo_key=f"sitecmp_{site_idx}")
    cpu_conv1 = _cpu_conv1_oracle(current, w_rep)
    dc_base = (cpu_conv1.float() - repconv_out.float()).abs()
    dc_fused = (cpu_conv1.float() - fused_conv1.float()).abs()
    print(f"site={site_idx} cpu-vs-prod-conv1 max={float(dc_base.max()):.6f} mean={float(dc_base.mean()):.6f}; cpu-vs-fused-conv1 max={float(dc_fused.max()):.6f} mean={float(dc_fused.mean()):.6f}")
    d1 = (fused_conv1.float() - repconv_out.float()).abs()
    d1flat = d1.reshape(-1)
    d1vals, d1idxs = torch.topk(d1flat, k=min(8, d1flat.numel()))
    print(f"site={site_idx} conv1 max={float(d1vals[0]):.6f} mean={float(d1.mean()):.6f} p99={float(torch.quantile(d1flat, 0.99)):.6f}")
    for rank, (v, idx) in enumerate(zip(d1vals[:6], d1idxs[:6])):
        c1 = int(idx % 48)
        tmp1 = int(idx // 48)
        w1p = tmp1 % 40
        h1p = tmp1 // 40
        print(f"  c1top{rank}: h={h1p:02d} w={w1p:02d} c={c1:02d} diff={float(v):.6f} base={float(repconv_out[h1p,w1p,c1]):.6f} fused={float(fused_conv1[h1p,w1p,c1]):.6f}")
    d = (fused.float() - baseline.float()).abs()
    flat = d.reshape(-1)
    vals, idxs = torch.topk(flat, k=min(20, flat.numel()))
    print(f"site={site_idx} max={float(vals[0]):.6f} mean={float(d.mean()):.6f} p99={float(torch.quantile(flat, 0.99)):.6f}")
    st = last_stats()
    if st is not None:
        print(f"  fused_stats kernel_ms={st.kernel_ms:.3f} total_ms={st.total_ms:.3f} written={st.n_written} bytes={st.bytes_written}")
    for rank, (v, idx) in enumerate(zip(vals[:10], idxs[:10])):
        c = int(idx % 48)
        tmp = int(idx // 48)
        w = tmp % 40
        h = tmp // 40
        print(f"  top{rank}: h={h:02d} w={w:02d} c={c:02d} diff={float(v):.6f} base={float(baseline[h,w,c]):.6f} fused={float(fused[h,w,c]):.6f}")
    # Tile/edge summaries.
    edge_mask = torch.zeros((40, 40), dtype=torch.bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    tile_boundary = torch.zeros((40, 40), dtype=torch.bool)
    for b in [7, 8, 15, 16, 23, 24, 31, 32]:
        tile_boundary[b, :] = True
        tile_boundary[:, b] = True
    print(f"  edge_max={float(d[edge_mask].max()):.6f} interior_max={float(d[~edge_mask].max()):.6f} boundary_max={float(d[tile_boundary].max()):.6f} nonboundary_max={float(d[~tile_boundary].max()):.6f}")
    return baseline


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, default=0, help="re6 rn3 pair site in rep_elan6: 0-2 for conv2[0], 3-5 for conv3[0], or -1 for all 6")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)
    os.environ.setdefault("USE_GEMM_CONV1X1", "1")
    model = tfm._setup_model()
    x_hwc = _input(args.seed)
    current = _reach_rep_elan6_input(model, x_hwc)

    layer = model.rep_elan6
    c1 = tfm.rt('mc_re6_c1', 're6_conv1', current, tfm.fuse_bn(layer.conv1), 40, 40, 192, 8, 8, 32, 1, 1, 0)
    half = 96
    x2_outer = c1[:, :, half:]

    def run_inner_baseline(repncsp, inp):
        x1, x2 = tfm.run_gemm_pair('gemm_re6_rn1', 're6_rn_c1', inp,
                                   tfm.fuse_bn(repncsp.conv1), tfm.fuse_bn(repncsp.conv2),
                                   40, 40, 48)
        cur = x1
        for block in repncsp.bottleneck:
            w_rep = tfm.fuse_repconv(block.conv1)
            w_c2 = tfm.fuse_bn(block.conv2)
            repconv_out = tfm.rt('mc_re6_rn3', 're6_rn_c3', cur, w_rep, 40, 40, 48, 8, 8, 16, 1, 3, 1)
            out = tfm.rt('mc_re6_rn3', 're6_rn_c3', repconv_out, w_c2, 40, 40, 48, 8, 8, 16, 1, 3, 1)
            cur = cur + out if block.residual else out
        return tfm.rt('mc_re6_rnm', 're6_rn_merge', torch.cat([cur, x2], dim=2),
                      tfm.fuse_bn(repncsp.conv3), 40, 40, 96, 8, 8, 48, 1, 1, 0)

    # First inner RepNCSP: rep_elan6.conv2[0], sites 0..2.
    rep2 = layer.conv2[0]
    x1, x2 = tfm.run_gemm_pair('gemm_re6_rn1', 're6_rn_c1', x2_outer,
                               tfm.fuse_bn(rep2.conv1), tfm.fuse_bn(rep2.conv2),
                               40, 40, 48)
    cur = x1
    wanted = set(range(6)) if args.site < 0 else {args.site}
    for i, block in enumerate(rep2.bottleneck):
        if i in wanted:
            out = _compare_one(cur, block, i)
        else:
            out = _compare_one(cur, block, i) if args.site < 0 else None
            if out is None:
                w_rep = tfm.fuse_repconv(block.conv1)
                w_c2 = tfm.fuse_bn(block.conv2)
                repconv_out = tfm.rt('mc_re6_rn3', 're6_rn_c3', cur, w_rep, 40, 40, 48, 8, 8, 16, 1, 3, 1)
                out = tfm.rt('mc_re6_rn3', 're6_rn_c3', repconv_out, w_c2, 40, 40, 48, 8, 8, 16, 1, 3, 1)
        cur = cur + out if block.residual else out
    x3rn = tfm.rt('mc_re6_rnm', 're6_rn_merge', torch.cat([cur, x2], dim=2),
                  tfm.fuse_bn(rep2.conv3), 40, 40, 96, 8, 8, 48, 1, 1, 0)
    x3 = tfm.rt('mc_re6_c3', 're6_conv3x3', x3rn, tfm.fuse_bn(layer.conv2[1]), 40, 40, 96, 8, 8, 16, 1, 3, 1)

    # Second inner RepNCSP: rep_elan6.conv3[0], sites 3..5.
    rep3 = layer.conv3[0]
    x1b, x2b = tfm.run_gemm_pair('gemm_re6_rn1', 're6_rn_c1', x3,
                                 tfm.fuse_bn(rep3.conv1), tfm.fuse_bn(rep3.conv2),
                                 40, 40, 48)
    cur = x1b
    for j, block in enumerate(rep3.bottleneck):
        site = j + 3
        if site in wanted:
            out = _compare_one(cur, block, site)
        else:
            if args.site < 0:
                out = _compare_one(cur, block, site)
            else:
                w_rep = tfm.fuse_repconv(block.conv1)
                w_c2 = tfm.fuse_bn(block.conv2)
                repconv_out = tfm.rt('mc_re6_rn3', 're6_rn_c3', cur, w_rep, 40, 40, 48, 8, 8, 16, 1, 3, 1)
                out = tfm.rt('mc_re6_rn3', 're6_rn_c3', repconv_out, w_c2, 40, 40, 48, 8, 8, 16, 1, 3, 1)
        cur = cur + out if block.residual else out
    close_runner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
