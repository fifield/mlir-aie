#!/usr/bin/env python3
"""
MDV6 full model end-to-end test — 32-core multicore version.

All Conv+BN+SiLU sub-layers use 32-core spatial parallelism.
RepConv, AvgPool, Upsample, concat, split, detection run on host CPU.
"""
import sys, os, time, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))
import torch
import torch.nn as nn
from mdv6.model import MDV6MITYOLOv9c
from aie.utils import NPUKernel, DefaultNPURuntime

# Import helpers
_base = os.path.dirname(__file__)
spec1 = importlib.util.spec_from_file_location('ett', os.path.join(_base, 'elan', 'test_tiled.py'))
ett = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ett)
spec3 = importlib.util.spec_from_file_location('mcr', os.path.join(_base, 'run_tiled_mc.py'))
mcr = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(mcr)

fuse_bn = ett.fuse_bn
fuse_bn_transposed = ett.fuse_bn_transposed
run_tiled_mc = mcr.run_tiled_fused_conv_mc
run_gemm_conv1x1 = mcr.run_gemm_conv1x1_mc

# Set to True to use GEMM conv1x1 for all 1×1 convs
USE_GEMM_CONV1X1 = os.environ.get("USE_GEMM_CONV1X1", "1") == "1"

def to_nchw(hwc):
    return hwc.float().permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)

def rt(mc_name, sc_name, input_hwc, weights, oh, ow, oc, th, tw, ob, s=1, ks=1, pad=0):
    """Run tiled conv with multicore (SC loaded lazily only on fallback).

    For 1×1 convs with USE_GEMM_CONV1X1=1, dispatches to the GEMM path which
    uses mmul<4,8,8> vectorized kernels instead of scalar loops.
    """
    if ks == 1 and s == 1 and USE_GEMM_CONV1X1:
        gemm_name = mc_name.replace('mc_', 'gemm_')
        return run_gemm_conv1x1(gemm_name, sc_name, input_hwc, weights,
                                 oh, ow, oc)
    return run_tiled_mc(mc_name, sc_name, input_hwc, weights,
                         oh, ow, oc, th, tw, ob, s, ks, pad)

def run_aconv_mc(mc_name, sc_name, module, input_nchw, out_h, out_w, oc, tile, ocb):
    with torch.no_grad(): pooled = module.avg_pool(input_nchw)
    return rt(mc_name, sc_name, pooled.squeeze(0).permute(1,2,0).contiguous(),
              fuse_bn(module.conv), out_h, out_w, oc, tile, tile, ocb, 2, 3, 1)

def run_elan_mc(model_elan, input_hwc, H, W, ic, oc,
                mc_c1, sc_c1, mc_c3, sc_c3, mc_c4, sc_c4,
                t1, o1, t3, o3, t4, o4, part, proc):
    c1 = rt(mc_c1, sc_c1, input_hwc, fuse_bn(model_elan.conv1), H, W, part, t1, t1, o1, 1, 1, 0)
    x1 = c1[:,:,:proc]; x2 = c1[:,:,proc:]
    x3 = rt(mc_c3, sc_c3, x2, fuse_bn(model_elan.conv2), H, W, proc, t3, t3, o3, 1, 3, 1)
    x4 = rt(mc_c3, sc_c3, x3, fuse_bn(model_elan.conv3), H, W, proc, t3, t3, o3, 1, 3, 1)
    return rt(mc_c4, sc_c4, torch.cat([x1, x2, x3, x4], dim=2),
              fuse_bn(model_elan.conv4), H, W, oc, t4, t4, o4, 1, 1, 0)

def run_rn_mc(repncsp, inp, H, W, ic, oc,
              mc_rn1, sc_rn1, mc_rn3, sc_rn3, mc_rnm, sc_rnm,
              trn1, orn1, trn3, orn3, trnm, ornm):
    """RepNCSP with multicore conv sub-layers + host CPU RepConv."""
    neck = int(oc * 0.5)
    x1 = rt(mc_rn1, sc_rn1, inp, fuse_bn(repncsp.conv1), H, W, neck, trn1, trn1, orn1, 1, 1, 0)
    current = x1
    for bn_block in repncsp.bottleneck:
        residual = current.clone()
        nchw_in = current.float().permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)
        with torch.no_grad():
            repconv_out_nchw = bn_block.conv1(nchw_in)
        repconv_out = repconv_out_nchw.squeeze(0).permute(1, 2, 0).contiguous()
        conv2_out = rt(mc_rn3, sc_rn3, repconv_out, fuse_bn(bn_block.conv2),
                       H, W, neck, trn3, trn3, orn3, 1, 3, 1)
        current = (residual + conv2_out) if bn_block.residual else conv2_out
    x2 = rt(mc_rn1, sc_rn1, inp, fuse_bn(repncsp.conv2), H, W, neck, trn1, trn1, orn1, 1, 1, 0)
    concat = torch.cat([current, x2], dim=2)
    return rt(mc_rnm, sc_rnm, concat, fuse_bn(repncsp.conv3), H, W, oc, trnm, trnm, ornm, 1, 1, 0)


def run_re_mc(layer, inp, H, W, ic, oc, part, proc,
              mc_c1, sc_c1, mc_c3, sc_c3, mc_c4, sc_c4,
              mc_rn1, sc_rn1, mc_rn3, sc_rn3, mc_rnm, sc_rnm,
              tc1, oc1, tc3, oc3, tc4, oc4, trn1, orn1, trn3, orn3, trnm, ornm):
    half = part // 2
    c1 = rt(mc_c1, sc_c1, inp, fuse_bn(layer.conv1), H, W, part, tc1, tc1, oc1, 1, 1, 0)
    x1 = c1[:,:,:half]; x2 = c1[:,:,half:]
    x3rn = run_rn_mc(layer.conv2[0], x2, H, W, half, proc,
                      mc_rn1, sc_rn1, mc_rn3, sc_rn3, mc_rnm, sc_rnm,
                      trn1, orn1, trn3, orn3, trnm, ornm)
    x3 = rt(mc_c3, sc_c3, x3rn, fuse_bn(layer.conv2[1]), H, W, proc, tc3, tc3, oc3, 1, 3, 1)
    x4rn = run_rn_mc(layer.conv3[0], x3, H, W, proc, proc,
                      mc_rn1, sc_rn1, mc_rn3, sc_rn3, mc_rnm, sc_rnm,
                      trn1, orn1, trn3, orn3, trnm, ornm)
    x4 = rt(mc_c3, sc_c3, x4rn, fuse_bn(layer.conv3[1]), H, W, proc, tc3, tc3, oc3, 1, 3, 1)
    return rt(mc_c4, sc_c4, torch.cat([x1, x2, x3, x4], dim=2),
              fuse_bn(layer.conv4), H, W, oc, tc4, tc4, oc4, 1, 1, 0)


def main():
    print("=" * 70)
    print("MDV6 Full Model — 32-Core Multicore")
    print("=" * 70)

    model = MDV6MITYOLOv9c(num_classes=3).eval().to(torch.bfloat16)
    torch.manual_seed(42)
    x = torch.randn(1, 3, 640, 640, dtype=torch.bfloat16)

    print("\nPyTorch reference...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad(): ref = model(x)
    print(f"{time.time()-t0:.1f}s")

    print("\n--- Forward pass (32-core) ---")
    t_start = time.time()

    inp = x.squeeze(0).permute(1, 2, 0).contiguous()

    # Conv0 (640→320, 3→32, s=2)
    print("  conv0...", end=" ", flush=True); t = time.time()
    conv0_hwc = rt('mc_ftconv0', 'ftconv0', inp, fuse_bn(model.conv0),
                    320, 320, 32, 24, 24, 32, 2, 3, 1)
    print(f"{time.time()-t:.1f}s")

    # Conv1 (320→160, 32→64, s=2)
    print("  conv1...", end=" ", flush=True); t = time.time()
    conv1_hwc = rt('mc_ftconv1', 'ftconv1', conv0_hwc, fuse_bn(model.conv1),
                    160, 160, 64, 12, 12, 16, 2, 3, 1)
    print(f"{time.time()-t:.1f}s")

    # ELAN2 (160×160, 64→64)
    print("  elan2...", end=" ", flush=True); t = time.time()
    elan2 = run_elan_mc(model.elan2, conv1_hwc, 160, 160, 64, 64,
                         'mc_elan_c1', 'tf_elan_conv1',
                         'mc_elan_c3', 'tf_elan_conv3x3',
                         'mc_elan_c4', 'tf_elan_conv4',
                         8, 64, 16, 32, 8, 64, 64, 32)
    print(f"{time.time()-t:.1f}s")

    # AConv3 + rep_elan4 [B3]
    print("  aconv3...", end=" ", flush=True); t = time.time()
    ac3 = run_aconv_mc('mc_aconv3', 'tf_aconv3', model.aconv3, to_nchw(elan2), 80, 80, 128, 8, 16)
    print(f"{time.time()-t:.1f}s")
    print("  rep_elan4...", end=" ", flush=True); t = time.time()
    b3 = run_re_mc(model.rep_elan4, ac3, 80, 80, 128, 128, 128, 64,
                    'mc_re4_c1', 're4_conv1', 'mc_re4_c3', 're4_conv3x3', 'mc_re4_c4', 're4_conv4',
                    'mc_re4_rn1', 're4_rn_conv1x1_64_32', 'mc_re4_rn3', 're4_rn_conv3x3_32_32',
                    'mc_elan_c1', 'tf_elan_conv1',
                    10, 64, 12, 16, 8, 32, 16, 32, 16, 32, 8, 64)
    print(f"{time.time()-t:.1f}s")

    # AConv5 + rep_elan6 [B4]
    print("  aconv5...", end=" ", flush=True); t = time.time()
    b3n = to_nchw(b3)
    ac5 = run_aconv_mc('mc_aconv5', 'aconv5', model.aconv5, b3n, 40, 40, 192, 4, 8)
    print(f"{time.time()-t:.1f}s")
    print("  rep_elan6...", end=" ", flush=True); t = time.time()
    b4 = run_re_mc(model.rep_elan6, ac5, 40, 40, 192, 192, 192, 96,
                    'mc_re6_c1', 're6_conv1', 'mc_re6_c3', 're6_conv3x3', 'mc_re6_c4', 're6_conv4',
                    'mc_re6_rn1', 're6_rn_c1', 'mc_re6_rn3', 're6_rn_c3',
                    'mc_re6_rnm', 're6_rn_merge',
                    8, 32, 8, 16, 4, 32, 10, 48, 8, 16, 8, 48)
    print(f"{time.time()-t:.1f}s")

    # AConv7 + rep_elan8 [B5]
    print("  aconv7...", end=" ", flush=True); t = time.time()
    b4n = to_nchw(b4)
    ac7 = run_aconv_mc('mc_aconv7', 'aconv7', model.aconv7, b4n, 20, 20, 256, 4, 4)
    print(f"{time.time()-t:.1f}s")
    print("  rep_elan8...", end=" ", flush=True); t = time.time()
    b5 = run_re_mc(model.rep_elan8, ac7, 20, 20, 256, 256, 256, 128,
                    'mc_re8_c1', 're8_conv1', 'mc_re8_c3', 're8_conv3x3', 'mc_re8_c4', 're8_conv4',
                    'mc_re8_rn1', 're8_rn_c1', 'mc_re8_rn3', 're8_rn_c3',
                    'mc_re8_c1', 're8_rn_merge',
                    4, 32, 4, 16, 4, 16, 8, 64, 8, 16, 4, 32)
    print(f"{time.time()-t:.1f}s")

    # SPP9
    print("  spp9...", end=" ", flush=True); t = time.time()
    spp_c1 = rt('mc_re8_c1', 'spp_conv1', b5, fuse_bn(model.spp9.conv1), 20, 20, 128, 4, 4, 32, 1, 1, 0)
    spp_n = to_nchw(spp_c1)
    feats = [spp_c1]; cur = spp_n
    for pool in model.spp9.pools:
        with torch.no_grad(): cur = pool(cur)
        feats.append(cur.squeeze(0).permute(1,2,0).contiguous())
    n3 = rt('mc_re8_c4', 're8_conv4', torch.cat(feats, dim=2), fuse_bn(model.spp9.conv5),
            20, 20, 256, 4, 4, 16, 1, 1, 0)
    print(f"{time.time()-t:.1f}s")

    # Neck
    print("  rep_elan12...", end=" ", flush=True); t = time.time()
    n3n = to_nchw(n3)
    with torch.no_grad(): up1 = nn.Upsample(scale_factor=2, mode='nearest')(n3n)
    cat12 = torch.cat([up1, b4n], dim=1).squeeze(0).permute(1,2,0).contiguous()
    n4 = run_re_mc(model.rep_elan12, cat12, 40, 40, 448, 192, 192, 96,
                    'mc_re12_c1', 're12_conv1', 'mc_re6_c3', 're6_conv3x3', 'mc_re6_c4', 're6_conv4',
                    'mc_re6_rn1', 're6_rn_c1', 'mc_re6_rn3', 're6_rn_c3',
                    'mc_re6_rnm', 're6_rn_merge',
                    4, 32, 8, 16, 4, 32, 10, 48, 8, 16, 8, 48)
    print(f"{time.time()-t:.1f}s")

    print("  rep_elan15...", end=" ", flush=True); t = time.time()
    n4n = to_nchw(n4)
    with torch.no_grad(): up2 = nn.Upsample(scale_factor=2, mode='nearest')(n4n)
    cat15 = torch.cat([up2, b3n], dim=1).squeeze(0).permute(1,2,0).contiguous()
    p3 = run_re_mc(model.rep_elan15, cat15, 80, 80, 320, 128, 128, 64,
                    'mc_re15_c1', 're15_conv1', 'mc_re4_c3', 're4_conv3x3', 'mc_re4_c4', 're15_conv4',
                    'mc_re4_rn1', 're4_rn_conv1x1_64_32', 'mc_re4_rn3', 're4_rn_conv3x3_32_32',
                    'mc_elan_c4', 're15_rn_merge',
                    6, 32, 12, 16, 8, 32, 16, 32, 16, 32, 8, 64)
    print(f"{time.time()-t:.1f}s")

    # Head P4
    print("  head P4...", end=" ", flush=True); t = time.time()
    p3n = to_nchw(p3)
    ac16 = run_aconv_mc('mc_aconv16', 'aconv16', model.aconv16, p3n, 40, 40, 96, 4, 8)
    cat18 = torch.cat([to_nchw(ac16), n4n], dim=1).squeeze(0).permute(1,2,0).contiguous()
    p4 = run_re_mc(model.rep_elan18, cat18, 40, 40, 288, 192, 192, 96,
                    'mc_re18_c1', 're18_conv1', 'mc_re6_c3', 're6_conv3x3', 'mc_re6_c4', 're6_conv4',
                    'mc_re6_rn1', 're6_rn_c1', 'mc_re6_rn3', 're6_rn_c3',
                    'mc_re6_rnm', 're6_rn_merge',
                    4, 32, 8, 16, 4, 32, 10, 48, 8, 16, 8, 48)
    print(f"{time.time()-t:.1f}s")

    # Head P5
    print("  head P5...", end=" ", flush=True); t = time.time()
    p4n = to_nchw(p4)
    ac19 = run_aconv_mc('mc_aconv19', 'aconv19', model.aconv19, p4n, 20, 20, 128, 4, 4)
    cat21 = torch.cat([to_nchw(ac19), n3n], dim=1).squeeze(0).permute(1,2,0).contiguous()
    p5 = run_re_mc(model.rep_elan21, cat21, 20, 20, 384, 256, 256, 128,
                    'mc_re6_c4', 're21_conv1', 'mc_re8_c3', 're8_conv3x3', 'mc_re8_c4', 're8_conv4',
                    'mc_re8_rn1', 're8_rn_c1', 'mc_re8_rn3', 're8_rn_c3',
                    'mc_re8_c1', 're8_rn_merge',
                    4, 32, 4, 16, 4, 16, 8, 64, 8, 16, 4, 32)
    print(f"{time.time()-t:.1f}s")

    # Detection (CPU)
    print("  detect...", end=" ", flush=True)
    p5n = to_nchw(p5)
    with torch.no_grad(): det = model.detect([p3n, p4n, p5n])
    print("CPU")

    t_total = time.time() - t_start

    # Compare
    print(f"\n{'='*70}")
    print(f"Total forward pass: {t_total:.1f}s")
    print(f"{'='*70}")
    for i, ((cr, ar, vr), (ca, aa, va)) in enumerate(zip(ref, det)):
        cd = torch.abs(cr.float() - ca.float()).max().item()
        vd = torch.abs(vr.float() - va.float()).max().item()
        scale = ['P3 80×80', 'P4 40×40', 'P5 20×20'][i]
        print(f"  {scale}: class_diff={cd:.4f}, vector_diff={vd:.4f}")

    max_cls = max(torch.abs(cr.float()-ca.float()).max().item() for (cr,_,_),(ca,_,_) in zip(ref, det))
    max_vec = max(torch.abs(vr.float()-va.float()).max().item() for (_,_,vr),(_,_,va) in zip(ref, det))
    print(f"\n  Overall: max_class_diff={max_cls:.4f}, max_vector_diff={max_vec:.4f}")
    ok = max_cls < 5.0 and max_vec < 5.0
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
