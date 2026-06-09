#!/usr/bin/env python3
"""
MDV6 full model end-to-end test — 32-core multicore version.

All Conv+BN+SiLU sub-layers use 32-core spatial parallelism.
RepConv, AvgPool, Upsample, concat, split, detection run on host CPU.
"""
import sys, os, time, importlib.util
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))
import torch
import torch.nn as nn
from mdv6.model import MDV6MITYOLOv9c
from aie.utils import NPUKernel, DefaultNPURuntime

# Import helpers
_base = os.path.dirname(__file__)
spec1 = importlib.util.spec_from_file_location('ett', os.path.join(_base, '_full_model_helpers', 'elan_test_tiled.py'))
ett = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ett)
spec3 = importlib.util.spec_from_file_location('mcr', os.path.join(_base, 'run_tiled_mc.py'))
mcr = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(mcr)

fuse_bn = ett.fuse_bn
fuse_bn_transposed = ett.fuse_bn_transposed
fuse_repconv = ett.fuse_repconv
run_tiled_mc = mcr.run_tiled_fused_conv_mc
run_gemm_conv1x1 = mcr.run_gemm_conv1x1_mc
run_gemm_pair = mcr.run_gemm_pair_mc

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
    if os.environ.get("DEBUG_ELAN2_SUB"):
        with torch.no_grad():
            ref = model_elan.conv1(input_hwc.float().permute(2,0,1).unsqueeze(0).to(torch.bfloat16)).squeeze(0).permute(1,2,0).bfloat16()
        # Also compare ABSOLUTE value distributions + a few elements
        diff = (c1.float() - ref.float()).abs()
        print(f"\n      [c1 1x1] npu:std={c1.float().std():.4f} ref:std={ref.float().std():.4f} "
              f"ratio={c1.float().std()/max(ref.float().std(),1e-9):.3f} max_diff={diff.max():.4f}")
        # Per-channel ratio to see if uniform loss
        npu_ch_std = c1.float().std(dim=(0,1))
        ref_ch_std = ref.float().std(dim=(0,1))
        ratios = (npu_ch_std / ref_ch_std.clamp(min=1e-9))
        print(f"        per-channel ratio min={ratios.min():.3f} max={ratios.max():.3f} mean={ratios.mean():.3f}")
        # Sample a few pixels
        for (h, w, c) in [(0,0,0), (80,80,10), (100,50,30), (150,150,63)]:
            print(f"        [{h},{w},{c}] npu={c1[h,w,c].float():.5f} ref={ref[h,w,c].float():.5f}")
    x1 = c1[:,:,:proc]; x2 = c1[:,:,proc:]
    x3 = rt(mc_c3, sc_c3, x2, fuse_bn(model_elan.conv2), H, W, proc, t3, t3, o3, 1, 3, 1)
    if os.environ.get("DEBUG_ELAN2_SUB"):
        with torch.no_grad():
            ref = model_elan.conv2(x2.float().permute(2,0,1).unsqueeze(0).to(torch.bfloat16)).squeeze(0).permute(1,2,0).bfloat16()
        print(f"      [c2 3x3] npu:std={x3.float().std():.4f} ref:std={ref.float().std():.4f} "
              f"ratio={x3.float().std()/max(ref.float().std(),1e-9):.3f}")
    x4 = rt(mc_c3, sc_c3, x3, fuse_bn(model_elan.conv3), H, W, proc, t3, t3, o3, 1, 3, 1)
    if os.environ.get("DEBUG_ELAN2_SUB"):
        with torch.no_grad():
            ref = model_elan.conv3(x3.float().permute(2,0,1).unsqueeze(0).to(torch.bfloat16)).squeeze(0).permute(1,2,0).bfloat16()
        print(f"      [c3 3x3] npu:std={x4.float().std():.4f} ref:std={ref.float().std():.4f} "
              f"ratio={x4.float().std()/max(ref.float().std(),1e-9):.3f}")
    concat = torch.cat([x1, x2, x3, x4], dim=2)
    out = rt(mc_c4, sc_c4, concat, fuse_bn(model_elan.conv4), H, W, oc, t4, t4, o4, 1, 1, 0)
    if os.environ.get("DEBUG_ELAN2_SUB"):
        with torch.no_grad():
            ref = model_elan.conv4(concat.float().permute(2,0,1).unsqueeze(0).to(torch.bfloat16)).squeeze(0).permute(1,2,0).bfloat16()
        print(f"      [c4 1x1] npu:std={out.float().std():.4f} ref:std={ref.float().std():.4f} "
              f"ratio={out.float().std()/max(ref.float().std(),1e-9):.3f}")
    return out

def run_rn_mc(repncsp, inp, H, W, ic, oc,
              mc_rn1, sc_rn1, mc_rn3, sc_rn3, mc_rnm, sc_rnm,
              trn1, orn1, trn3, orn3, trnm, ornm):
    """RepNCSP with multicore conv sub-layers + host CPU RepConv.

    Phase C step A: x1 and x2 are both 1x1 convs on the same `inp` input
    (only weights differ), so they fuse into one xrt.run via run_gemm_pair.
    x2 is computed upfront alongside x1 — semantically identical since neither
    feeds the bottleneck loop.
    """
    neck = int(oc * 0.5)
    gemm_rn1 = mc_rn1.replace('mc_', 'gemm_')
    x1, x2 = run_gemm_pair(gemm_rn1, sc_rn1, inp,
                            fuse_bn(repncsp.conv1), fuse_bn(repncsp.conv2),
                            H, W, neck)
    current = x1
    for bn_block in repncsp.bottleneck:
        residual = current.clone()
        # Phase D: RepConv on NPU via weight reparameterization.
        # bn_block.conv1 is a RepConv module — SiLU(BN(3x3(x)) + BN(1x1(x)))
        # — which folds into a single 3x3 conv with bias at inference.
        # fuse_repconv collapses both BN+conv branches and sets bn_w=1 so
        # the mc_*_rn3 kernel runs it straight (its built-in SiLU matches
        # RepConv.act). This adds one mc_rn3 dispatch per bottleneck (~1.4
        # ms NPU) while removing ~600 µs CPU RepConv per iteration —
        # currently a net wall regression on its own, because each mc_rn3
        # dispatch costs more than the CPU it replaces. The real win comes
        # when a fused 3x3+3x3+add+SiLU kernel lands (Phase E): the
        # back-to-back conv pair stays in memtile L2 with no host
        # reshape between them. fuse_repconv is the host-side prep that
        # makes that future fusion drop-in. See PHASE_C_PLAN.md.
        use_rn3pair_memtile = (os.environ.get('MDV6_USE_RN3PAIR_MEMTILE', '0') not in ('', '0', 'false', 'False')
                and mc_rn3 == 'mc_re6_rn3' and H == 40 and W == 40 and neck == 48)
        compare_rn3pair_memtile = (os.environ.get('MDV6_COMPARE_RN3PAIR_MEMTILE', '0') not in ('', '0', 'false', 'False')
                and mc_rn3 == 'mc_re6_rn3' and H == 40 and W == 40 and neck == 48)
        if use_rn3pair_memtile or compare_rn3pair_memtile:
            from conv.rn3_pair_vector_memtile_runner import run_re6_rn3_pair, last_stats
            w_rep = fuse_repconv(bn_block.conv1)
            w_c2 = fuse_bn(bn_block.conv2)
            fused_out = run_re6_rn3_pair(current, w_rep, w_c2)
            if os.environ.get('MDV6_RN3PAIR_MEMTILE_VERBOSE', '0') not in ('', '0', 'false', 'False'):
                st = last_stats()
                if st is not None:
                    print(f" rn3pair_memtile kernel={st.kernel_ms:.3f}ms total={st.total_ms:.3f}ms", end="", flush=True)
            if compare_rn3pair_memtile:
                # Compare against the ordinary x1 production path, not the
                # OCB-unrolled mc_re6_rn3 artifact. The OCB artifact is known
                # to emit zeroed/suppressed small activations for some edge/OCB
                # positions; x1 matches the packed-weight CPU oracle and the
                # fused memtile path exactly at rn3pair sites.
                _saved_ocb = mcr._MERGED_LAYERS_OCB.pop(mc_rn3, None)
                try:
                    repconv_out = rt(mc_rn3, sc_rn3, current, w_rep,
                                     H, W, neck, trn3, trn3, orn3, 1, 3, 1)
                    baseline_out = rt(mc_rn3, sc_rn3, repconv_out, w_c2,
                                      H, W, neck, trn3, trn3, orn3, 1, 3, 1)
                finally:
                    if _saved_ocb is not None:
                        mcr._MERGED_LAYERS_OCB[mc_rn3] = _saved_ocb
                d = (fused_out.float() - baseline_out.float()).abs()
                print(f" rn3pair_compare max={float(d.max()):.6f} mean={float(d.mean()):.6f}", end="", flush=True)
                conv2_out = baseline_out
            else:
                conv2_out = fused_out
        else:
            repconv_out = rt(mc_rn3, sc_rn3, current, fuse_repconv(bn_block.conv1),
                             H, W, neck, trn3, trn3, orn3, 1, 3, 1)
            conv2_out = rt(mc_rn3, sc_rn3, repconv_out, fuse_bn(bn_block.conv2),
                           H, W, neck, trn3, trn3, orn3, 1, 3, 1)
        current = (residual + conv2_out) if bn_block.residual else conv2_out
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


def _setup_model():
    """Build MDV6, load trained bf16 weights, prewarm fuse_bn caches.

    One-time per process — pulled out of main() so _profile_main can call it
    once before the timed loop instead of repeating it per "frame".
    """
    print("=" * 70)
    print("MDV6 Full Model — 32-Core Multicore")
    print("=" * 70)

    model = MDV6MITYOLOv9c(num_classes=3).eval()
    _weights_path = os.path.join(os.path.dirname(__file__), 'mdv6_bf16_weights.pt')
    if not os.path.exists(_weights_path):
        _jit_path = '/home/jfifield/mdv6/mdv6.pt'
        if os.path.exists(_jit_path):
            _jit = torch.jit.load(_jit_path, map_location='cpu')
            _sd = {k.replace('model.', '', 1): v for k, v in _jit.state_dict().items()}
            model.load_state_dict(_sd, strict=False)
            print("  (loaded trained weights from TorchScript)")
    elif os.path.exists(_weights_path):
        model.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
        print("  (loaded trained bf16 weights)")
    model = model.to(torch.bfloat16)

    # Bead mlir-aie-d6f: pre-fuse every Conv+BN at model load so id-keyed
    # packer caches in run_tiled_mc.py hit from the first rt() call.
    print("  prewarming weight fusion...", end=" ", flush=True)
    _tpw = time.time()
    _n_fused = 0
    for _m in model.modules():
        if hasattr(_m, "conv") and hasattr(_m, "bn"):
            fuse_bn(_m)
            _n_fused += 1
    print(f"{_n_fused} layers in {time.time()-_tpw:.2f}s")

    return model


def _make_input():
    torch.manual_seed(42)
    return torch.randn(1, 3, 640, 640, dtype=torch.bfloat16)


def _compute_ref(model, x):
    print("\nPyTorch reference...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        ref = model(x)
    print(f"{time.time()-t0:.3f}s")
    return ref


def _forward_one_frame(model, x, _chk=lambda tag, t: None):
    """Run one NPU forward pass through MDV6. Returns the Detection output `det`.

    Hoist target: this is the only piece that should sit inside the steady-state
    profile loop. Model setup, PyTorch reference, and correctness checks all
    live outside.
    """
    print("\n--- Forward pass (32-core) ---")
    t_start = time.time()

    inp = x.squeeze(0).permute(1, 2, 0).contiguous()

    # Conv0 (640→320, 3→32, s=2)
    # Pad IC=3→8: Peano auto-vectorizes IC loop with 8-wide vectors,
    # so IC < 8 produces incorrect results. Zero-pad input and weights.
    print("  conv0...", end=" ", flush=True); t = time.time()
    inp_padded = torch.zeros(640, 640, 8, dtype=torch.bfloat16)
    inp_padded[:, :, :3] = inp
    conv0_wt = fuse_bn(model.conv0)
    oc0, ic0, ks0 = 32, 3, 3
    ic0_pad = 8
    wt_conv = conv0_wt[:oc0*ic0*ks0*ks0]
    wt_bn = conv0_wt[oc0*ic0*ks0*ks0:]
    w_orig = torch.from_numpy(wt_conv.copy()).view(torch.bfloat16).reshape(oc0, ic0, ks0, ks0)
    w_pad = torch.zeros(oc0, ic0_pad, ks0, ks0, dtype=torch.bfloat16)
    w_pad[:, :ic0, :, :] = w_orig
    conv0_wt_padded = np.concatenate([
        w_pad.flatten().view(torch.uint16).numpy(),
        wt_bn,
    ])
    conv0_hwc = rt('mc_ftconv0', 'ftconv0', inp_padded, conv0_wt_padded,
                    320, 320, 32, 20, 20, 32, 2, 3, 1)
    print(f"{time.time()-t:.3f}s"); _chk("conv0", conv0_hwc)

    # Conv1 (320→160, 32→64, s=2)
    print("  conv1...", end=" ", flush=True); t = time.time()
    conv1_hwc = rt('mc_ftconv1', 'ftconv1', conv0_hwc, fuse_bn(model.conv1),
                    160, 160, 64, 12, 12, 16, 2, 3, 1)
    print(f"{time.time()-t:.3f}s"); _chk("conv1", conv1_hwc)

    # ELAN2 (160×160, 64→64)
    print("  elan2...", end=" ", flush=True); t = time.time()
    elan2 = run_elan_mc(model.elan2, conv1_hwc, 160, 160, 64, 64,
                         'mc_elan_c1', 'tf_elan_conv1',
                         'mc_elan_c3', 'tf_elan_conv3x3',
                         'mc_elan_c4', 'tf_elan_conv4',
                         8, 64, 8, 32, 8, 64, 64, 32)
    print(f"{time.time()-t:.3f}s"); _chk("elan2", elan2)

    # AConv3 + rep_elan4 [B3]
    print("  aconv3...", end=" ", flush=True); t = time.time()
    ac3 = run_aconv_mc('mc_aconv3', 'tf_aconv3', model.aconv3, to_nchw(elan2), 80, 80, 128, 8, 16)
    print(f"{time.time()-t:.3f}s"); _chk("aconv3", ac3)
    print("  rep_elan4...", end=" ", flush=True); t = time.time()
    b3 = run_re_mc(model.rep_elan4, ac3, 80, 80, 128, 128, 128, 64,
                    'mc_re4_c1', 're4_conv1', 'mc_re4_c3', 're4_conv3x3', 'mc_re4_c4', 're4_conv4',
                    'mc_re4_rn1', 're4_rn_conv1x1_64_32', 'mc_re4_rn3', 're4_rn_conv3x3_32_32',
                    'mc_elan_c1', 'tf_elan_conv1',
                    10, 64, 12, 16, 8, 32, 16, 32, 8, 32, 8, 64)
    print(f"{time.time()-t:.3f}s"); _chk("rep_elan4", b3)

    # AConv5 + rep_elan6 [B4]
    print("  aconv5...", end=" ", flush=True); t = time.time()
    b3n = to_nchw(b3)
    ac5 = run_aconv_mc('mc_aconv5', 'aconv5', model.aconv5, b3n, 40, 40, 192, 4, 8)
    print(f"{time.time()-t:.3f}s"); _chk("aconv5", ac5)
    print("  rep_elan6...", end=" ", flush=True); t = time.time()
    b4 = run_re_mc(model.rep_elan6, ac5, 40, 40, 192, 192, 192, 96,
                    'mc_re6_c1', 're6_conv1', 'mc_re6_c3', 're6_conv3x3', 'mc_re6_c4', 're6_conv4',
                    'mc_re6_rn1', 're6_rn_c1', 'mc_re6_rn3', 're6_rn_c3',
                    'mc_re6_rnm', 're6_rn_merge',
                    8, 32, 8, 16, 4, 32, 10, 48, 8, 16, 8, 48)
    print(f"{time.time()-t:.3f}s"); _chk("rep_elan6", b4)

    # AConv7 + rep_elan8 [B5]
    print("  aconv7...", end=" ", flush=True); t = time.time()
    b4n = to_nchw(b4)
    ac7 = run_aconv_mc('mc_aconv7', 'aconv7', model.aconv7, b4n, 20, 20, 256, 4, 8)
    print(f"{time.time()-t:.3f}s"); _chk("aconv7", ac7)
    print("  rep_elan8...", end=" ", flush=True); t = time.time()
    b5 = run_re_mc(model.rep_elan8, ac7, 20, 20, 256, 256, 256, 128,
                    'mc_re8_c1', 're8_conv1', 'mc_re8_c3', 're8_conv3x3', 'mc_re8_c4', 're8_conv4',
                    'mc_re8_rn1', 're8_rn_c1', 'mc_re8_rn3', 're8_rn_c3',
                    'mc_re8_c1', 're8_rn_merge',
                    4, 32, 4, 16, 4, 16, 8, 64, 8, 16, 4, 32)
    print(f"{time.time()-t:.3f}s"); _chk("rep_elan8", b5)

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
    print(f"{time.time()-t:.3f}s"); _chk("spp9/n3", n3)

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
    print(f"{time.time()-t:.3f}s"); _chk("rep_elan12/n4", n4)

    print("  rep_elan15...", end=" ", flush=True); t = time.time()
    n4n = to_nchw(n4)
    with torch.no_grad(): up2 = nn.Upsample(scale_factor=2, mode='nearest')(n4n)
    cat15 = torch.cat([up2, b3n], dim=1).squeeze(0).permute(1,2,0).contiguous()
    p3 = run_re_mc(model.rep_elan15, cat15, 80, 80, 320, 128, 128, 64,
                    'mc_re15_c1', 're15_conv1', 'mc_re4_c3', 're4_conv3x3', 'mc_re4_c4', 're15_conv4',
                    'mc_re4_rn1', 're4_rn_conv1x1_64_32', 'mc_re4_rn3', 're4_rn_conv3x3_32_32',
                    'mc_elan_c4', 're15_rn_merge',
                    6, 32, 12, 16, 8, 32, 16, 32, 8, 32, 8, 64)
    print(f"{time.time()-t:.3f}s"); _chk("rep_elan15/p3", p3)

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
    print(f"{time.time()-t:.3f}s"); _chk("head_p4", p4)

    # Head P5
    print("  head P5...", end=" ", flush=True); t = time.time()
    p4n = to_nchw(p4)
    ac19 = run_aconv_mc('mc_aconv19', 'aconv19', model.aconv19, p4n, 20, 20, 128, 4, 8)
    cat21 = torch.cat([to_nchw(ac19), n3n], dim=1).squeeze(0).permute(1,2,0).contiguous()
    p5 = run_re_mc(model.rep_elan21, cat21, 20, 20, 384, 256, 256, 128,
                    'mc_re6_c4', 're21_conv1', 'mc_re8_c3', 're8_conv3x3', 'mc_re8_c4', 're8_conv4',
                    'mc_re8_rn1', 're8_rn_c1', 'mc_re8_rn3', 're8_rn_c3',
                    'mc_re8_c1', 're8_rn_merge',
                    4, 32, 4, 16, 4, 16, 8, 64, 8, 16, 4, 32)
    print(f"{time.time()-t:.3f}s"); _chk("head_p5", p5)

    # Detection (CPU)
    print("  detect...", end=" ", flush=True)
    p5n = to_nchw(p5)
    with torch.no_grad(): det = model.detect([p3n, p4n, p5n])
    print("CPU")

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Total forward pass: {t_total:.3f}s")
    print(f"{'='*70}")
    return det


def _check_correctness(ref, det):
    """Compare NPU detection output vs PyTorch reference. Returns PASS bool."""
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


def main():
    model = _setup_model()
    x = _make_input()
    ref = _compute_ref(model, x)

    if os.environ.get("DEBUG_GEMM_TRAINED"):
        # Test 1x1 GEMM with TRAINED elan2 weights + NPU-like small input
        print("\n--- 1x1 GEMM with trained elan2 weights ---")
        torch.manual_seed(42)
        x_orig = torch.randn(1, 3, 640, 640, dtype=torch.bfloat16)
        with torch.no_grad():
            conv0_ref = model.conv0(x_orig)
            conv1_ref = model.conv1(conv0_ref)
        # Use PyTorch conv1 output as input (this is what NPU conv1 approximates)
        inp = conv1_ref.squeeze(0).permute(1,2,0).contiguous()  # 160x160x64
        print(f"  input std={inp.float().std():.4f}")

        # elan2.conv1 is 1x1 64→64
        weights_u16 = fuse_bn(model.elan2.conv1)
        npu_out = run_gemm_conv1x1('gemm_elan_c1', 'tf_elan_conv1', inp, weights_u16, 160, 160, 64)
        with torch.no_grad():
            ref = model.elan2.conv1(inp.float().permute(2,0,1).unsqueeze(0).to(torch.bfloat16))
            ref_hwc = ref.squeeze(0).permute(1,2,0).bfloat16()
        diff = (npu_out.float() - ref_hwc.float()).abs()
        ratio = npu_out.float().std().item() / max(ref_hwc.float().std().item(), 1e-9)
        print(f"  [elan2.conv1 TRAINED] npu:std={npu_out.float().std():.4f} ref:std={ref_hwc.float().std():.4f} "
              f"ratio={ratio:.3f} max_diff={diff.max():.4f}")

        # Also check BN parameters
        bn = model.elan2.conv1.bn
        print(f"  BN params: gamma.std={bn.weight.data.float().std():.3f} beta.std={bn.bias.data.float().std():.3f} "
              f"mean.std={bn.running_mean.data.float().std():.3f} var.max={bn.running_var.data.float().max():.3f} var.min={bn.running_var.data.float().min():.4f}")
        inv_std = 1.0 / torch.sqrt(bn.running_var.data.float() + bn.eps)
        bn_w_fused = bn.weight.data.float() * inv_std
        bn_b_fused = bn.bias.data.float() - bn.weight.data.float() * bn.running_mean.data.float() * inv_std
        print(f"  fused: bn_w mean={bn_w_fused.mean():.2f} std={bn_w_fused.std():.2f} max={bn_w_fused.max():.2f} "
              f"| bn_b mean={bn_b_fused.mean():.2f} std={bn_b_fused.std():.2f} max={bn_b_fused.abs().max():.2f}")
        return True

    if os.environ.get("DEBUG_GEMM"):
        # Test 1x1 GEMM path — same sub-layers as elan2
        print("\n--- Standalone 1x1 GEMM tests (elan2 sub-layers) ---")
        torch.manual_seed(0)

        gemm_configs = [
            # (H, W, IC, OC, label)  — all 1x1 stride=1
            (160, 160, 64, 64, 'elan_c1 1x1 64→64'),
            (160, 160, 128, 64, 'elan_c4 1x1 128→64'),
            (80, 80, 128, 128, 're4_c1 1x1 128→128'),
            (80, 80, 256, 128, 're4_c4 1x1 256→128'),
        ]
        any_fail = False
        for (H, W, IC, OC, label) in gemm_configs:
            test_in = (torch.randn(H, W, IC, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            test_w = (torch.randn(OC, IC, dtype=torch.float32) * 0.1).to(torch.bfloat16)
            bn_w = torch.ones(OC, dtype=torch.bfloat16)
            bn_b = torch.zeros(OC, dtype=torch.bfloat16)
            wt_packed = np.concatenate([
                ett.bf16_to_uint16(test_w.flatten()),
                ett.bf16_to_uint16(bn_w),
                ett.bf16_to_uint16(bn_b),
            ])
            with torch.no_grad():
                ref = torch.nn.functional.conv2d(
                    test_in.permute(2,0,1).unsqueeze(0).float(),
                    test_w.reshape(OC, IC, 1, 1).float(), padding=0, stride=1)
                ref = torch.nn.functional.silu(ref)
                ref_hwc = ref.squeeze(0).permute(1,2,0).bfloat16()
            npu_out = run_gemm_conv1x1('gemm_xxx', 'sc_xxx', test_in, wt_packed, H, W, OC)
            diff = (npu_out.float() - ref_hwc.float()).abs()
            n_nan = torch.isnan(npu_out.float()).float().mean().item()
            ratio = npu_out.float().std().item() / max(ref_hwc.float().std().item(), 1e-9)
            ok = (diff.max().item() < 1.0) and (n_nan == 0) and (0.5 < ratio < 2.0)
            any_fail = any_fail or not ok
            print(f"  [{label}] ref:std={ref_hwc.float().std():.4f} npu:std={npu_out.float().std():.4f} "
                  f"ratio={ratio:.3f} diff:max={diff.max():.4f} mean={diff.mean():.4f} nan={n_nan:.3f}  "
                  f"{'PASS' if ok else 'FAIL'}")
        return not any_fail

    if os.environ.get("DEBUG_KERNEL"):
        # Standalone tests with RANDOM weights (exercise weight indexing thoroughly)
        print("\n--- Standalone 3x3 kernel tests (random weights) ---")
        torch.manual_seed(0)

        configs = [
            # (xclbin, sc, H, W, IC, OC, tile, ocb, stride, label)
            # aconv3 stride=2 with real trained weights
            ('mc_aconv3', 'tf_aconv3', 159, 159, 64, 128, 8, 16, 2, 'aconv3 stride=2 t=8'),
            ('mc_aconv5', 'aconv5', 79, 79, 96, 192, 4, 8, 2, 'aconv5 stride=2 t=4'),
            ('mc_aconv7', 'aconv7', 39, 39, 128, 256, 4, 8, 2, 'aconv7 stride=2 t=4'),
        ]
        all_pass = True
        USE_IDENTITY = os.environ.get("IDENTITY") == "1"
        for (mc, sc, H, W, IC, OC, tile, ocb, stride, label) in configs:
            test_in = (torch.randn(H, W, IC, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            if USE_IDENTITY:
                test_w = torch.zeros(OC, IC, 3, 3, dtype=torch.bfloat16)
                for i in range(min(OC, IC)):
                    test_w[i, i, 1, 1] = 1.0
            else:
                test_w = (torch.randn(OC, IC, 3, 3, dtype=torch.float32) * 0.1).to(torch.bfloat16)
            bn_w = torch.ones(OC, dtype=torch.bfloat16)
            bn_b = torch.zeros(OC, dtype=torch.bfloat16)
            wt_packed = np.concatenate([
                ett.bf16_to_uint16(test_w.flatten()),
                ett.bf16_to_uint16(bn_w),
                ett.bf16_to_uint16(bn_b),
            ])
            # Compute expected output dimensions for stride=2
            if stride == 2:
                out_h = (H + 2 - 3) // 2 + 1   # standard stride-2 conv with padding=1
                out_w = (W + 2 - 3) // 2 + 1
            else:
                out_h, out_w = H, W
            with torch.no_grad():
                ref = torch.nn.functional.conv2d(
                    test_in.permute(2,0,1).unsqueeze(0).float(),
                    test_w.float(), padding=1, stride=stride)
                ref = torch.nn.functional.silu(ref)
                ref_hwc = ref.squeeze(0).permute(1,2,0).bfloat16()
            npu_out = run_tiled_mc(mc, sc, test_in, wt_packed,
                                    out_h, out_w, OC, tile, tile, ocb, stride, 3, 1)
            diff = (npu_out.float() - ref_hwc.float()).abs()
            n_nan = torch.isnan(npu_out.float()).float().mean().item()
            ok = diff.max().item() < 1.0 and n_nan == 0
            # Per-channel stats
            npu_ch_std = npu_out.float().std(dim=(0,1))  # [OC]
            nonzero_ch = (npu_ch_std > 1e-6).sum().item()
            print(f"  [{label}] ref:std={ref_hwc.float().std():.4f} npu:std={npu_out.float().std():.4f} "
                  f"diff:max={diff.max():.4f} mean={diff.mean():.4f} nan={n_nan:.3f} "
                  f"nonzero_ch={nonzero_ch}/{OC}  {'PASS' if ok else 'FAIL'}")
            if not ok and OC <= 32:
                # Print per-channel std to spot which OCs work
                ch_stats = ", ".join(f"{npu_ch_std[i].item():.2f}" for i in range(OC))
                print(f"    per-OC std: [{ch_stats}]")
            all_pass = all_pass and ok
        return all_pass

    det = _forward_one_frame(model, x)
    return _check_correctness(ref, det)



def _profile_main(n_frames: int, baseline: str | None, out_json: str | None) -> int:
    """Run N forward passes under the Profiler; report breakdown.

    Model construction, weight load, fuse_bn prewarm, and the PyTorch
    reference forward all happen ONCE outside the profile region — these are
    one-time costs in a real video-stream pipeline and shouldn't inflate the
    per-frame buckets. Same for the correctness check, which runs after the
    timed loop. The result is that `pre_post` collapses to just the residual
    sequencing overhead around the first/last NPU launch.
    """
    import gc
    from profile_harness import Profiler

    # One-time setup outside the timed region.
    model = _setup_model()
    x = _make_input()
    ref = _compute_ref(model, x)

    n_warmup = 1 if n_frames > 1 else 0
    n_measure = n_frames - n_warmup

    last_det = None
    with Profiler(n_warmup=n_warmup, n_measure=n_measure) as prof:
        for i in range(n_frames):
            last_det = _forward_one_frame(model, x)
            if i < n_frames - 1:
                prof.next_frame()
            gc.collect()

    # Correctness check outside the timed region.
    ok = _check_correctness(ref, last_det)

    rc = prof.report(baseline_path=baseline, out_json=out_json)
    return 0 if (ok and rc == 0) else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MDV6 full-model NPU test")
    parser.add_argument("--profile", type=int, metavar="N", default=0,
                        help="Run N forward passes under the profiler "
                             "(first is warmup; remaining are measured). "
                             "Reports a per-category breakdown.")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Compare profile against this baseline JSON "
                             "(emitted by --save-baseline). Exits 1 on >10%% "
                             "regression on any category.")
    parser.add_argument("--save-baseline", type=str, default=None,
                        help="Write warm-frame profile to this JSON file.")
    args = parser.parse_args()

    if args.profile > 0:
        sys.exit(_profile_main(args.profile, args.baseline, args.save_baseline))
    sys.exit(0 if main() else 1)
