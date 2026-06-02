#!/usr/bin/env python3
"""Absolute-truth correctness check for GEMM 1x1 ELFs (vs torch reference).

The bytewise tests (test_pair.py, test_packed_gemm.py) compare one ELF
against another ELF. That validates ABI/layout consistency but every
chain is anchored to "the GEMM kernel is correct". This test computes
a float-precision reference in torch and compares the NPU output to it,
allowing a bf16-appropriate relative tolerance.

Picks one non-K-blocked shape and one K-blocked shape so both kernel
variants in kernels/rep_elan_bf16.cc have a truth anchor.

Run from the mdv6 dir:
  source env.sh && source venv/bin/activate
  flock /tmp/npu-dev.lock python3 conv/test_gemm_truth.py --shape all
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyxrt as xrt
from build_merged import build_merged, _resolve_build_dir

N_CORES = 32
_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "gemm_conv1x1", "aie2_gemm_conv1x1.py"))


# (label, tile_m, ic, oc, k_block, ppc) — k_block=0 means non-K-blocked path.
# One shape per distinct k_block value used by the model so the K-blocked
# kernel has an absolute-truth anchor at every operating point:
#
#   kb16  → re4_c4   (n_kb=16, smallest k_block, longest accumulator chain)
#   kb32  → re6_c1   (n_kb=6)
#   kb48  → re6_c4   (n_kb=8, only kb48 layer in the model)
#   kb64  → re8_c1   (n_kb=4)
#   kb72  → re18_c1  (n_kb=4, only kb72 layer)
#   kb128 → spp_c1   (n_kb=2, shortest accumulator chain)
_SHAPES = [
    ("re6_rn1",  164,  96,  48,   0, 1),   # non-K-blocked
    ("re4_c4",    68, 256, 128,  16, 1),
    ("re6_c1",    56, 192, 192,  32, 1),
    ("re6_c4",    32, 384, 192,  48, 2),
    ("re8_c1",    20, 256, 256,  64, 1),
    ("re18_c1",   28, 288, 192,  72, 2),
    ("spp_c1",    28, 256, 128, 128, 1),
]


def _shape(label):
    for t in _SHAPES:
        if t[0] == label:
            return t
    raise KeyError(label)


def _bf16_to_u16(t):
    return t.to(torch.bfloat16).view(torch.uint16).numpy()


def _u16_to_bf16(a):
    return torch.from_numpy(a.copy()).view(torch.bfloat16)


def _pack_weights_blocked(conv_wt_f, bn_w_f, bn_b_f):
    """Pack non-K-blocked GEMM: [ic/8, oc/8, 8ic, 8oc] + bn_w + bn_b.

    Mirrors run_tiled_mc._repack_weights_for_gemm. The kernel reads weights
    in this blocked layout to feed aie::mmul<4,8,8>.
    """
    oc, ic = conv_wt_f.shape
    oc_blks = oc // 8
    ic_blks = ic // 8
    blocked = conv_wt_f.reshape(oc_blks, 8, ic_blks, 8).permute(2, 0, 3, 1).contiguous()
    return np.concatenate([
        _bf16_to_u16(blocked.flatten()),
        _bf16_to_u16(bn_w_f),
        _bf16_to_u16(bn_b_f),
    ])


def _pack_weights_kblocked(conv_wt_f, bn_w_f, bn_b_f, k_block):
    """Pack to K-blocked layout: n_kb × [k_block/8, oc/8, 8ic, 8oc, bn_w, bn_b].

    conv_wt_f shape: [oc, ic] bf16. Mirrors run_tiled_mc._repack_weights_kblocked.
    """
    oc, ic = conv_wt_f.shape
    n_kb = ic // k_block
    chunks = []
    for kb in range(n_kb):
        # [oc, k_block] slice for this chunk
        sl = conv_wt_f[:, kb * k_block:(kb + 1) * k_block].contiguous()
        # Reshape [oc, k_block] = [oc/8, 8, kb/8, 8] → permute (2, 0, 3, 1)
        oc_blks = oc // 8
        kb_blks = k_block // 8
        blocked = sl.reshape(oc_blks, 8, kb_blks, 8).permute(2, 0, 3, 1).contiguous()
        chunk = np.concatenate([
            _bf16_to_u16(blocked.flatten()),
            _bf16_to_u16(bn_w_f),
            _bf16_to_u16(bn_b_f),
        ])
        chunks.append(chunk)
    return np.concatenate(chunks)


def _silu_kernel(x):
    """Kernel's SiLU approximation: x * (0.5 + x / (2 + 2|x|)).

    Mirrors the inner loop in conv3x3_fused_packed_bf16 and
    gemm_conv1x1_kblocked_bf16. Differs from torch.nn.functional.silu
    (x * sigmoid(x)) by ~0.5% relative.
    """
    ax = torch.abs(x)
    return x * (0.5 + x / (2.0 + 2.0 * ax))


def _torch_reference(in_f, wt_f, bn_w_f, bn_b_f):
    """Compute the same fused matmul + BN + SiLU the kernel does, in float32.

    Returns bf16 output matching what the kernel writes to memory.
    """
    # All inputs are torch tensors (bf16).
    # Matmul in float32 to mirror the AIE float accumulator.
    in_f32 = in_f.to(torch.float32)
    wt_f32 = wt_f.to(torch.float32)
    matmul = in_f32 @ wt_f32.T              # [m, oc]
    # Cast back to bf16 (matches acc.to_vector<bfloat16>() per K-block boundary
    # and after the full matmul in non-K-blocked path).
    matmul_bf16 = matmul.to(torch.bfloat16).to(torch.float32)
    bn = matmul_bf16 * bn_w_f.to(torch.float32)
    bn_bf16 = bn.to(torch.bfloat16).to(torch.float32)
    bn_added = bn_bf16 + bn_b_f.to(torch.float32)
    bn_added_bf16 = bn_added.to(torch.bfloat16).to(torch.float32)
    out = _silu_kernel(bn_added_bf16)
    return out.to(torch.bfloat16)


def _elf_name(label, tile_m, ic, oc, k_block, ppc):
    kb_str = f"kb{k_block}_" if k_block > 0 else ""
    return f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_{kb_str}p{ppc}_x1"


def _ensure_elf(elf_name, tile_m, ic, oc, k_block, ppc):
    elf = os.path.join(_resolve_build_dir(), f"{elf_name}.elf")
    if os.path.exists(elf):
        return elf
    print(f"  building {elf_name}.elf (missing)...")
    sub_label = elf_name[len("merged_"):-len("_x1")]
    sub_args = [str(N_CORES), str(tile_m), str(ic), str(oc), str(ppc), str(k_block)]
    path = build_merged(
        elf_name, [sub_label], kind="gemm", share_arg_idxs={1},
        sub_spec_overrides={sub_label: (_GEMM_SCRIPT, sub_args)},
    )
    if path is None:
        raise RuntimeError(f"build failed: {elf_name}")
    return path


def _bo_fill(bo, arr_u16):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr_u16.nbytes),
              np.frombuffer(arr_u16, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _run_one(label, device):
    _, tile_m, ic, oc, k_block, ppc = _shape(label)
    print(f"[{label}] tile_m={tile_m} ic={ic} oc={oc} "
          f"k_block={k_block if k_block else 'NONE'} ppc={ppc}")
    elf_name = _elf_name(label, tile_m, ic, oc, k_block, ppc)
    elf_path = _ensure_elf(elf_name, tile_m, ic, oc, k_block, ppc)

    # Synthetic but well-conditioned inputs. Small-magnitude bf16 values
    # keep the SiLU approximation linearizable and avoid clipping.
    torch.manual_seed(2026)
    in_per_core = N_CORES * ppc
    # Per-core input shape: [tile_m, ic]; total host: in_per_core × tile_m × ic.
    in_f = (torch.randn(in_per_core, tile_m, ic, dtype=torch.float32) * 0.3).to(torch.bfloat16)
    wt_f = (torch.randn(oc, ic, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    bn_w_f = (torch.ones(oc, dtype=torch.float32) + 0.05 * torch.randn(oc)).to(torch.bfloat16)
    bn_b_f = (torch.zeros(oc, dtype=torch.float32) + 0.05 * torch.randn(oc)).to(torch.bfloat16)

    # NPU host buffers — match aie2_gemm_conv1x1.py's host layout.
    in_u16 = _bf16_to_u16(in_f.flatten().contiguous())
    if k_block > 0:
        wt_u16 = _pack_weights_kblocked(wt_f, bn_w_f, bn_b_f, k_block)
    else:
        wt_u16 = _pack_weights_blocked(wt_f, bn_w_f, bn_b_f)
    out_nelem = in_per_core * tile_m * oc

    # Dispatch.
    elf_obj = xrt.elf(elf_path)
    kernel = xrt.ext.kernel(xrt.hw_context(device, elf_obj), "main")
    wt_bo = xrt.ext.bo(device, wt_u16.nbytes)
    in_bo = xrt.ext.bo(device, in_u16.nbytes)
    out_bo = xrt.ext.bo(device, out_nelem * 2)
    _bo_fill(wt_bo, wt_u16)
    _bo_fill(in_bo, in_u16)
    r = xrt.run(kernel)
    r.set_arg(0, wt_bo)
    r.set_arg(1, in_bo)
    r.set_arg(2, out_bo)
    r.start()
    r.wait2()
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    npu_out_u16 = np.frombuffer(out_bo.map(), dtype=np.uint16,
                                 count=out_nelem).copy()
    npu_out = _u16_to_bf16(npu_out_u16).reshape(in_per_core, tile_m, oc)

    # Torch reference (per-core, then concat).
    ref = torch.zeros_like(npu_out)
    for c in range(in_per_core):
        ref[c] = _torch_reference(in_f[c], wt_f, bn_w_f, bn_b_f)

    diff = (npu_out.to(torch.float32) - ref.to(torch.float32)).abs()
    ref_mag = ref.to(torch.float32).abs().max().item()
    max_abs = diff.max().item()
    rel = max_abs / max(ref_mag, 1e-6)
    # Tolerance depends on the accumulation path. bf16 has ~8b mantissa
    # (~0.4% step). The non-K-blocked kernel runs one fp32 accumulator over
    # all IC; only the matmul result gets quantized once. The K-blocked
    # kernel stores bf16 partial sums between each K-block, adding one
    # quantization step per chunk. With n_kb chunks the worst-case relative
    # error scales ~sqrt(n_kb)*eps; allow generous headroom.
    tol_pct = 6.0 if k_block > 0 else 3.0
    print(f"  max |NPU - torch| = {max_abs:.5f}  "
          f"(rel-to-max = {rel*100:.2f}%, max_ref = {ref_mag:.3f}, "
          f"tol = {tol_pct:.1f}%)")
    if rel * 100 < tol_pct:
        print(f"[{label}] PASS")
        return True
    print(f"[{label}] FAIL — relative error {rel*100:.2f}% exceeds {tol_pct:.1f}%")
    above = (diff > tol_pct / 100.0 * ref_mag).nonzero()[:6]
    for idx in above:
        c, m, oc_i = idx.tolist()
        print(f"    [{c},{m},{oc_i}] npu={npu_out[c,m,oc_i].item():.4f} "
              f"ref={ref[c,m,oc_i].item():.4f}")
    return False


def main():
    p = argparse.ArgumentParser()
    labels = [t[0] for t in _SHAPES]
    p.add_argument("--shape", choices=labels + ["all"], default="all")
    args = p.parse_args()

    selected = labels if args.shape == "all" else [args.shape]
    device = xrt.device(0)
    results = []
    for label in selected:
        try:
            ok = _run_one(label, device)
        except Exception as e:
            print(f"[{label}] FAIL: {e}")
            ok = False
        results.append((label, ok))
        print()
    n_pass = sum(1 for _, ok in results if ok)
    print(f"=== {n_pass}/{len(results)} shapes match torch reference ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
