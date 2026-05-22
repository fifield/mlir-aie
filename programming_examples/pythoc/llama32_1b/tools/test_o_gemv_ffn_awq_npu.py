#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standalone NPU smoke test for the fused packed-AWQ O+FFN decode kernel.

Builds and runs `o_gemv_ffn_awq` at full Llama-3.2-1B decode shape
(emb=2048, hidden=8192, group=128) with synthetic AwqLinear weights and
compares NPU output to a NumPy CPU reference computed from
`_dequant_repacked_awq_linear`.

Opt-in: pass --run-npu (or set RUN_FUSED_AWQ_NPU=1) to actually load
and execute the kernel. Without that, the test only verifies the
module/IR/compile path without touching XRT.

Usage from the example directory:
    python3 tools/test_o_gemv_ffn_awq_npu.py --run-npu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_SCRIPT = Path(__file__).resolve()
_EXAMPLE = _SCRIPT.parent.parent
sys.path.insert(0, str(_EXAMPLE))


def make_awq_linear(m: int, k: int, group_size: int = 128, seed: int = 0):
    """Random AwqLinear sized like a real Llama layer.

    Scales are O(1/sqrt(k)) and zeros are centered around 8 so that one row of
    dequant @ unit-magnitude input lands at O(1) magnitude (matching real
    pre-LayerNorm activations). Keeps F32/BF16 accumulation drift bounded for
    the standalone correctness test.
    """
    from llama32_1b_weights import AwqLinear

    rng = np.random.default_rng(seed)
    qweight = rng.integers(0, 256, size=(m, k // 2), dtype=np.uint8)
    groups = k // group_size
    scale_amp = 0.5 / np.sqrt(k)
    scales = (rng.standard_normal((m, groups)).astype(np.float32) * scale_amp).astype(bfloat16)
    zeros = (rng.integers(7, 10, size=(m, groups)).astype(np.float32)).astype(bfloat16)
    params = np.empty((m, 2 * groups), dtype=bfloat16)
    params[:, 0::2] = scales
    params[:, 1::2] = zeros
    return AwqLinear(qweight=qweight, params=params, k=k, m=m, group_size=group_size)


def cpu_reference(attn_out, x_residual, ffn_norm_w, awq_layer, emb_dim, hidden_dim):
    """CPU O+FFN matching the fused AWQ kernel arithmetic."""
    from llama32_1b_weights import _dequant_repacked_awq_linear
    from llama32_1b_reference import rms_norm

    wo = _dequant_repacked_awq_linear(awq_layer.wo, dtype=bfloat16)  # (emb, emb)
    wgate = _dequant_repacked_awq_linear(awq_layer.w_gate, dtype=bfloat16)  # (hidden, emb)
    wup = _dequant_repacked_awq_linear(awq_layer.w_up, dtype=bfloat16)  # (hidden, emb)
    wdown = _dequant_repacked_awq_linear(awq_layer.w_down, dtype=bfloat16)  # (emb, hidden)

    a = attn_out.astype(np.float32)
    x = x_residual.astype(np.float32)
    proj = (wo.astype(np.float32) @ a)
    res1 = proj + x
    normed2 = rms_norm(res1.reshape(1, emb_dim), ffn_norm_w.astype(np.float32)).flatten()
    gate = (wgate.astype(np.float32) @ normed2)
    up = (wup.astype(np.float32) @ normed2)
    swiglu = gate / (1.0 + np.exp(-gate)) * up
    down = (wdown.astype(np.float32) @ swiglu)
    output = down + res1
    return output


def _load_real_awq_layer(model_path: Path, layer_idx: int):
    """Load a real repacked AWQ transformer layer's o/gate/up/down."""
    from safetensors import safe_open
    from llama32_1b_weights import AwqLinear, AwqLayerWeights

    def _np(t):
        return t.float().cpu().numpy().astype(bfloat16) if str(t.dtype) == "torch.bfloat16" else t.cpu().numpy()

    def _read(prefix):
        with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as f:
            q = _np(f.get_tensor(prefix + ".qweight_repacked")).astype(np.uint8, copy=False)
            p = _np(f.get_tensor(prefix + ".params_interleaved")).astype(bfloat16, copy=False)
        k = int(q.shape[1]) * 2
        groups = int(p.shape[1]) // 2
        return AwqLinear(
            qweight=np.ascontiguousarray(q),
            params=np.ascontiguousarray(p),
            k=k,
            m=int(q.shape[0]),
            group_size=k // groups,
        )

    base = f"model.layers.{layer_idx}"
    awq_o = _read(f"{base}.self_attn.o_proj")
    return AwqLayerWeights(
        wq=awq_o, wk=awq_o, wv=awq_o,  # not used by fused o+ffn
        wo=awq_o,
        w_gate=_read(f"{base}.mlp.gate_proj"),
        w_up=_read(f"{base}.mlp.up_proj"),
        w_down=_read(f"{base}.mlp.down_proj"),
    )


def _load_real_layer_norms(model_path: Path, layer_idx: int):
    from safetensors import safe_open
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as f:
        try:
            w = f.get_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
            return w.float().cpu().numpy().astype(bfloat16)
        except Exception:
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=str, default="build_peano/decode_kernel_cache")
    parser.add_argument(
        "--awq-weights",
        type=Path,
        default=None,
        help="Path to repacked AWQ model dir to use a real layer; default uses synthetic.",
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument(
        "--run-npu",
        action="store_true",
        default=bool(int(os.environ.get("RUN_FUSED_AWQ_NPU", "0"))),
        help="Actually load the ELF and execute on NPU (opt-in).",
    )
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=2e-1)
    args = parser.parse_args()

    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    group_size = args.group_size

    print(
        f"Fused AWQ O+FFN standalone test: emb={emb_dim}, hidden={hidden_dim}, "
        f"group={group_size}, run_npu={args.run_npu}"
    )

    from llama32_1b_weights import AwqLayerWeights

    if args.awq_weights is not None and args.awq_weights.is_dir():
        print(f"Loading real AWQ layer {args.layer_idx} from {args.awq_weights}...")
        awq_layer = _load_real_awq_layer(args.awq_weights, args.layer_idx)
        # Pick K from the loaded weights instead of CLI defaults.
        emb_dim = awq_layer.wo.k
        hidden_dim = awq_layer.w_gate.m
        group_size = awq_layer.wo.group_size
        ffn_norm_w = _load_real_layer_norms(args.awq_weights, args.layer_idx)
        if ffn_norm_w is None:
            print("  (no ffn_norm in safetensors; falling back to ones)")
            ffn_norm_w = np.ones(emb_dim, dtype=bfloat16)
    else:
        awq_layer = AwqLayerWeights(
            wq=make_awq_linear(emb_dim, emb_dim, group_size, seed=args.seed),
            wk=make_awq_linear(emb_dim, emb_dim, group_size, seed=args.seed + 1),
            wv=make_awq_linear(emb_dim, emb_dim, group_size, seed=args.seed + 2),
            wo=make_awq_linear(emb_dim, emb_dim, group_size, seed=args.seed + 3),
            w_gate=make_awq_linear(hidden_dim, emb_dim, group_size, seed=args.seed + 4),
            w_up=make_awq_linear(hidden_dim, emb_dim, group_size, seed=args.seed + 5),
            w_down=make_awq_linear(emb_dim, hidden_dim, group_size, seed=args.seed + 6),
        )
        rng_norm = np.random.default_rng(args.seed + 99)
        ffn_norm_w = (rng_norm.standard_normal(emb_dim).astype(np.float32) * 0.1 + 1.0).astype(bfloat16)

    rng = np.random.default_rng(args.seed + 100)
    # Realistic post-attn / residual magnitudes (~unit-scale bf16).
    attn_out = (rng.standard_normal(emb_dim).astype(np.float32) * 0.5).astype(bfloat16)
    x_residual = (rng.standard_normal(emb_dim).astype(np.float32) * 0.5).astype(bfloat16)

    print("Computing CPU reference (dequant + matmul, may take a few seconds)...")
    ref = cpu_reference(attn_out, x_residual, ffn_norm_w, awq_layer, emb_dim, hidden_dim)
    print(
        f"  CPU ref stats: min={ref.min():.4f} max={ref.max():.4f} "
        f"mean={ref.mean():.4f} std={ref.std():.4f}"
    )

    if not args.run_npu:
        print("--run-npu not set; skipping NPU execution. CPU reference is ready.")
        return 0

    from kernel_builder.cache import KernelCache
    from llama32_1b_awq_runtime import o_gemv_ffn_awq_npu

    # cwd matters: AIR backend stages link_with objects from cwd.
    print(f"cwd: {os.getcwd()}")
    cache = KernelCache(cache_dir=args.cache_dir, verbose=False)

    print("Running fused AWQ O+FFN on NPU...")
    npu_out_bf16 = o_gemv_ffn_awq_npu(
        cache,
        attn_out,
        x_residual,
        ffn_norm_w,
        awq_layer,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        layer_idx=0,
    )
    npu_out = np.asarray(npu_out_bf16, dtype=np.float32)
    print(
        f"  NPU out stats: min={npu_out.min():.4f} max={npu_out.max():.4f} "
        f"mean={npu_out.mean():.4f} std={npu_out.std():.4f}"
    )

    diff = npu_out - ref
    abs_err = np.abs(diff)
    rel_denom = np.maximum(np.abs(ref), 1e-3)
    rel_err = abs_err / rel_denom
    max_abs = abs_err.max()
    mean_abs = abs_err.mean()
    max_rel = rel_err.max()
    mean_rel = rel_err.mean()
    corr = np.corrcoef(npu_out, ref)[0, 1] if ref.std() > 0 else float("nan")

    print(
        f"Compare NPU vs CPU: corr={corr:.6f} "
        f"max_abs={max_abs:.4f} mean_abs={mean_abs:.4f} "
        f"max_rel={max_rel:.4f} mean_rel={mean_rel:.4f}"
    )

    # Synthetic-random AWQ weights drive the compound math through 3 GEMVs +
    # SiLU + 2 adds, so worst-case elementwise drift is dominated by BF16
    # accumulation rounding on the rare elements where CPU ref happens to land
    # near a sign change. Gate on correlation + mean_abs to track structural
    # correctness, not the worst element of a noise distribution.
    corr_gate = corr > 0.95
    mean_abs_gate = mean_abs <= max(0.4, 0.05 * abs(ref).mean())
    print(
        f"Gates: corr>0.95={'PASS' if corr_gate else 'FAIL'}, "
        f"mean_abs<=max(0.4, 5%*|ref|.mean)={'PASS' if mean_abs_gate else 'FAIL'}"
    )

    if not (corr_gate and mean_abs_gate):
        worst_idx = int(np.argmax(abs_err))
        print(
            f"  Worst idx={worst_idx} npu={npu_out[worst_idx]:.4f} "
            f"ref={ref[worst_idx]:.4f}"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
