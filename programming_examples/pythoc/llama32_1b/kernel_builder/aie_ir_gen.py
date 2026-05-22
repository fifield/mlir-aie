# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cached-only MLIR-AIE IR provider for the PythoC + IRON llama32_1b port.

The AIR-tree counterpart of this file runs aircc on multi_launch_builder modules
to harvest post-stitched `npu.air.mlir`. In the pythoc tree we ship the cached
IR up-front (under `reference_mlir/`) and never invoke aircc at compile time.
Every `build_*_ir(...)` here returns the matching cached text; the orchestration
hands that text to `KernelCache.compile_and_cache`, which calls aiecc.

Replacing one of these entry points with a placed-iron Python builder that
emits the same `aie/aiex`-dialect text is the Phase 4 work.
"""

import os
import sys
from pathlib import Path

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference_mlir"

# Phase 4 feature flag.  Set
# `PYTHOC_LLAMA_USE_PLACED_BUILDERS=lm_head_gemv` (or "all") to route
# build_*_ir() calls through the placed-IRON Python builder under
# ../builders/<name>.py instead of reading the cached AIR-stitched IR.
# Comma-separated list of kernel names; "all" enables every builder.
_PLACED_BUILDERS_ENV = "PYTHOC_LLAMA_USE_PLACED_BUILDERS"


def _placed_builder_enabled(name: str) -> bool:
    val = os.environ.get(_PLACED_BUILDERS_ENV, "").strip()
    if not val:
        return False
    if val == "all":
        return True
    return name in {tok.strip() for tok in val.split(",") if tok.strip()}


def _ensure_builders_on_path() -> None:
    project_root = _REFERENCE_DIR.parent
    p = str(project_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_cached(name: str) -> str:
    path = _REFERENCE_DIR / f"{name}.npu.air.mlir"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached MLIR for {name} at {path}. "
            "Seed reference_mlir/ from the AIR build_peano cache, "
            "or implement a placed-iron builder for this kernel."
        )
    return path.read_text()


# Signatures match the AIR-tree aie_ir_gen.py so prefill/decode call sites are
# drop-in compatible. The dimension arguments are accepted but unused here -
# the cached IR is already specialized for the llama32 model dimensions.


def build_rms_gemms_rope_ir(seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim,
                            *, verbose=False, omit_while_true_loop=False):
    del seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim
    del verbose, omit_while_true_loop
    return _load_cached("rms_gemms_rope")


def build_o_ffn_ir(seq_len, emb_dim, hidden_dim, *, verbose=False,
                   omit_while_true_loop=False):
    del seq_len, emb_dim, hidden_dim, verbose, omit_while_true_loop
    return _load_cached("o_ffn")


def build_flash_attn_ir(seq_len, n_heads, n_kv_heads, head_dim, *, verbose=False):
    del seq_len, n_heads, n_kv_heads, head_dim, verbose
    return _load_cached("flash_attn")


def build_rms_gemv_rope_ir(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim,
                           *, verbose=False):
    if _placed_builder_enabled("rms_gemv_rope"):
        _ensure_builders_on_path()
        from builders.rms_gemv_rope import build_rms_gemv_rope_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for rms_gemv_rope "
                  f"(emb_dim={emb_dim}, kv_dim={kv_dim}, head_dim={head_dim})")
        return build_rms_gemv_rope_module(
            emb_dim=emb_dim, kv_dim=kv_dim,
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        )
    del emb_dim, kv_dim, n_heads, n_kv_heads, head_dim, verbose
    return _load_cached("rms_gemv_rope")


def build_o_gemv_ffn_ir(emb_dim, hidden_dim, *, verbose=False):
    if _placed_builder_enabled("o_gemv_ffn"):
        _ensure_builders_on_path()
        from builders.o_gemv_ffn import build_o_gemv_ffn_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for o_gemv_ffn "
                  f"(emb_dim={emb_dim}, hidden_dim={hidden_dim})")
        return build_o_gemv_ffn_module(
            emb_dim=emb_dim, hidden_dim=hidden_dim,
        )
    del emb_dim, hidden_dim, verbose
    return _load_cached("o_gemv_ffn")


def build_lm_head_gemv_ir(emb_dim, *, verbose=False):
    if _placed_builder_enabled("lm_head_gemv"):
        _ensure_builders_on_path()
        from builders.lm_head_gemv import build_lm_head_gemv_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for lm_head_gemv "
                  f"(emb_dim={emb_dim})")
        return build_lm_head_gemv_module(emb_dim=emb_dim)
    del verbose
    return _load_cached("lm_head_gemv")


def build_o_gemv_ffn_awq_ir(emb_dim, hidden_dim, *, group_size=128, verbose=False):
    del emb_dim, hidden_dim, group_size, verbose
    return _load_cached("o_gemv_ffn_awq")


def build_awq_gemv_ir(k, m, group_size, *, verbose=False):
    """Packed uint4 AWQ GEMV primitive (deferred to Phase 6)."""
    del verbose
    return _load_cached(f"awq_gemv_k{int(k)}_m{int(m)}_g{int(group_size)}")
