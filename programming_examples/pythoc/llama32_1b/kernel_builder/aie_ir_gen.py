# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MLIR-AIE IR provider for the PythoC + IRON llama32_1b port.

By default every `build_*_ir(...)` here dispatches to a placed-IRON Python
builder under `../builders/<name>.py`, which emits the `aie/aiex`-dialect
text directly from Python. The orchestration hands that text to
`KernelCache.compile_and_cache`, which calls aiecc.

The cached AIR-emitted MLIR under `reference_mlir/` is kept as a fallback
substrate for two reasons:
  1. The 4 GEMM devices in `o_ffn` (og/dg/gg/ug) are spliced from cached
     by `builders/o_ffn.py` -- pending future debugging of a hang/garbage
     issue that doesn't appear in the structurally-identical
     `rms_gemms_rope::v_matmul_seg` device. See README "Phase 4 status".
  2. Setting `PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` forces every builder
     onto the cached path -- useful for A/B regression-testing.

The pythoc tree never invokes aircc at compile time; Phase 6 Stage 4
removed the last aircc shell-out (the AIR-tree AWQ-builder fallback).
"""

import os
import sys
from pathlib import Path

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference_mlir"


# Override which builders use the placed-IRON Python path vs the cached
# `reference_mlir/<name>.npu.air.mlir` substrate. Default (env unset): every
# builder in `_DEFAULT_PLACED_BUILDERS` is placed-IRON.
#
# Env var values:
#   unset / empty   -> default set below (all six current builders)
#   "all"           -> identical to default; kept for backwards-compat
#   "cached"/"none" -> force every builder onto the cached MLIR substrate
#   "n1,n2,..."     -> explicit allowlist; only these builders are placed-IRON
_PLACED_BUILDERS_ENV = "PYTHOC_LLAMA_USE_PLACED_BUILDERS"

# Builders that default to placed-IRON. Phase 4.6's `o_ffn` is included even
# though it splices 4 GEMM devices from cached MLIR -- the splice is internal
# to `builders/o_ffn.py` and is transparent to call sites here. Phase 6 AWQ
# entry points stay cached-only and are not listed.
_DEFAULT_PLACED_BUILDERS = frozenset({
    "lm_head_gemv",    # Phase 4.1
    "flash_attn",      # Phase 4.2
    "rms_gemv_rope",   # Phase 4.3
    "o_gemv_ffn",      # Phase 4.4
    "rms_gemms_rope",  # Phase 4.5 (v/k/q matmul stride bug fixed)
    "o_ffn",           # Phase 4.6 (5 of 9 devices placed; 4 GEMM devices spliced)
    "o_gemv_ffn_awq",  # Phase 6 -- fused AWQ uint4 O+FFN decode (Stage 3 Subtask A)
    "awq_matvec",      # Phase 6 -- standalone AWQ GEMV (dim-specialized)
    "lm_head_gemv_awq", # Phase 6 -- packed-AWQ LM head GEMV (8 partitions)
})


def _placed_builder_enabled(name: str) -> bool:
    val = os.environ.get(_PLACED_BUILDERS_ENV, "").strip()
    if not val:
        return name in _DEFAULT_PLACED_BUILDERS
    if val == "all":
        return True
    if val.lower() in ("cached", "none", "off"):
        return False
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
    if _placed_builder_enabled("rms_gemms_rope"):
        _ensure_builders_on_path()
        from builders.rms_gemms_rope import build_rms_gemms_rope_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for rms_gemms_rope "
                  f"(seq_len={seq_len}, emb_dim={emb_dim}, kv_dim={kv_dim}, "
                  f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim})")
        return build_rms_gemms_rope_module(
            seq_len=seq_len, emb_dim=emb_dim, kv_dim=kv_dim,
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            verbose=verbose,
        )
    del seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim
    del verbose, omit_while_true_loop
    return _load_cached("rms_gemms_rope")


def build_o_ffn_ir(seq_len, emb_dim, hidden_dim, *, verbose=False,
                   omit_while_true_loop=False):
    if _placed_builder_enabled("o_ffn"):
        _ensure_builders_on_path()
        from builders.o_ffn import build_o_ffn_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for o_ffn "
                  f"(seq_len={seq_len}, emb_dim={emb_dim}, hidden_dim={hidden_dim})")
        return build_o_ffn_module(
            seq_len=seq_len, emb_dim=emb_dim, hidden_dim=hidden_dim,
            verbose=verbose, omit_while_true_loop=omit_while_true_loop,
        )
    del seq_len, emb_dim, hidden_dim, verbose, omit_while_true_loop
    return _load_cached("o_ffn")


def build_flash_attn_ir(seq_len, n_heads, n_kv_heads, head_dim, *, verbose=False):
    if _placed_builder_enabled("flash_attn"):
        _ensure_builders_on_path()
        from builders.flash_attn import build_flash_attn_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for flash_attn "
                  f"(seq_len={seq_len}, n_heads={n_heads}, "
                  f"n_kv_heads={n_kv_heads}, head_dim={head_dim})")
        return build_flash_attn_module(
            seq_len=seq_len, n_heads=n_heads,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            verbose=verbose,
        )
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


def build_lm_head_gemv_awq_ir(emb_dim, *, verbose=False):
    """Packed-AWQ LM head GEMV IR.

    8 partitions, each handling 16384 rows of the 128256-row vocab matrix
    via packed-uint4 weights (ui8[16384, K/2 + 4*groups]) + scale/zero
    params interleaved per group.  Calls into ``awq_mv_pythoc.o``.
    """
    if _placed_builder_enabled("lm_head_gemv_awq"):
        _ensure_builders_on_path()
        from builders.lm_head_gemv_awq import build_lm_head_gemv_awq_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for lm_head_gemv_awq "
                  f"(emb_dim={emb_dim})")
        return build_lm_head_gemv_awq_module(emb_dim=emb_dim)
    del verbose
    return _load_cached("lm_head_gemv_awq")


def build_o_gemv_ffn_awq_ir(emb_dim, hidden_dim, *, group_size=128, verbose=False):
    """Fused packed-AWQ O+FFN decode kernel IR.

    Phase 6 / Stage 3 Subtask A: when ``o_gemv_ffn_awq`` is enabled (the
    default), emit the module directly from ``builders/o_gemv_ffn_awq.py``.
    Otherwise fall back to the cached AIR-stitched MLIR.
    """
    if _placed_builder_enabled("o_gemv_ffn_awq"):
        _ensure_builders_on_path()
        from builders.o_gemv_ffn_awq import build_o_gemv_ffn_awq_module
        if verbose:
            print(f"  [aie_ir_gen] Using placed-IRON builder for o_gemv_ffn_awq "
                  f"(emb_dim={emb_dim}, hidden_dim={hidden_dim}, "
                  f"group_size={group_size})")
        return build_o_gemv_ffn_awq_module(
            emb_dim=emb_dim, hidden_dim=hidden_dim,
            group_size=group_size, verbose=verbose,
        )
    del emb_dim, hidden_dim, group_size, verbose
    return _load_cached("o_gemv_ffn_awq")


def build_awq_gemv_ir(k, m, group_size, *, variant="vecdeq", verbose=False):
    """Packed uint4 AWQ GEMV primitive.

    Phase 6: emits aie/aiex dialect from ``builders/awq_matvec.py``
    (placed-IRON, default).  Force ``PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached``
    to fall back to the seeded ``reference_mlir/<name>.npu.air.mlir``.
    Only the ``vecdeq`` variant is supported after Stage 4 cleanup --
    the AIR-tree builder that produced scalar IR was deleted.
    """
    if _placed_builder_enabled("awq_matvec"):
        _ensure_builders_on_path()
        from builders.awq_matvec import build_awq_matvec_module
        if verbose:
            print(
                f"  [aie_ir_gen] Using placed-IRON builder for awq_matvec "
                f"(k={k}, m={m}, group_size={group_size}, variant={variant})"
            )
        return build_awq_matvec_module(
            k=k, m=m, group_size=group_size, variant=variant,
            verbose=verbose,
        )

    cache_name = f"awq_gemv_k{int(k)}_m{int(m)}_g{int(group_size)}_{variant}"
    cache_path = _REFERENCE_DIR / f"{cache_name}.npu.air.mlir"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached MLIR for {cache_name} at {cache_path}. "
            "Either seed reference_mlir/ for this shape or unset "
            "PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached so the placed-IRON "
            "builder handles it."
        )
    if verbose:
        print(f"  [aie_ir_gen] Using cached MLIR for {cache_name}")
    return cache_path.read_text()
