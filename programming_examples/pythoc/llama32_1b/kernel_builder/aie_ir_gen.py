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
    "rms_gemv_rope_awq", # Phase 6 -- packed-AWQ Q/K/V GEMV + RMS + RoPE decode
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


# ---------------------------------------------------------------------------
# Decode device-packing pack modes.
#
# These default to the proven-good packed variants validated end-to-end in
# DEVICE_PACKING_ANALYSIS.md §14/§15 (bit-exact tokens, passes hf-gate,
# +6.7% decode throughput from 232 -> 104 segment dispatches/token):
#
#   o_gemv_ffn    -> "d1d3d4_rms" (8 -> 3 devices; d1d3d4 4-dev + RMS fold)
#   rms_gemv_rope -> "rgr1_ddr" (6 -> 1 device; rgr2_ddr 6->2 + RMS fold)
#
# Set the corresponding env var to "none" to revert a kernel to its unpacked
# baseline (e.g. for A/B regression-testing). The resolved mode is recorded
# in the decode kernel-cache manifest so a stale ELF built under a different
# mode is auto-rebuilt rather than silently reused (see cache.py).
# ---------------------------------------------------------------------------
_O_GEMV_FFN_PACK_ENV = "PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE"
_RMS_GEMV_ROPE_PACK_ENV = "PYTHOC_LLAMA_RMS_GEMV_ROPE_PACK_MODE"
# c2_merged is the full call-2 collapse: O / add1 / rms-fold / gate / up /
# swiglu / down / add2 all in ONE aie.device / ONE aiex.configure = 1 LoadPDI
# (was 3 under d1d3d4_rms). Bit-exact vs d1d3d4_rms / unpacked (hf-gate 9/9),
# ~87 ms/token (structural win, perf-neutral). The mat activation ships
# per-column via the mem-tiles (no shim-row broadcast fan) and every shim
# S2MM0 producer uses a distinct output packet id (matvec-y=1/add=5/swiglu=6/
# down=7) so the output convergence has no multi-producer routing conflict.
# Set the env var to "d1d3d4_rms"/"d1d3d4"/"none" to step back down the
# collapse ladder (see RESIDENT_DEVICE_EVOLUTION.md).
_O_GEMV_FFN_PACK_DEFAULT = "c2_merged"
# rgr1_ddr folds the standalone r_rms_seg RMSNorm device into the Q/K/V+RoPE
# pack (6->1 device for decode call 1), one fewer full-device LoadPDI/layer.
# Bit-exact + perf-neutral vs rgr2_ddr (same kernels, same DDR handoff); the
# value is structural -- another stage made device-resident on the path to a
# persistent spatial layer. Set the env var to "rgr2_ddr" or "none" to revert.
_RMS_GEMV_ROPE_PACK_DEFAULT = "rgr1_ddr"

# AWQ O+FFN decode packing: validated on hardware (Paris, ~14.5 tok/s).
# Defaults to `c2_merged` -- the full call-2 collapse ported from the BF16
# builder: O / add1 / rms-fold / gate / up / swiglu / down / add2 all in ONE
# aie.device / ONE aiex.configure = 1 LoadPDI (was 3 under d1d3d4_rms). Same
# fixes as BF16 C2 (per-column mem-tile activation, distinct shim-S2MM0 output
# packet ids matvec-y=1/add=5/swiglu=6/down=7) with uint4-dequant matvec
# kernels. Step back with "d1d3d4_rms"/"d1d3d4"/"none".
_O_GEMV_FFN_AWQ_PACK_ENV = "PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE"
_O_GEMV_FFN_AWQ_PACK_DEFAULT = "c2_merged"

# AWQ RMS+GEMV+RoPE decode packing: validated on hardware (passes `make
# hf-gate QUANT=awq`; combined with the O+FFN pack, AWQ decode goes
# 9.90 -> 10.75 tok/s, +8.6%), so it defaults to the 6->2 rgr2_ddr pack.
# Set the env var to "none" to revert.
_RMS_GEMV_ROPE_AWQ_PACK_ENV = "PYTHOC_LLAMA_RMS_GEMV_ROPE_AWQ_PACK_MODE"
# rgr1_ddr folds r_rms_awq_seg into the AWQ Q/K/V+RoPE pack (call 1 -> 1
# device), the AWQ counterpart of the BF16 rgr1_ddr fold. RMSNorm stays BF16;
# bit-exact + structural. Set to "rgr2_ddr" (2 dev) or "none" (6 dev) to revert.
_RMS_GEMV_ROPE_AWQ_PACK_DEFAULT = "rgr1_ddr"


def _resolve_pack_mode(env_var: str, default: str) -> str:
    return (os.environ.get(env_var, default).strip() or default)


def decode_pack_modes() -> dict:
    """Resolve the decode device-packing modes from the environment.

    Returns a ``{kernel_name: pack_mode}`` dict for the packable BF16 decode
    kernels, applying the packed-by-default values above. Used both to build
    the IR and to compute the manifest cache signature so packed/unpacked
    ELFs are auto-distinguished.
    """
    return {
        "o_gemv_ffn": _resolve_pack_mode(
            _O_GEMV_FFN_PACK_ENV, _O_GEMV_FFN_PACK_DEFAULT),
        "rms_gemv_rope": _resolve_pack_mode(
            _RMS_GEMV_ROPE_PACK_ENV, _RMS_GEMV_ROPE_PACK_DEFAULT),
    }


def o_gemv_ffn_awq_pack_mode() -> str:
    """Resolve the AWQ ``o_gemv_ffn_awq`` device-packing mode from the env.

    Unlike the BF16 ``decode_pack_modes`` dict (which feeds the BF16 manifest
    signature), the AWQ pack mode is resolved at the AWQ IR-build call site.
    This helper exposes the SAME resolution to the host so the decode loop can
    detect ``c2_attn`` (on-NPU attention wave 0) and route to the AWQ
    attention packer instead of CPU attention.
    """
    return _resolve_pack_mode(
        _O_GEMV_FFN_AWQ_PACK_ENV, _O_GEMV_FFN_AWQ_PACK_DEFAULT)


def rms_gemv_rope_awq_pack_mode() -> str:
    """Resolve the AWQ ``rms_gemv_rope_awq`` device-packing mode from the env.

    Mirror of ``o_gemv_ffn_awq_pack_mode`` -- exposes the resolution so the AWQ
    manifest signature can auto-distinguish ELFs built under different pack
    modes (a pack-mode toggle invalidates the cached binary across processes).
    """
    return _resolve_pack_mode(
        _RMS_GEMV_ROPE_AWQ_PACK_ENV, _RMS_GEMV_ROPE_AWQ_PACK_DEFAULT)


def attn_hp_enabled() -> bool:
    """True when the bf16-MAC high-precision attention variant is selected
    (PYTHOC_ATTN_HP). It changes which attention matmul symbols the c2_attn
    device links, so it must factor into the decode cache signature -- toggling
    it has to invalidate the cached ELF across processes (else a stale-precision
    binary is silently reused). Only meaningful under the c2_attn pack mode."""
    return os.environ.get("PYTHOC_ATTN_HP", "").strip() not in ("", "0")


def o_gemv_ffn_cache_config() -> dict:
    """Decode cache signature for the BF16 o_gemv_ffn slot (pack mode + the
    attn-hp flag when c2_attn folds attention in)."""
    cfg = {"pack_mode": _resolve_pack_mode(
        _O_GEMV_FFN_PACK_ENV, _O_GEMV_FFN_PACK_DEFAULT)}
    if cfg["pack_mode"] == "c2_attn":
        cfg["attn_hp"] = attn_hp_enabled()
    return cfg


def o_gemv_ffn_awq_cache_config() -> dict:
    """Decode cache signature for the AWQ o_gemv_ffn_awq slot (pack mode + the
    attn-hp flag when c2_attn folds attention in)."""
    cfg = {"pack_mode": o_gemv_ffn_awq_pack_mode()}
    if cfg["pack_mode"] == "c2_attn":
        cfg["attn_hp"] = attn_hp_enabled()
    return cfg


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
        pack_mode = decode_pack_modes()["rms_gemv_rope"]
        if verbose:
            suffix = f", pack_mode={pack_mode}" if pack_mode != "none" else ""
            print(f"  [aie_ir_gen] Using placed-IRON builder for rms_gemv_rope "
                  f"(emb_dim={emb_dim}, kv_dim={kv_dim}, head_dim={head_dim}{suffix})")
        return build_rms_gemv_rope_module(
            emb_dim=emb_dim, kv_dim=kv_dim,
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            pack_mode=pack_mode,
        )
    del emb_dim, kv_dim, n_heads, n_kv_heads, head_dim, verbose
    return _load_cached("rms_gemv_rope")


def build_o_gemv_ffn_ir(emb_dim, hidden_dim, *, verbose=False):
    if _placed_builder_enabled("o_gemv_ffn"):
        _ensure_builders_on_path()
        from builders.o_gemv_ffn import build_o_gemv_ffn_module
        pack_mode = decode_pack_modes()["o_gemv_ffn"]
        if verbose:
            suffix = f", pack_mode={pack_mode}" if pack_mode != "none" else ""
            print(f"  [aie_ir_gen] Using placed-IRON builder for o_gemv_ffn "
                  f"(emb_dim={emb_dim}, hidden_dim={hidden_dim}{suffix})")
        return build_o_gemv_ffn_module(
            emb_dim=emb_dim, hidden_dim=hidden_dim, pack_mode=pack_mode,
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


def build_rms_gemv_rope_awq_ir(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim,
                                *, group_size=128, verbose=False):
    """Packed-AWQ RMS + Q/K/V GEMV + RoPE decode IR.

    Same 6-segment topology as ``rms_gemv_rope`` but Q/K/V GEMVs read
    packed-uint4 AWQ weight matrices and call ``awq_mv_pythoc.o``.
    RMSNorm and RoPE stay BF16.
    """
    if _placed_builder_enabled("rms_gemv_rope_awq"):
        _ensure_builders_on_path()
        from builders.rms_gemv_rope_awq import build_rms_gemv_rope_awq_module
        pack_mode = _resolve_pack_mode(
            _RMS_GEMV_ROPE_AWQ_PACK_ENV, _RMS_GEMV_ROPE_AWQ_PACK_DEFAULT)
        if verbose:
            suffix = f", pack_mode={pack_mode}" if pack_mode != "none" else ""
            print(f"  [aie_ir_gen] Using placed-IRON builder for rms_gemv_rope_awq "
                  f"(emb_dim={emb_dim}, kv_dim={kv_dim}, "
                  f"group_size={group_size}{suffix})")
        return build_rms_gemv_rope_awq_module(
            emb_dim=emb_dim, kv_dim=kv_dim,
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            group_size=group_size, pack_mode=pack_mode,
        )
    del emb_dim, kv_dim, n_heads, n_kv_heads, head_dim, group_size, verbose
    return _load_cached("rms_gemv_rope_awq")


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
        pack_mode = _resolve_pack_mode(
            _O_GEMV_FFN_AWQ_PACK_ENV, _O_GEMV_FFN_AWQ_PACK_DEFAULT)
        if verbose:
            suffix = f", pack_mode={pack_mode}" if pack_mode != "none" else ""
            print(f"  [aie_ir_gen] Using placed-IRON builder for o_gemv_ffn_awq "
                  f"(emb_dim={emb_dim}, hidden_dim={hidden_dim}, "
                  f"group_size={group_size}{suffix})")
        return build_o_gemv_ffn_awq_module(
            emb_dim=emb_dim, hidden_dim=hidden_dim,
            group_size=group_size, pack_mode=pack_mode, verbose=verbose,
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
