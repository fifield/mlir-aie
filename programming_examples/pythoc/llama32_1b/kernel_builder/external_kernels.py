# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""External-kernel staging for the PythoC + IRON llama32_1b port.

Phase 1 ships the reference C++-built `.o` files under `build/` (copied from
the AIR worktree's build_peano). `compile_all_external_kernels` is a verify
helper that surfaces a clear error if a required `.o` is missing.

In Phase 3 each `.o` here gets superseded by a PythoC `@aie_kernel` whose
`PythocKernel` instance generates the matching object file at compile time -
at which point the cached MLIR is patched to `link_with = "<kernel>.o"` to
pick up the PythoC build instead.
"""

import os
from pathlib import Path

_BUILD_DIR = Path(__file__).resolve().parent.parent / "reference_o"


# Object files still on the AIR reference (no PythoC port yet) -- staged
# from reference_o/ into CWD so aiecc's link step finds them. As kernels
# get ported, their entry moves to _PYTHOC_KERNELS below.
_REQUIRED_OBJS = [
    # Phase 3.4 complete: attn.o / attn_npu2.o / attn_decode_npu2.o are
    # all superseded by attn_pythoc.o; nothing left on the AIR reference.
]

# PythoC-built kernels - compiled lazily from kernels/*.py into CWD.
_PYTHOC_KERNELS = [
    ("rms_norm_2048_bf16.o", "compile_rms_norm"),
    ("silu_and_mul_bf16.o", "compile_silu_and_mul"),
    ("rope_pythoc.o", "compile_rope"),
    ("mv_pythoc.o", "compile_matvec"),
    ("mv_pythoc.ll", "compile_matvec_inline"),  # inlined alwaysinline matvec
    ("matvec_rms_pythoc.o", "compile_matvec_rms"),
    # inlined alwaysinline .ll variant of rms_norm_packed (link_with=.ll
    # -> aiecc llvm-links + inlines, no func.call).
    ("matvec_rms_pythoc.ll", "compile_matvec_rms_inline"),
    ("mv_k8192_pythoc.o", "compile_matvec_k8192"),
    ("attn_pythoc.o", "compile_attn"),
    # Phase 4.5c: bf16 GEMM .o consumed by the placed-IRON v_matmul_seg
    # device (and later k/q_matmul_seg + o_ffn).  Strides match the cached
    # contract's actual access pattern (A=X 2048 elts walked M_BLOCKS=8 K=4;
    # B=W 4096 elts walked N_BLOCKS=16 K=4).  See kernels/build.py for
    # derivation.
    ("bf16_gemm_pythoc_M8_N16_K4_AT_bf16out_s64_512_64_256_64_512.o",
     "_compile_bf16_gemm_rms_gemms_rope"),
    # Phase 4.6d: bf16 GEMM .o consumed by the placed-IRON og_matmul_seg
    # device (O-projection of o_ffn).  N_BLOCKS halves from 16 to 8 because
    # the og C buffer is `1x1x8x8x8x8` (vs v_matmul's `1x1x16x8x8x8`);
    # strides remain identical.  See kernels/build.py for derivation.
    ("bf16_gemm_pythoc_M8_N8_K4_AT_bf16out_s64_512_64_256_64_512.o",
     "_compile_bf16_gemm_og_o_ffn"),
    # Phase 6 (Stage 2): packed-uint4 AWQ kernels.  Each .py kernel uses
    # the scalar per-nibble decode path; vectorization is deferred.
    ("awq_mv_pythoc.o", "compile_awq_mv"),
    ("awq_mv_pythoc.ll", "compile_awq_mv_inline"),        # inlined (alwaysinline)
    ("awq_mv_k8192_pythoc.o", "compile_awq_mv_k8192"),
    ("awq_mv_k8192_pythoc.ll", "compile_awq_mv_k8192_inline"),  # inlined
    ("awq_gemv_k2048_m32_g128_vecdeq_pythoc.o",
     "compile_awq_gemv_k2048_m32_g128_vecdeq"),
    ("awq_gemv_k8192_m8_g128_vecdeq_pythoc.o",
     "compile_awq_gemv_k8192_m8_g128_vecdeq"),
]


def _stage_required_objs():
    """Copy required `.o` files from reference_o/ + build PythoC kernels into CWD.

    KernelCache stages link_with files from CWD into aiecc's tmpdir, so we
    must drop them into CWD before the first `compile_and_cache` call.
    """
    import shutil
    cwd = Path.cwd()
    missing = []
    for name in _REQUIRED_OBJS:
        src = _BUILD_DIR / name
        dst = cwd / name
        if not src.exists():
            missing.append(name)
            continue
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
    if missing:
        raise FileNotFoundError(
            f"Reference `.o` files missing from {_BUILD_DIR}: {missing}. "
            "Seed reference_o/ from the AIR worktree's build_peano."
        )

    # Build PythoC-defined kernels (Phase 2+ replacements).
    import importlib
    kernels_build = importlib.import_module("kernels.build")
    for obj_name, builder_name in _PYTHOC_KERNELS:
        if (cwd / obj_name).exists():
            continue
        builder = getattr(kernels_build, builder_name)
        out = builder(output_dir=str(cwd), verbose=False)
        # compile_pythoc_source returns the basename in output_dir; sanity-check.
        produced = (cwd / Path(out).name)
        if not produced.exists() or produced.name != obj_name:
            raise RuntimeError(
                f"PythoC kernel {builder_name} produced {out!r} but expected {obj_name}"
            )


def compile_all_external_kernels(head_dim=64):
    """Stage Phase-1 reference `.o` files into CWD.

    Called by orchestration before the first `compile_and_cache`. In the
    AIR tree this function compiled each `.cc` with Peano; here we just
    stage prebuilt artifacts.
    """
    del head_dim  # accepted for API compat with the AIR-tree version
    _stage_required_objs()


# ---------------------------------------------------------------------------
# Packed-uint4 AWQ kernels (Stage 2 PythoC ports).
#
# The fused-decode kernels (awq_mv_pythoc.o, awq_mv_k8192_pythoc.o) are
# registered in `_PYTHOC_KERNELS` above and compiled by `_stage_required_objs`
# without a wrapper here.  The standalone GEMV is dim-specialized at runtime:
# `compile_awq_gemv(k, m, group_size, variant)` dispatches to the matching
# `compile_awq_gemv_k{K}_m{M}_g{G}_{variant}` helper in kernels/build.py.
# ---------------------------------------------------------------------------


_AWQ_GEMV_VARIANTS = {"vecdeq"}


def _validate_awq_gemv_variant(variant):
    variant = str(variant)
    if variant not in _AWQ_GEMV_VARIANTS:
        raise ValueError(
            f"Unsupported AWQ GEMV variant {variant!r}; expected one of {sorted(_AWQ_GEMV_VARIANTS)}"
        )
    return variant


def awq_gemv_kernel_name(k, m, group_size, *, variant="vecdeq"):
    """Return the dimension-specialized packed-AWQ GEMV kernel name (no .o).

    Used as both the cache entry name (``<name>.npu.air.mlir`` /
    ``<name>.elf``) and as the prefix for the corresponding object file
    (``<name>_pythoc.o`` -- see ``awq_gemv_object_name``).
    """
    k = int(k); m = int(m); group_size = int(group_size)
    if k <= 0 or m <= 0 or group_size <= 0:
        raise ValueError(f"AWQ GEMV dimensions must be positive: k={k}, m={m}, g={group_size}")
    if k % 2 != 0:
        raise ValueError(f"AWQ GEMV K must be even for uint4 packing, got {k}")
    if k % group_size != 0:
        raise ValueError(f"AWQ GEMV K={k} must be divisible by group_size={group_size}")
    variant = _validate_awq_gemv_variant(variant)
    return f"awq_gemv_k{k}_m{m}_g{group_size}_{variant}"


def awq_gemv_object_name(k, m, group_size, *, variant="vecdeq"):
    """Return the dimension-specialized packed-AWQ GEMV object filename.

    The ``_pythoc`` suffix mirrors the other PythoC-built ``.o`` outputs
    (mv_pythoc.o, attn_pythoc.o, rope_pythoc.o, ...). Stage 2 ports only
    the ``vecdeq`` variant; scalar can be added later if a smoke test
    requires it.
    """
    return f"{awq_gemv_kernel_name(k, m, group_size, variant=variant)}_pythoc.o"


def compile_awq_gemv(k, m, group_size, *, variant="vecdeq", force=False):
    """Compile the dim-specialized packed-uint4 AWQ GEMV PythoC kernel.

    Dispatches to ``kernels.build.compile_awq_gemv_k{K}_m{M}_g{G}_{variant}``
    for the requested shape.  Each shape has its own .py source (one ELF
    per shape) because the standalone GEMV bakes K/M/GROUP_SIZE as
    Python-source constants -- see the kernels/awq_gemv_*_vecdeq.py files
    and the Stage-2 plan rationale.

    Raises ``NotImplementedError`` for shapes that haven't been ported yet.
    """
    del force  # the PythoC compile path is idempotent; CWD is the cache
    k = int(k); m = int(m); group_size = int(group_size)
    variant = _validate_awq_gemv_variant(variant)
    output = awq_gemv_object_name(k, m, group_size, variant=variant)
    helper_name = f"compile_awq_gemv_k{k}_m{m}_g{group_size}_{variant}"

    import importlib
    kb = importlib.import_module("kernels.build")
    if not hasattr(kb, helper_name):
        raise NotImplementedError(
            f"PythoC AWQ GEMV shape (K={k}, M={m}, group_size={group_size}, "
            f"variant={variant!r}) not implemented. Add "
            f"kernels/awq_gemv_k{k}_m{m}_g{group_size}_{variant}.py and "
            f"a matching {helper_name}(output_dir, verbose) helper in "
            f"kernels/build.py."
        )
    helper = getattr(kb, helper_name)
    helper(output_dir=os.getcwd(), verbose=False)
    return output


def compile_silu_and_mul(): _stage_required_objs()
def compile_rope(): _stage_required_objs()
def compile_attn_npu2(head_dim=64): del head_dim; _stage_required_objs()
def compile_attn_decode_npu2(head_dim=64): del head_dim; _stage_required_objs()
def compile_mv(tile_m=8): del tile_m; _stage_required_objs()
def compile_mv_k8192(): _stage_required_objs()
