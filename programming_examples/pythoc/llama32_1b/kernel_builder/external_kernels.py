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
    ("mv_k8192_pythoc.o", "compile_matvec_k8192"),
    ("attn_pythoc.o", "compile_attn"),
    # Phase 4.5c: bf16 GEMM .o consumed by the placed-IRON v_matmul_seg
    # device (and later k/q_matmul_seg + o_ffn).  Strides match the cached
    # contract's actual access pattern (A=X 2048 elts walked M_BLOCKS=8 K=4;
    # B=W 4096 elts walked N_BLOCKS=16 K=4).  See kernels/build.py for
    # derivation.
    ("bf16_gemm_pythoc_M8_N16_K4_AT_bf16out_s64_512_64_256_64_512.o",
     "_compile_bf16_gemm_rms_gemms_rope"),
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
# AWQ helpers (Phase 6 - deferred). Kept here so the AWQ runtime imports
# don't break; raise on first use.
# ---------------------------------------------------------------------------


def awq_gemv_object_name(k, m, group_size, *, variant="scalar"):
    """Return the dimension-specialized packed-AWQ GEMV object filename."""
    k = int(k); m = int(m); group_size = int(group_size)
    if k <= 0 or m <= 0 or group_size <= 0:
        raise ValueError(f"AWQ GEMV dimensions must be positive: k={k}, m={m}, g={group_size}")
    if k % 2 != 0:
        raise ValueError(f"AWQ GEMV K must be even for uint4 packing, got {k}")
    if k % group_size != 0:
        raise ValueError(f"AWQ GEMV K={k} must be divisible by group_size={group_size}")
    return f"awq_gemv_k{k}_m{m}_g{group_size}_{variant}.o"


def compile_awq_gemv(k, m, group_size, *, variant="scalar", force=False):
    raise NotImplementedError(
        "AWQ uint4 GEMV is deferred to Phase 6 of the PythoC port. "
        "Drop the AWQ runtime path (--awq-decode-experimental) until then."
    )


def compile_silu_and_mul(): _stage_required_objs()
def compile_rope(): _stage_required_objs()
def compile_attn_npu2(head_dim=64): del head_dim; _stage_required_objs()
def compile_attn_decode_npu2(head_dim=64): del head_dim; _stage_required_objs()
def compile_mv(tile_m=8): del tile_m; _stage_required_objs()
def compile_mv_k8192(): _stage_required_objs()
def compile_awq_mv(group_size=128, tile_m=8): raise NotImplementedError("AWQ deferred to Phase 6")
def compile_awq_mv_k8192(group_size=128, tile_m=2): raise NotImplementedError("AWQ deferred to Phase 6")
