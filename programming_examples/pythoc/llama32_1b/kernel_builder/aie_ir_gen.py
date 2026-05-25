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

The AIR-tree counterpart of this file runs aircc on multi_launch_builder
modules to harvest post-stitched `npu.air.mlir`; the pythoc tree never
invokes aircc at compile time.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference_mlir"


# ---------------------------------------------------------------------------
# Stage 1 AWQ scaffolding: aircc shell-out for compile-on-demand AIR module
# lowering.  Stage 2 retires the AIR-tree path entirely; this whole block
# goes away then.
# ---------------------------------------------------------------------------


def _resolve_peano_dir() -> str:
    p = os.environ.get("PEANO_INSTALL_DIR", "")
    if p:
        return p
    raise RuntimeError("PEANO_INSTALL_DIR is not set")


_LINK_OBJS = [
    "silu_and_mul.o", "rope.o", "attn.o", "attn_npu2.o",
    "mv.o", "mv_k8192.o", "attn_decode_npu2.o",
    "awq_mv.o", "awq_mv_k8192.o",
]


def lower_air_to_npu_air_mlir(
    air_module_text: str,
    *,
    device: str = "npu2",
    num_cols: int = 8,
    omit_while_true_loop: bool = False,
    omit_pingpong: Optional[str] = None,
    runtime_loop_tiling_sizes=(),
    use_lock_race_condition_fix: bool = False,
    workdir: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Run AIR passes via aircc and return the post-stitched npu.air.mlir text.

    Used by the AIR-tree AWQ builders (`awq_gemv_builder`, `awq_matvec`) that
    we copied in as Stage-1 scaffolding so compile-on-demand smoke-test shapes
    still work. Stage 2 replaces this with PythoC + placed-IRON and removes
    the function.
    """
    aircc_exe = shutil.which("aircc")
    if not aircc_exe:
        raise RuntimeError("aircc not found on PATH")

    work = Path(workdir or tempfile.mkdtemp(prefix="air_lower_"))
    work.mkdir(parents=True, exist_ok=True)

    cwd = Path.cwd()
    staged = set()
    for obj_name in _LINK_OBJS:
        src = cwd / obj_name
        if src.exists():
            shutil.copy2(src, work / obj_name)
            staged.add(src.resolve())
    for src in sorted(cwd.glob("awq_gemv_*.o")):
        if src.resolve() not in staged:
            shutil.copy2(src, work / src.name)

    air_path = work / "air.mlir"
    air_path.write_text(air_module_text)

    cmd = [
        aircc_exe,
        "--device", device,
        "--output-format", "elf",
        "--elf-name", "aie.elf",
        f"--tmpdir={work}",
        f"--peano={_resolve_peano_dir()}",
        "--no-xchesscc",
        "--no-xbridge",
    ]
    if num_cols:
        cmd += [f"--num-cols={num_cols}"]
    if omit_while_true_loop:
        cmd += ["--omit-while-true-loop"]
    if omit_pingpong is not None:
        pp = "all" if omit_pingpong is True else str(omit_pingpong)
        cmd += [f"--omit-ping-pong-transform={pp}"]
    for s in runtime_loop_tiling_sizes:
        cmd += [f"--air-runtime-loop-tiling-sizes={s}"]
    if use_lock_race_condition_fix:
        cmd += ["--use-lock-race-condition-fix"]
    if verbose:
        cmd += ["-v"]
    cmd.append(str(air_path))

    if verbose:
        print(f"  [aircc lowering] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True)
    dt = time.time() - t0

    npu_path = work / "npu.air.mlir"
    if not npu_path.exists():
        msg = proc.stderr or proc.stdout
        raise RuntimeError(
            f"aircc lowering produced no npu.air.mlir in {dt:.1f}s "
            f"(returncode={proc.returncode}):\n{msg}"
        )
    if verbose and proc.returncode != 0:
        print(f"  [aircc] backend step failed but IR was recovered; "
              f"returncode={proc.returncode}")
    return npu_path.read_text()

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


def build_awq_gemv_ir(k, m, group_size, *, variant="scalar", verbose=False):
    """Packed uint4 AWQ GEMV primitive.

    Stage 3 wires this through the placed-IRON builder gate. When
    ``awq_matvec`` is enabled (the default), emit the module directly
    from ``builders/awq_matvec.py``. Otherwise prefer cached MLIR; if
    no cached file exists for the requested shape, fall back to the
    AIR-tree ``awq_gemv_builder.build_awq_gemv_ir`` for compile-on-demand.
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
    if cache_path.exists():
        if verbose:
            print(f"  [aie_ir_gen] Using cached MLIR for {cache_name}")
        return cache_path.read_text()
    # Fallback: AIR-tree builder for compile-on-demand (smoke tests, new shapes).
    try:
        from kernel_builder.awq_gemv_builder import build_awq_gemv_ir as _build
    except ImportError as exc:
        raise FileNotFoundError(
            f"No cached MLIR for {cache_name} at {cache_path}, and "
            f"awq_gemv_builder is not importable: {exc}"
        ) from exc
    if verbose:
        print(f"  [aie_ir_gen] Building AWQ GEMV IR on demand for {cache_name}")
    return str(_build(k=k, m=m, group_size=group_size, variant=variant))
