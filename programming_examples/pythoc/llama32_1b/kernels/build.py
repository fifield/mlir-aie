# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build helpers for the llama32_1b PythoC kernels.

`compile_pythoc_kernel` from aie.iron.pythoc only seeds user_globals from a
hard-coded import list -- lazy intrinsics like `invsqrt` are not visible to
the AST visitor unless passed explicitly via `extra_globals`. Each builder
here wires that up and writes the resulting `.o` into CWD so the cache
stages it for aiecc link.
"""

from pathlib import Path
from typing import Optional

from aie.iron.pythoc.compiler import compile_pythoc_source

_KERNELS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_KERNELS_DIR / name).read_text()


def compile_rms_norm(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/rms_norm.py -> rms_norm_2048_bf16.o for aiecc linking."""
    from pythoc.aie import invsqrt
    return compile_pythoc_source(
        source_code=_read("rms_norm.py"),
        function_name="rms_norm_2048_bf16",
        target_arch="aie2p",
        output_dir=output_dir,
        verbose=verbose,
        extra_globals={"invsqrt": invsqrt},
    )


def compile_silu_and_mul(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/silu_and_mul.py -> silu_and_mul_bf16.o for aiecc linking."""
    from pythoc.aie import getTanhBf16
    return compile_pythoc_source(
        source_code=_read("silu_and_mul.py"),
        function_name="silu_and_mul_bf16",
        target_arch="aie2p",
        output_dir=output_dir,
        verbose=verbose,
        extra_globals={"getTanhBf16": getTanhBf16},
    )


def compile_attn(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/attn.py -> attn_pythoc.o.

    Source carries all 19 @aie_kernel functions for flash attention.
    The named entry is `fused_softmax` (defined last); every earlier
    function becomes a "helper" and is exported in the same .o thanks
    to PythoC's AST walker breaking on the named match. `fused_softmax`
    itself calls back into six of its helper symbols, which works
    because compile_pythoc_source's helper loop registers each compiled
    helper in user_globals before compiling the next function.

    All lazy AIE2P intrinsics referenced inside the kernels must be
    injected via `extra_globals` -- compile_pythoc_source only seeds
    its hard-coded import list and the AST visitor's NameError trap
    fires on anything else.
    """
    import shutil, tempfile
    from pythoc.aie import (
        ACC2048_accfloat_add_conf,
        BFP576_BFP576_ACC2048_mac_conf,
        I1024_I1024_ACC2048_bf_mul_conf,
        I512_I512_ACC1024_bf_mac_conf,
        I512_I512_ACC1024_bf_mul_conf,
        I512_I512_ACC1024_bf_negmul_conf,
        acc_extract,
        acc_grow,
        concat,
        exp2,
        extract_elem,
        getExpBf16,
        insert_elem,
        reduce_add,
        set_ctrl_reg,
        v32accfloat_to_v32bf16,
        v32bf16_to_v32accfloat,
        v64accfloat_to_v64bfp16ebs8,
        vector_blend,
        vector_cast,
        vector_extract,
        vector_grow,
        vector_insert,
        vector_sub,
        vmax_ltbf16,
        vshuffle,
    )
    extras = {
        "ACC2048_accfloat_add_conf": ACC2048_accfloat_add_conf,
        "BFP576_BFP576_ACC2048_mac_conf": BFP576_BFP576_ACC2048_mac_conf,
        "I1024_I1024_ACC2048_bf_mul_conf": I1024_I1024_ACC2048_bf_mul_conf,
        "I512_I512_ACC1024_bf_mac_conf": I512_I512_ACC1024_bf_mac_conf,
        "I512_I512_ACC1024_bf_mul_conf": I512_I512_ACC1024_bf_mul_conf,
        "I512_I512_ACC1024_bf_negmul_conf": I512_I512_ACC1024_bf_negmul_conf,
        "acc_extract": acc_extract,
        "acc_grow": acc_grow,
        "concat": concat,
        "exp2": exp2,
        "extract_elem": extract_elem,
        "getExpBf16": getExpBf16,
        "insert_elem": insert_elem,
        "reduce_add": reduce_add,
        "set_ctrl_reg": set_ctrl_reg,
        "v32accfloat_to_v32bf16": v32accfloat_to_v32bf16,
        "v32bf16_to_v32accfloat": v32bf16_to_v32accfloat,
        "v64accfloat_to_v64bfp16ebs8": v64accfloat_to_v64bfp16ebs8,
        "vector_blend": vector_blend,
        "vector_cast": vector_cast,
        "vector_extract": vector_extract,
        "vector_grow": vector_grow,
        "vector_insert": vector_insert,
        "vector_sub": vector_sub,
        "vmax_ltbf16": vmax_ltbf16,
        "vshuffle": vshuffle,
    }
    # attn_pythoc.o packs all 19 flash-attn helpers into one TU. Without
    # `--function-sections` they all land in a single .text and the
    # AIE2P core ELFs overflow program memory (the AIR-reference
    # attn_npu2.o uses per-function .text.* sections via `aie::reduce_max`
    # tree intrinsics, so the linker can gc-section the unused ones).
    # Persist any user-provided PYTHOC_LLC_FLAGS while we splice ours in.
    import os
    user_flags = os.environ.get("PYTHOC_LLC_FLAGS", "")
    new_flags = (user_flags + " --function-sections").strip()
    os.environ["PYTHOC_LLC_FLAGS"] = new_flags
    try:
        with tempfile.TemporaryDirectory(prefix="attn_pythoc_") as tmp:
            produced = compile_pythoc_source(
                source_code=_read("attn.py"),
                function_name="fused_softmax",
                target_arch="aie2p",
                output_dir=tmp,
                verbose=verbose,
                extra_globals=extras,
            )
            dst_dir = Path(output_dir) if output_dir else Path.cwd()
            dst = dst_dir / "attn_pythoc.o"
            shutil.copy2(produced, dst)
            return dst
    finally:
        if user_flags:
            os.environ["PYTHOC_LLC_FLAGS"] = user_flags
        else:
            os.environ.pop("PYTHOC_LLC_FLAGS", None)


def compile_matvec_k8192(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/matvec_k8192.py -> mv_k8192_pythoc.o.

    Same shape as compile_matvec but with the FFN down-projection symbol
    names (`dg_matvec_vectorized_bf16_bf16`, `dg_linalg_fill_bf16`).
    """
    import shutil, tempfile
    from pythoc.aie import I512_I512_ACC1024_bf_mac_conf, reduce_add
    with tempfile.TemporaryDirectory(prefix="mv_k8192_pythoc_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("matvec_k8192.py"),
            function_name="dg_matvec_vectorized_bf16_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals={
                "I512_I512_ACC1024_bf_mac_conf": I512_I512_ACC1024_bf_mac_conf,
                "reduce_add": reduce_add,
            },
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "mv_k8192_pythoc.o"
        shutil.copy2(produced, dst)
        return dst


def compile_matvec(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/matvec.py -> mv_pythoc.o.

    Source has two @aie_kernel functions; the helper (`linalg_fill_bf16`)
    is defined FIRST so compile_pythoc_source picks it up via helper_nodes
    while compiling `matvec_vectorized_bf16_bf16`. Both symbols land in
    one .o, which is renamed to mv_pythoc.o so the AIR reference at
    reference_o/mv.o is preserved.
    """
    import shutil, tempfile
    from pythoc.aie import I512_I512_ACC1024_bf_mac_conf, reduce_add

    with tempfile.TemporaryDirectory(prefix="mv_pythoc_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("matvec.py"),
            function_name="matvec_vectorized_bf16_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals={
                "I512_I512_ACC1024_bf_mac_conf": I512_I512_ACC1024_bf_mac_conf,
                "reduce_add": reduce_add,
            },
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "mv_pythoc.o"
        shutil.copy2(produced, dst)
        return dst


def compile_rope(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/rope.py -> rope_pythoc.o for aiecc linking.

    PythoC writes `<function_name>.o`; we rename to `rope_pythoc.o` so the
    AIR reference at reference_o/rope.o is preserved during incremental
    development. The MLIR link_with for migrated cores is patched to
    "rope_pythoc.o"; the symbol inside stays `rope` so func.call sites in
    the cached MLIR don't need to change.
    """
    import tempfile, shutil
    with tempfile.TemporaryDirectory(prefix="rope_pythoc_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("rope.py"),
            function_name="rope",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "rope_pythoc.o"
        shutil.copy2(produced, dst)
        return dst
