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
    from pythoc.aie import (
        I1024_I1024_ACC2048_bf_mac_conf,
        loop_range,
        prepare_for_pipelining,
        reduce_add_reassoc,
    )
    with tempfile.TemporaryDirectory(prefix="mv_k8192_pythoc_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("matvec_k8192.py"),
            function_name="dg_matvec_vectorized_bf16_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals={
                "I1024_I1024_ACC2048_bf_mac_conf": I1024_I1024_ACC2048_bf_mac_conf,
                "loop_range": loop_range,
                "prepare_for_pipelining": prepare_for_pipelining,
                "reduce_add_reassoc": reduce_add_reassoc,
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
    from pythoc.aie import (
        I1024_I1024_ACC2048_bf_mac_conf,
        loop_range,
        prepare_for_pipelining,
        reduce_add_reassoc,
    )

    with tempfile.TemporaryDirectory(prefix="mv_pythoc_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("matvec.py"),
            function_name="matvec_vectorized_bf16_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals={
                "I1024_I1024_ACC2048_bf_mac_conf": I1024_I1024_ACC2048_bf_mac_conf,
                "loop_range": loop_range,
                "prepare_for_pipelining": prepare_for_pipelining,
                "reduce_add_reassoc": reduce_add_reassoc,
            },
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "mv_pythoc.o"
        shutil.copy2(produced, dst)
        return dst


def compile_bf16_gemm(
    M_BLOCKS: int,
    N_BLOCKS: int,
    K_MICRO: int,
    *,
    output_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    A_layout_transposed: bool = False,
    c_dtype: str = "f32",
    A_M_STRIDE: Optional[int] = None,
    A_K_STRIDE: Optional[int] = None,
    B_K_STRIDE: Optional[int] = None,
    B_N_STRIDE: Optional[int] = None,
    C_M_STRIDE: Optional[int] = None,
    C_N_STRIDE: Optional[int] = None,
) -> Path:
    """Compile kernels/bf16_gemm.py -> bf16_gemm_pythoc[_<tag>].o for one tile shape.

    Parameters
    ----------
    M_BLOCKS, N_BLOCKS, K_MICRO : int
        Per-core tile dimensions in 8x8 micro-blocks. Output tile is
        (M_BLOCKS*8) x (N_BLOCKS*8); reduction depth is (K_MICRO*8).
        M_BLOCKS and N_BLOCKS must each be even (2x2 register blocking).
    A_layout_transposed : bool, default False
        Use False for `A : [M_BLOCKS, K_MICRO, 8, 8]` (row-major A blocks).
        Use True  for `A : [K_MICRO, M_BLOCKS, 8, 8]` (col-major A blocks --
        matches the AIR-tree prefill `dims_to_stream` retile for K/V GEMMs).
        Internally swaps A_M_STRIDE / A_K_STRIDE.
    c_dtype : "f32" | "bf16", default "f32"
        Output buffer dtype:
          - "f32": uses `bf16_gemm_kernel(a: bf16, b: bf16, c: f32)`. C buffer is
            held as f32 in L1; no boundary truncation. Matches the standalone
            bf16_gemm_multi_core kernel.
          - "bf16": uses `bf16_gemm_kernel_bf16out(a: bf16, b: bf16, c: bf16)`.
            C buffer is bf16 in L1 with explicit extf-on-load / truncf-on-store
            around an f32 register-resident accumulator. Matches the cached
            prefill MLIR's L1 layout. Output name gets a `_bf16out` suffix
            unless overridden.
    output_name : str, optional
        Filename (without `.o`). Default: `bf16_gemm_pythoc_M{M_BLOCKS}_N{N_BLOCKS}_K{K_MICRO}`
        with `_AT` if transposed and `_bf16out` if c_dtype=="bf16".

    Returns
    -------
    Path to the compiled `.o` in `output_dir` (or CWD).
    """
    if M_BLOCKS % 2 != 0:
        raise ValueError(f"M_BLOCKS must be even (2x2 reg blocking); got {M_BLOCKS}")
    if N_BLOCKS % 2 != 0:
        raise ValueError(f"N_BLOCKS must be even (2x2 reg blocking); got {N_BLOCKS}")
    if c_dtype not in ("f32", "bf16"):
        raise ValueError(f"c_dtype must be 'f32' or 'bf16'; got {c_dtype!r}")

    import shutil, tempfile
    from pythoc.aie import (
        BFP576_BFP576_ACC2048_mac_conf,
        concat,
        set_ctrl_reg,
        v32accfloat_to_v32bf16,
        v32bf16_to_v32accfloat,
        v64accfloat_to_v64bfp16ebs8,
        vector_cast,
        vector_extract,
        vshuffle,
    )

    # Stride scalars for the [M_BLOCKS, K_MICRO, 8, 8] reference layout. Each
    # 8x8 micro-tile is 64 elems; strides count elements (not bytes).
    #
    # Each stride can be overridden via an explicit argument; the defaults
    # below assume the kernel's M_BLOCKS / N_BLOCKS / K_MICRO loop bounds
    # match the L1 buffer's outer-dim layout. When the kernel iterates beyond
    # the buffer's own M/N/K extent (e.g. accumulating into a buf_C that's
    # larger than one (M_BLOCKS, N_BLOCKS) tile), the strides must reference
    # the BUFFER's outer dims, not the loop bounds -- pass overrides then.
    if A_layout_transposed:
        # A laid out as [K_MICRO, M_BLOCKS, 8, 8].
        A_M_STRIDE_def = 64
        A_K_STRIDE_def = M_BLOCKS * 64
    else:
        A_M_STRIDE_def = K_MICRO * 64
        A_K_STRIDE_def = 64
    A_M_STRIDE = A_M_STRIDE if A_M_STRIDE is not None else A_M_STRIDE_def
    A_K_STRIDE = A_K_STRIDE if A_K_STRIDE is not None else A_K_STRIDE_def
    B_K_STRIDE = B_K_STRIDE if B_K_STRIDE is not None else N_BLOCKS * 64
    B_N_STRIDE = B_N_STRIDE if B_N_STRIDE is not None else 64
    C_M_STRIDE = C_M_STRIDE if C_M_STRIDE is not None else N_BLOCKS * 64
    C_N_STRIDE = C_N_STRIDE if C_N_STRIDE is not None else 64

    extras = {
        "BFP576_BFP576_ACC2048_mac_conf": BFP576_BFP576_ACC2048_mac_conf,
        "concat": concat,
        "set_ctrl_reg": set_ctrl_reg,
        "v32accfloat_to_v32bf16": v32accfloat_to_v32bf16,
        "v32bf16_to_v32accfloat": v32bf16_to_v32accfloat,
        "v64accfloat_to_v64bfp16ebs8": v64accfloat_to_v64bfp16ebs8,
        "vector_cast": vector_cast,
        "vector_extract": vector_extract,
        "vshuffle": vshuffle,
        # MAC_CONF=780 matches the bf16_gemm_multi_core reference; this is
        # the canonical conf for BFP16 emulation on AIE2P.
        "MAC_CONF": 780,
        "M_BLOCKS_CONST": int(M_BLOCKS),
        "N_BLOCKS_CONST": int(N_BLOCKS),
        "K_MICRO_CONST": int(K_MICRO),
        "A_M_STRIDE_CONST": int(A_M_STRIDE),
        "A_K_STRIDE_CONST": int(A_K_STRIDE),
        "B_K_STRIDE_CONST": int(B_K_STRIDE),
        "B_N_STRIDE_CONST": int(B_N_STRIDE),
        "C_M_STRIDE_CONST": int(C_M_STRIDE),
        "C_N_STRIDE_CONST": int(C_N_STRIDE),
    }

    # llvm-aie's software pipeliner can produce an invalid BM spill on some
    # tile shapes (seen on 128x32x64 in bf16_gemm_multi_core). Match the
    # reference example and disable the pipeliner for this .o.
    import os
    user_flags = os.environ.get("PYTHOC_LLC_FLAGS", "")
    new_flags = (user_flags + " -enable-pipeliner=false").strip()
    os.environ["PYTHOC_LLC_FLAGS"] = new_flags

    tag = f"M{M_BLOCKS}_N{N_BLOCKS}_K{K_MICRO}"
    if A_layout_transposed:
        tag += "_AT"
    if c_dtype == "bf16":
        tag += "_bf16out"
        function_name = "bf16_gemm_kernel_bf16out"
    else:
        function_name = "bf16_gemm_kernel"
    # If any stride differs from the AT/non-AT default, mangle the tag so
    # the output name disambiguates from the default-stride variant.
    if (A_M_STRIDE != A_M_STRIDE_def or A_K_STRIDE != A_K_STRIDE_def
        or B_K_STRIDE != N_BLOCKS * 64 or B_N_STRIDE != 64
        or C_M_STRIDE != N_BLOCKS * 64 or C_N_STRIDE != 64):
        tag += f"_s{A_M_STRIDE}_{A_K_STRIDE}_{B_K_STRIDE}_{B_N_STRIDE}_{C_M_STRIDE}_{C_N_STRIDE}"
    name = output_name or f"bf16_gemm_pythoc_{tag}"

    try:
        with tempfile.TemporaryDirectory(prefix="bf16_gemm_pythoc_") as tmp:
            produced = compile_pythoc_source(
                source_code=_read("bf16_gemm.py"),
                function_name=function_name,
                target_arch="aie2p",
                output_dir=tmp,
                verbose=verbose,
                extra_globals=extras,
            )
            dst_dir = Path(output_dir) if output_dir else Path.cwd()
            dst = dst_dir / f"{name}.o"
            shutil.copy2(produced, dst)
            return dst
    finally:
        if user_flags:
            os.environ["PYTHOC_LLC_FLAGS"] = user_flags
        else:
            os.environ.pop("PYTHOC_LLC_FLAGS", None)


def _compile_bf16_gemm_rms_gemms_rope(
    output_dir: Optional[str] = None, verbose: bool = False
) -> Path:
    """Compile the bf16 GEMM .o consumed by the placed-IRON v_matmul_seg device.

    Strides match the CACHED AIR-emitted contract chain's access pattern in
    `reference_mlir/rms_gemms_rope.npu.air.mlir` (verified empirically against
    the cached output via tests/test_v_matmul_oracle.py):

      * a_buf = compute buf_A = X data, shape 1x1x4x8x8x8 (2048 elts).
        Cached reads at offset (arg_m * 64 + arg_k * 512), so kernel uses
        A_M=64, A_K=512, with M_BLOCKS=8 (= cached's outer arg1, range 8 with
        2x2 reg blocking).
      * b_buf = compute buf_B = W data, shape 1x1x16x4x8x8 (4096 elts).
        Cached reads at offset (arg_n * 256 + arg_k * 64), so kernel uses
        B_N=256, B_K=64, with N_BLOCKS=16 (= cached's outer arg2, range 16).
      * c_buf = buf_C, shape 1x1x16x8x8x8 (8192 elts).  Cached writes at
        offset (arg_n * 512 + arg_m * 64), so C_M=64, C_N=512.

    These strides differ from the M_BLOCKS=16/N_BLOCKS=8 configuration that
    was used in Phase 4.5c through 2026-05-23: that produced runtime garbage
    (corr=0.007 vs cached) because the kernel walked a_buf as if it were
    4096 elts (M=16 outer) when the actual buffer is 2048 elts (M=8 outer),
    and walked b_buf with K and N strides swapped.  Beads PythoC-8ns.13.

    Symbol inside is ``bf16_gemm_kernel_bf16out`` (bf16-out variant: C is held
    as bf16 in L1, extf-on-load / truncf-on-store around an f32 register
    accumulator).
    """
    return compile_bf16_gemm(
        M_BLOCKS=8,
        N_BLOCKS=16,
        K_MICRO=4,
        A_layout_transposed=True,
        c_dtype="bf16",
        A_M_STRIDE=64,
        A_K_STRIDE=512,
        B_K_STRIDE=64,
        B_N_STRIDE=256,
        C_M_STRIDE=64,
        C_N_STRIDE=512,
        output_dir=output_dir,
        verbose=verbose,
    )


def _compile_bf16_gemm_og_o_ffn(
    output_dir: Optional[str] = None, verbose: bool = False
) -> Path:
    """Compile the bf16 GEMM .o consumed by the placed-IRON og_matmul_seg device.

    `og_matmul_seg` is the O-projection GEMM of the o_ffn prefill block.
    Its per-core C tile is **64x64** (half the per-core C of v/k/q matmul),
    matching the cached AIR IR in
    ``reference_mlir/o_ffn.npu.air.mlir`` (device begins at line 27548).

    Per-core L1 buffer shapes derived from the cached compute-tile mem
    block (e.g. ``aie.mem(%tile_0_2)`` at line 34031) and per-tile buffer
    declarations:

      * buf_C : memref<1x1x8x8x8x8 xbf16, 2>     (4096 elts; was 8192 for v)
      * buf_A : memref<1x1x4x8x8x8 xbf16, 2>     (2048 elts; same as v)
      * buf_B : memref<1x1x8x4x8x8 xbf16, 2>     (2048 elts; was 4096 for v)

    The cached AIR-emitted core body walks these buffers as a nested
    8 outer (arg0) x 4 inner step=2 (arg1) loop (= 32 ping/pong pairs =
    64 inline contract iterations).  Each contract iteration reads:

      * A: 64-elt slices at offsets ``arg2*64 + arg3*64`` walking by
        +512 every K step (4 micro-K iters per call) -- so kernel
        ``A_M=64``, ``A_K=512`` over an outer M loop of size **8**.
      * B: 64-elt slices at offsets ``arg3*256 + arg2*64`` walking by
        +64 every K step (4 micro-K iters per call) -- so kernel
        ``B_N=256``, ``B_K=64`` over an outer N loop of size **8**.
      * C: 64-elt micro-tiles at offsets ``arg2*512 + arg3*64``, so
        kernel ``C_M=64``, ``C_N=512`` (same as v_matmul's strides on
        a `1x1x8x8x8x8` buf).

    M_BLOCKS=N_BLOCKS=8 because the C buffer's outer (M,N) extent is
    (8,8) -- the kernel must walk that exact 8x8 grid of 8x8 micro-tiles
    per call, accumulating K_MICRO=4 micro-K slices into each.

    These strides differ from `_compile_bf16_gemm_rms_gemms_rope` (which
    bakes the v/k/q matmul's M_BLOCKS=8, N_BLOCKS=16 contract loop) only
    in N_BLOCKS (8 instead of 16): the per-call K depth is identical
    (4 micro-K slices) and the per-element strides match (A: 64/512,
    B: 64/256, C: 64/512) because both v_matmul and og_matmul write into
    a `1x1xMxNx8x8` buf where N==C's inner dim equals 8 (not 16) on og.

    Output .o symbol is ``bf16_gemm_kernel_bf16out`` (bf16-out variant:
    C is held as bf16 in L1, extf-on-load / truncf-on-store around an
    f32 register accumulator).  Compiled name:
    ``bf16_gemm_pythoc_M8_N8_K4_AT_bf16out_s64_512_64_256_64_512.o``.
    """
    return compile_bf16_gemm(
        M_BLOCKS=8,
        N_BLOCKS=8,
        K_MICRO=4,
        A_layout_transposed=True,
        c_dtype="bf16",
        A_M_STRIDE=64,
        A_K_STRIDE=512,
        B_K_STRIDE=64,
        B_N_STRIDE=256,
        C_M_STRIDE=64,
        C_N_STRIDE=512,
        output_dir=output_dir,
        verbose=verbose,
    )


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


# ---------------------------------------------------------------------------
# Packed-uint4 AWQ kernel builders (Stage 2).
#
# Each helper mirrors compile_matvec / compile_matvec_k8192 above; the
# kernels themselves use the SCALAR per-nibble decode path (the vectorized
# uint4 -> bf16 chain requires a PythoC bitcast op that doesn't exist yet,
# per the Stage-0 receipts).  ``set_ctrl_reg(1, 12)`` is the only lazy
# intrinsic needed inside the AWQ kernels (no MAC intrinsics in the scalar
# fallback path, so conf=60 doesn't apply here).
# ---------------------------------------------------------------------------


def compile_awq_mv(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/awq_mv.py -> awq_mv_pythoc.o.

    Fused-decode AWQ matvec with runtime m/k/row_offset and combined-row
    ABI.  Source has two @aie_kernel functions; the helper
    (``awq_linalg_fill_bf16``) is defined FIRST so compile_pythoc_source
    picks it up via helper_nodes while compiling the named entry
    ``awq_matvec_vectorized_u4_bf16``.  Both symbols land in one .o.

    Module-level constants (``GROUP_SIZE``, ``DIM_M_OUTPUT``) are
    referenced inside the kernel bodies, so they must be seeded via
    ``extra_globals`` -- the PythoC AST walker only auto-imports its
    hard-coded list.
    """
    import shutil, tempfile
    import ml_dtypes
    from pythoc.aie import (
        set_ctrl_reg,
        I512_I512_ACC1024_bf_msc_conf,
        I1024_I1024_ACC2048_bf_mac_conf,
        I1024_I1024_ACC2048_bf_msc_conf,
        v32accfloat_to_v32bf16,
        reduce_add_reassoc,
        unpack_unsigned,
        unpack_I512_I8_I4,
        vector_add,
        vector_mul,
        vector_cast,
        vector_extract,
        broadcast,
        concat,
        loop_range,
        prepare_for_pipelining,
    )
    extras = {
        "set_ctrl_reg": set_ctrl_reg,
        "I512_I512_ACC1024_bf_msc_conf": I512_I512_ACC1024_bf_msc_conf,
        "I1024_I1024_ACC2048_bf_mac_conf": I1024_I1024_ACC2048_bf_mac_conf,
        "I1024_I1024_ACC2048_bf_msc_conf": I1024_I1024_ACC2048_bf_msc_conf,
        "v32accfloat_to_v32bf16": v32accfloat_to_v32bf16,
        "reduce_add_reassoc": reduce_add_reassoc,
        "unpack_unsigned": unpack_unsigned,
        "unpack_I512_I8_I4": unpack_I512_I8_I4,
        "vector_add": vector_add,
        "vector_mul": vector_mul,
        "vector_cast": vector_cast,
        "vector_extract": vector_extract,
        "broadcast": broadcast,
        "concat": concat,
        "loop_range": loop_range,
        "prepare_for_pipelining": prepare_for_pipelining,
        "GROUP_SIZE": 128,
        "DIM_M_OUTPUT": 8,
        # Fix2Float magic constants (see kernels/awq_mv.py docstring).
        "MAGIC_L_I32": 0x4b010000,
        "MAGIC_L_BF": ml_dtypes.bfloat16(8454144.0),
        "CONF_BF16_MAC": 60,
    }
    with tempfile.TemporaryDirectory(prefix="awq_mv_pythoc_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("awq_mv.py"),
            function_name="awq_matvec_vectorized_u4_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals=extras,
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "awq_mv_pythoc.o"
        shutil.copy2(produced, dst)
        return dst


def compile_awq_mv_k8192(output_dir: Optional[str] = None, verbose: bool = False) -> Path:
    """Compile kernels/awq_mv_k8192.py -> awq_mv_k8192_pythoc.o.

    Same shape as compile_awq_mv but with the FFN down-projection symbol
    names (``dg_awq_matvec_vectorized_u4_bf16``,
    ``dg_awq_linalg_fill_bf16``) and DIM_M_OUTPUT=2.
    """
    import shutil, tempfile
    import ml_dtypes
    from pythoc.aie import (
        set_ctrl_reg,
        I512_I512_ACC1024_bf_msc_conf,
        I1024_I1024_ACC2048_bf_mac_conf,
        I1024_I1024_ACC2048_bf_msc_conf,
        v32accfloat_to_v32bf16,
        reduce_add_reassoc,
        unpack_unsigned,
        unpack_I512_I8_I4,
        vector_add,
        vector_mul,
        vector_cast,
        vector_extract,
        broadcast,
        concat,
        loop_range,
        prepare_for_pipelining,
    )
    extras = {
        "set_ctrl_reg": set_ctrl_reg,
        "I512_I512_ACC1024_bf_msc_conf": I512_I512_ACC1024_bf_msc_conf,
        "I1024_I1024_ACC2048_bf_mac_conf": I1024_I1024_ACC2048_bf_mac_conf,
        "I1024_I1024_ACC2048_bf_msc_conf": I1024_I1024_ACC2048_bf_msc_conf,
        "v32accfloat_to_v32bf16": v32accfloat_to_v32bf16,
        "reduce_add_reassoc": reduce_add_reassoc,
        "unpack_unsigned": unpack_unsigned,
        "unpack_I512_I8_I4": unpack_I512_I8_I4,
        "vector_add": vector_add,
        "vector_mul": vector_mul,
        "vector_cast": vector_cast,
        "vector_extract": vector_extract,
        "broadcast": broadcast,
        "concat": concat,
        "loop_range": loop_range,
        "prepare_for_pipelining": prepare_for_pipelining,
        "GROUP_SIZE": 128,
        "DIM_M_OUTPUT": 2,
        "MAGIC_L_I32": 0x4b010000,
        "MAGIC_L_BF": ml_dtypes.bfloat16(8454144.0),
        "CONF_BF16_MAC": 60,
    }
    with tempfile.TemporaryDirectory(prefix="awq_mv_k8192_pythoc_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("awq_mv_k8192.py"),
            function_name="dg_awq_matvec_vectorized_u4_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals=extras,
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "awq_mv_k8192_pythoc.o"
        shutil.copy2(produced, dst)
        return dst


def compile_awq_gemv_k2048_m32_g128_vecdeq(
    output_dir: Optional[str] = None, verbose: bool = False
) -> Path:
    """Compile kernels/awq_gemv_k2048_m32_g128_vecdeq.py ->
    awq_gemv_k2048_m32_g128_vecdeq_pythoc.o.

    Standalone dim-specialized AWQ GEMV (K=2048, M=32, GS=128).  The
    symbol inside is ``awq_gemv_u4_bf16`` -- this is the same name used
    by the K=8192/M=8 variant (different ELF, can't co-link).  Each
    cached MLIR points at its own ``.o`` via ``link_with``.
    """
    import shutil, tempfile
    from pythoc.aie import set_ctrl_reg
    extras = {
        "set_ctrl_reg": set_ctrl_reg,
        "K": 2048,
        "M": 32,
        "GROUP_SIZE": 128,
    }
    with tempfile.TemporaryDirectory(prefix="awq_gemv_k2048_m32_g128_vecdeq_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("awq_gemv_k2048_m32_g128_vecdeq.py"),
            function_name="awq_gemv_u4_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals=extras,
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "awq_gemv_k2048_m32_g128_vecdeq_pythoc.o"
        shutil.copy2(produced, dst)
        return dst


def compile_awq_gemv_k8192_m8_g128_vecdeq(
    output_dir: Optional[str] = None, verbose: bool = False
) -> Path:
    """Compile kernels/awq_gemv_k8192_m8_g128_vecdeq.py ->
    awq_gemv_k8192_m8_g128_vecdeq_pythoc.o.

    Standalone dim-specialized AWQ GEMV (K=8192, M=8, GS=128).
    """
    import shutil, tempfile
    from pythoc.aie import set_ctrl_reg
    extras = {
        "set_ctrl_reg": set_ctrl_reg,
        "K": 8192,
        "M": 8,
        "GROUP_SIZE": 128,
    }
    with tempfile.TemporaryDirectory(prefix="awq_gemv_k8192_m8_g128_vecdeq_") as tmp:
        produced = compile_pythoc_source(
            source_code=_read("awq_gemv_k8192_m8_g128_vecdeq.py"),
            function_name="awq_gemv_u4_bf16",
            target_arch="aie2p",
            output_dir=tmp,
            verbose=verbose,
            extra_globals=extras,
        )
        dst_dir = Path(output_dir) if output_dir else Path.cwd()
        dst = dst_dir / "awq_gemv_k8192_m8_g128_vecdeq_pythoc.o"
        shutil.copy2(produced, dst)
        return dst
