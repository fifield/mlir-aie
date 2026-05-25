# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage-0 Subtask-B smoke test for uint4 unpack -> bf16 dequant.

This is a compile-only smoke test that exercises the Stage-0 PythoC
frontend extensions:

  * The newly-exported ``u4`` builtin type (Subtask A) parses without
    NameError. ``u4`` is currently used only in the public type-set
    sanity check below -- the kernel itself takes ``u8`` pointers and
    extracts nibbles via scalar bit-ops (matching ``awq_mv.cc:106-111``).
    A vectorized ``uint4 -> bf16`` chain (Subtask B option 1 / option 2)
    requires either a new llvm-aie intrinsic or a multi-step magic-number
    trick (see ``aie_api/detail/aie2p/elementary.hpp:Fix2Float``) and is
    deferred to Stage 2.

  * ``set_ctrl_reg(1, 12)`` at kernel entry -- the established attn.py
    pattern for ``aie::set_rounding(aie::rounding_mode::conv_even)``.
    Register 1 = crRnd; value 12 = rnd_conv_even (see
    ``llvm-aie/clang/lib/Headers/aie2p/aie2p_defines.h:45``).

  * The PythoC compile pipeline produces ``.ll``, ``.opt.ll``, and ``.o``.
    The ``.opt.ll`` is the post-optimization LLVM IR that the smoke test
    prints so a human reviewer can inspect the lowered nibble-extraction
    + bf16 dequant sequence.

Run as a script (not pytest):

    source env.sh
    python3 tools/test_pythoc_u4_unpack.py

Exit status: 0 on compile success; non-zero on any failure. The .opt.ll
path is printed so its content can be hand-inspected.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Subtask-A sanity: u4 must be importable from pythoc.
# ---------------------------------------------------------------------------

from pythoc import u4, u8, bf16  # noqa: E402

assert u4.get_name() == "u4", f"u4.get_name() returned {u4.get_name()!r}"
assert getattr(u4, "_is_integer", False), "u4 should be _is_integer=True"
assert getattr(u4, "_is_signed", True) is False, "u4 should be _is_signed=False"


# ---------------------------------------------------------------------------
# Kernel source: packed-u8 -> bf16 with per-element (nibble - zero) * scale.
#
# Mirrors awq_mv.cc:104-111 (scalar branch). 16 packed bytes = 32 nibbles.
# Low nibble first, high nibble second, per the C++ AWQ packing.
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = """
from pythoc import bf16, i32, u32, u8, ptr, void
from pythoc.aie import aie_vector, store_v, zeros, insert_elem

# Compile-time constants (seeded via extra_globals).
ZERO_BF16: bf16
SCALE_BF16: bf16
NUM_PAIRS: u32  # 16 -> 32 output lanes


def test_u4_unpack_bf16(q: ptr[u8, True], c_out: ptr[bf16, True]) -> void:
    # Round-to-even rounding mode (matches aie::set_rounding(conv_even)).
    set_ctrl_reg(i32(1), i32(12))

    # Scratch bf16 vector that we fill nibble-by-nibble.
    w: aie_vector[bf16, 32] = zeros(bf16, 32)

    i: u32 = u32(0)
    while i < NUM_PAIRS:
        packed: u8 = q[i]
        q_even: u8 = packed & u8(15)         # low  nibble (bits 0..3)
        q_odd: u8 = (packed >> u8(4)) & u8(15)  # high nibble (bits 4..7)

        # Scalar widen to bf16, then AWQ-dequant per element.
        f_even: bf16 = bf16(q_even) - ZERO_BF16
        s_even: bf16 = f_even * SCALE_BF16
        w = insert_elem(w, i32(2) * i32(i), s_even)

        f_odd: bf16 = bf16(q_odd) - ZERO_BF16
        s_odd: bf16 = f_odd * SCALE_BF16
        w = insert_elem(w, i32(2) * i32(i) + i32(1), s_odd)

        i = i + u32(1)

    store_v(c_out, w)
"""


def main() -> int:
    from aie.iron.pythoc.compiler import compile_pythoc_source

    # Seed lazy intrinsics and constants. The AST walker only auto-imports
    # the canned list inside compile_pythoc_source; everything else must be
    # passed explicitly (see kernels/build.py:181-208 for the pattern).
    from pythoc.aie import set_ctrl_reg

    # The dequant constants are bf16 literals; PythoC accepts Python floats
    # for bf16-typed globals.
    extras = {
        "set_ctrl_reg": set_ctrl_reg,
        "ZERO_BF16": 0.5,
        "SCALE_BF16": 2.0,
        "NUM_PAIRS": 16,
        # Expose u4 to the AST walker even though we don't use it inside
        # the kernel body -- this keeps the smoke aligned with Subtask A
        # (the export is callable as a type from kernel source if needed).
        "u4": u4,
    }

    # Use PYTHOC_DEBUG_DIR so the .ll / .opt.ll / .o all stick around for
    # post-mortem inspection. Without this, compile_pythoc_source writes
    # into a tmpdir that's removed before we can print the .opt.ll path.
    out_dir = tempfile.mkdtemp(prefix="test_pythoc_u4_unpack_")
    print(f"[smoke] output directory: {out_dir}")

    produced = compile_pythoc_source(
        source_code=_KERNEL_SOURCE,
        function_name="test_u4_unpack_bf16",
        target_arch="aie2p",
        output_dir=out_dir,
        verbose=False,
        extra_globals=extras,
    )

    produced = Path(produced)
    print(f"[smoke] compiled .o: {produced}")

    opt_ll = produced.with_suffix(".opt.ll")
    ll = produced.with_suffix(".ll")
    if opt_ll.exists():
        print(f"[smoke] post-opt LLVM IR: {opt_ll}")
    elif ll.exists():
        print(f"[smoke] LLVM IR: {ll}")
    else:
        # Fall back to scanning the output dir for any .opt.ll.
        cands = sorted(Path(out_dir).glob("*.opt.ll"))
        if cands:
            print(f"[smoke] post-opt LLVM IR: {cands[0]}")
        else:
            cands = sorted(Path(out_dir).glob("*.ll"))
            if cands:
                print(f"[smoke] LLVM IR: {cands[0]}")
            else:
                print("[smoke] WARNING: no .ll / .opt.ll found in output_dir")

    if not produced.exists():
        print(f"[smoke] FAIL: compiled object missing: {produced}")
        return 1

    print("[smoke] PASS: PythoC compiled the u4-unpack -> bf16 dequant kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
