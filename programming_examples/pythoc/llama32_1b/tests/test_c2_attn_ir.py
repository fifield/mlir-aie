# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""IR-validation tests for the c2_attn collapsed decode device (no HW).

Verifies the STEP-B collapse invariants purely from emitted IR:
  * c2_attn emits ONE compute device + ONE dispatcher, ONE aiex.configure,
    ONE aiex.run (i.e. 1 LoadPDI) -- identical configure/run count to
    c2_merged.
  * the c2_attn host ABI drops the attn_out input and appends q/k/v
    (18 args; arg1 widened to n_groups*4096).
  * the DEFAULT c2_merged emission contains NO c2_attn artifacts (gating is
    leak-free) and keeps its 15-arg signature -- the byte-for-byte regression
    guard for the production default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _build(pack_mode, seq_len=64):
    os.environ["PYTHOC_C2_ATTN_SEQ_LEN"] = str(seq_len)
    from builders.o_gemv_ffn import build_o_gemv_ffn_module
    return build_o_gemv_ffn_module(pack_mode=pack_mode)


def test_c2_attn_one_configure_one_run():
    ir = _build("c2_attn")
    assert ir.count("aiex.configure") == 1, "c2_attn must be ONE configure (1 LoadPDI)"
    assert ir.count("aiex.run") == 1, "c2_attn must be ONE run"
    # 2 devices: the c2_attn compute seg + the dispatcher.
    assert ir.count("aie.device") == 2


def test_c2_attn_abi_extended():
    ir = _build("c2_attn")
    sig = ir.split("@o_gemv_ffn(")[1].split(")")[0]
    assert sig.count("%arg") == 18, "c2_attn ABI = 15 c2 args + q/k/v (18)"
    # arg1 widened to the per-group attn_out scratch (8*4096 = 32768).
    assert "memref<32768xbf16>" in sig


def test_c2_merged_unchanged_no_attn_leak():
    """The production default must stay free of any c2_attn artifact."""
    ir = _build("c2_merged")
    for tok in ("a_q_", "a_gp_", "air_channel_90", "air_channel_91",
                "air_channel_93", "fused_softmax", "matmul_a_b_bf16"):
        assert tok not in ir, f"c2_attn artifact {tok!r} leaked into c2_merged"
    sig = ir.split("@o_gemv_ffn(")[1].split(")")[0]
    assert sig.count("%arg") == 15, "c2_merged stays 15-arg"
    assert ir.count("aiex.configure") == 1


def test_c2_attn_seq_len_guard():
    from builders.c2_attn import build_c2_attn_module
    with pytest.raises(NotImplementedError):
        build_c2_attn_module(seq_len=128)  # single-chunk wiring caps at 64


def test_c2_attn_resident_one_configure():
    """The RESIDENT c2_attn (fixed MAX_CHUNKS=4, runtime-L mask) is still ONE
    device / ONE configure / ONE run = 1 LoadPDI reused for every position --
    the fix for the two-full-fabric-PDI wedge."""
    from builders.c2_attn import (build_c2_attn_resident_module,
                                  c2_attn_resident_kernel_id)
    ir = build_c2_attn_resident_module(8)
    assert ir.count("aiex.configure") == 1
    assert ir.count("aiex.run") == 1
    assert ir.count("aie.device") == 2
    # runtime-L mask: the ILLEGAL vector ``arith.select`` (i1-masked vector,
    # which aie2p can't legalize) must NOT be emitted.  The mask is the runtime
    # v32-block path (whole 8-col blocks via vector.transfer_write of a -inf
    # v32; partial block scalar under scf.if).  A scalar index ``arith.select``
    # (the first_full = boundary_blk + (rem!=0) bump) IS legal and expected.
    import re
    assert not re.search(r"arith\.select.*vector<", ir), \
        "illegal vector arith.select leaked into the runtime mask"
    # one fixed-structure kernel id for ALL positions (not per-position).
    assert c2_attn_resident_kernel_id() in ir


def test_c2_merged_unchanged_under_resident_flag():
    """Setting the resident env flag must not perturb the c2_merged default."""
    os.environ["PYTHOC_C2_ATTN_RESIDENT"] = "1"
    try:
        ir = _build("c2_merged")
    finally:
        os.environ.pop("PYTHOC_C2_ATTN_RESIDENT", None)
    sig = ir.split("@o_gemv_ffn(")[1].split(")")[0]
    assert sig.count("%arg") == 15, "c2_merged stays 15-arg even with resident flag"
    assert "a_q_" not in ir and "fused_softmax" not in ir


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
