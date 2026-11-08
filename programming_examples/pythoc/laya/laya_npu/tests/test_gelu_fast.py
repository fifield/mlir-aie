"""The branchless GELU epilogue (src/dense/wrapper_m8b_d1.cc, M8B_GELU_FAST, used by dense_m2n2)
computes, lane by lane, exactly what gelu_observable computes, for every one of the 65536 BF16
inputs, with the gelu_stage LUT of src/dense/dense_native.mlir.in. Host model of both; the
device equality was checked exhaustively when the change was made (m35/engine/gelu)."""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stage():
    text = (ROOT / "src/dense/dense_native.mlir.in").read_text()
    blobs = re.findall(
        r'sym_name = "gelu_stage_\d_\d"\} : memref<3586xui16> = dense<"0x([0-9A-Fa-f]+)">', text
    )
    assert len(blobs) == 32 and len(set(blobs)) == 1
    return np.frombuffer(bytes.fromhex(blobs[0]), "<u2")


def gelu_observable(raw, s):
    """src/dense/gelu_observable.py, first loop, element by element."""
    out = np.empty(raw.size, np.int64)
    for i, r in enumerate(raw.tolist()):
        mag, sign = r & 32767, r & 32768
        if mag < 15104:
            v = ((mag - 128) if mag >= 256 else ((mag >> 1) + ((mag & 1) & ((mag >> 1) & 1)))) | sign
        elif mag > 16640:
            v = r if sign == 0 else sign
        else:
            v = int(s[512 + mag - 15104 + (1537 if sign else 0)])
        out[i] = v
    return out


def branchless(raw, s):
    """m8b_gelu_chunk on int16 lanes: clamped index, one aligned 32-bit LUT load, selects."""
    x = raw.astype(np.int64)
    x = np.where(x >= 32768, x - 65536, x)
    mag, neg = x & 0x7FFF, x < 0
    sign = np.where(neg, -32768, 0)
    idx = np.clip(mag, 15104, 16640) - 15104 + np.where(neg, 1537, 0)
    woff = 2 * (idx & 0x7FFE)
    word = s[512:].view("<u4")[woff >> 2].astype(np.int64)
    table = np.where(idx & 1, word >> 16, word & 0xFFFF)
    q = mag >> 1
    half = q + ((mag & 1) & (q & 1))  # vector SRS by 1 under crRnd 12 (round half to even)
    small = np.where(mag < 256, half | sign, (mag - 128) | sign)
    big = np.where(neg, sign, x)
    out = np.where(mag < 15104, small, np.where(mag > 16640, big, table))
    return out & 0xFFFF


def test_branchless_gelu_equals_gelu_observable_for_every_bf16():
    s = stage()
    assert s.size == 3586 and not s[:512].any()
    raw = np.arange(65536, dtype=np.int64)
    assert np.array_equal(branchless(raw, s), gelu_observable(raw, s))
