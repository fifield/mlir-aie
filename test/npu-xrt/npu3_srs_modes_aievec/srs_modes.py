# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

"""Explicit-mode aievec.aie4.srs on NPU3: generator, reference and goldens.

One variant per (rounding, saturation) choice. Each core narrows 256 i32
inputs with signed i8, unsigned i8 and signed i16 SRS at several shifts; the
host compares output bytes with the NumPy reference.
"""

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith as arith_d
from aie.dialects import memref as memref_d
from aie.dialects import vector as vector_d
from aie.helpers.dialects.aievec_aie4 import (
    CONV_EVEN,
    POS_INF,
    SAT,
    SYM_SAT,
    SYM_ZERO,
    TRUNCATE,
    srs,
)
from aie.ir import (
    AffineMap,
    AffineMapAttr,
    IndexType,
    IntegerType,
    StringAttr,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

SHIFTS = [0, 1, 12, 32, 59]
N_VECS = 4  # 64-lane input vectors
IN_WORDS = 64 * N_VECS  # i32
SHIFT_WORDS = 512  # i16 words per shift: i8s (128) | i8u (128) | i16s (256)
TAIL = SHIFT_WORDS * len(SHIFTS)  # 2560: ambient's crRnd / crSat readback
OUT_WORDS = TAIL + 32  # i16

RND_FLOOR, RND_POS_INF, RND_SYM_ZERO, RND_CONV_EVEN, RND_CONV_ODD = 0, 9, 10, 12, 13
SAT_TRUNC, SAT_SAT, SAT_SYM = 0, 1, 3

# Poison written by poison.o's main() before the core runs.
POISON_RND, POISON_SAT = RND_CONV_ODD, SAT_SYM

# variant -> (srs keyword modes, (crRnd, crSat) at run time)
VARIANTS = {
    "posinf_sat": (dict(rounding=POS_INF, saturation=SAT), (RND_POS_INF, SAT_SAT)),
    "conveven_symsat": (
        dict(rounding=CONV_EVEN, saturation=SYM_SAT),
        (RND_CONV_EVEN, SAT_SYM),
    ),
    "symzero_trunc": (
        dict(rounding=SYM_ZERO, saturation=TRUNCATE),
        (RND_SYM_ZERO, SAT_TRUNC),
    ),
    "ambient": (dict(), (POISON_RND, POISON_SAT)),
    "sat_only": (dict(saturation=SAT), (POISON_RND, SAT_SAT)),
}


def round_ref(v, shift, rnd):
    """x * 2^-shift rounded by crRnd, before saturation (int64)."""
    x = np.asarray(v, dtype=np.int64)
    if shift == 0:
        return x
    q = x >> shift  # floor(x / 2^shift); arithmetic shift
    f = x - (q << shift)  # fractional numerator, 0 <= f < 2^shift
    half = np.int64(1) << (shift - 1)
    gt, eq = f > half, f == half
    odd = (q & 1) == 1  # parity of the floor integer
    up = {
        RND_FLOOR: np.zeros_like(gt),
        RND_POS_INF: gt | eq,  # fraction >= 0.5
        RND_SYM_ZERO: gt | (eq & (x < 0)),  # ties toward zero
        RND_CONV_EVEN: gt | (eq & odd),
        RND_CONV_ODD: gt | (eq & ~odd),
    }[rnd]
    return q + up.astype(np.int64)


def srs_ref(v, shift, rnd, sat, signed, width):
    """Scale by 2^-shift, round by crRnd, then saturate or truncate by crSat.

    The accumulator lanes are signed; `signed` selects only the output range.
    """
    r = round_ref(v, shift, rnd)
    if signed:
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if sat == SAT_SYM:
            lo += 1  # Symmetric saturation uses -hi as the minimum.
    else:
        lo, hi = 0, (1 << width) - 1  # unsigned: symmetric == saturate
    if sat == SAT_TRUNC:  # Keep the low output-width bits.
        m = r & ((1 << width) - 1)
        if signed:
            m = np.where(m > hi, m - (1 << width), m)
        out = m
    else:
        out = np.clip(r, lo, hi)
    return out.astype(
        {(8, True): np.int8, (8, False): np.uint8, (16, True): np.int16}[
            (width, signed)
        ]
    )


def self_check():
    T = 4096
    one = lambda v, s, rnd, sat, sg, w: int(srs_ref([v], s, rnd, sat, sg, w)[0])
    rows = [
        ((2048, 12, 9, 1, True, 8), 1),
        ((2048, 12, 12, 1, True, 8), 0),
        ((2048, 12, 13, 1, True, 8), 1),
        ((2048, 12, 0, 1, True, 8), 0),
        ((-2048, 12, 9, 1, True, 8), 0),
        ((-2048, 12, 12, 1, True, 8), 0),
        ((-2048, 12, 13, 1, True, 8), -1),
        ((-2048, 12, 0, 1, True, 8), -1),
        ((127 * T + 2048, 12, 9, 1, True, 8), 127),
        ((127 * T + 2048, 12, 9, 0, True, 8), -128),
        ((-128 * T, 12, 9, 3, True, 8), -127),
        ((-128 * T, 12, 9, 1, True, 8), -128),
        ((-2049, 12, 9, 1, False, 8), 0),
        ((-2049, 12, 9, 0, False, 8), 255),
        ((-1, 59, 0, 1, True, 8), -1),
        ((-1, 59, 9, 1, True, 8), 0),
        ((200, 0, 9, 1, True, 8), 127),
        ((200, 0, 9, 0, True, 8), -56),
        ((-200, 0, 9, 3, True, 8), -127),
        ((2048, 12, 10, 0, True, 8), 0),
        ((-2048, 12, 10, 0, True, 8), 0),
        ((6144, 12, 10, 0, True, 8), 1),
        ((-6144, 12, 10, 0, True, 8), -1),
        ((2049, 12, 10, 0, True, 8), 1),
        ((127 * T + 2048, 12, 10, 0, True, 8), 127),
        ((128 * T, 12, 10, 0, True, 8), -128),
    ]
    for rnd, want in zip((9, 12, 13, 0), (2, 2, 1, 1)):
        rows.append(((6144, 12, rnd, 1, True, 8), want))
    for rnd, want in zip((9, 12, 13, 0), (-1, -2, -1, -2)):
        rows.append(((-6144, 12, rnd, 1, True, 8), want))
    for rnd, want in zip((9, 12, 13, 0), (0, 0, -1, -1)):
        rows.append(((-(2**31), 32, rnd, 1, True, 8), want))
    for args, want in rows:
        got = one(*args)
        assert got == want, f"reference self-check {args}: got {got}, want {want}"


def make_input():
    T = 4096
    planted = [
        0, 1, -1, 2, -2, 3, -3,
        2047, 2048, 2049, -2047, -2048, -2049,
        3 * T // 2, -3 * T // 2, 5 * T // 2, -5 * T // 2,
        127 * T + 2047, 127 * T + 2048, 128 * T,
        -128 * T, -128 * T - 2048, -128 * T - 2049,
        255 * T + 2047, 255 * T + 2048,
        32767 * T + 2047, 32767 * T + 2048,
        -32768 * T, -32768 * T - 2049,
        127, 128, -128, -129, 255, 256, 200, -200,
        32767, 32768, -32768, -32769,
        2**31 - 1, -(2**31), -(2**31) + 1, 2**30, -(2**30),
        2**31 - 2**30, -(2**31 - 2**30),
    ]  # fmt: skip
    i = 0
    while len(planted) < 64:
        planted.append(i * 4099 - 131072)
        i += 1
    rng = np.random.default_rng(2026)
    rest = []
    for _ in range(N_VECS - 1):
        n = 64
        a = rng.integers(-(2**20), 2**20, n // 3)
        b = rng.integers(-(2**31), 2**31, n // 3)
        c = rng.integers(-300, 301, n - 2 * (n // 3)) * 4096 + 2048
        v = np.concatenate([a, b, c])
        rng.shuffle(v)
        rest.append(v)
    data = np.concatenate([np.array(planted, dtype=np.int64)] + rest)
    assert data.min() >= -(2**31) and data.max() < 2**31
    return data.astype(np.int32)


def golden(data, rnd, sat, ambient):
    out = np.zeros(OUT_WORDS, dtype=np.int16)
    for k, s in enumerate(SHIFTS):
        base = SHIFT_WORDS * k
        for v in range(N_VECS):
            a = data[64 * v : 64 * v + 64]
            i8s = srs_ref(a, s, rnd, sat, True, 8)
            i8u = srs_ref(a, s, rnd, sat, False, 8)
            i16 = srs_ref(a, s, rnd, sat, True, 16)
            out[base + 32 * v : base + 32 * v + 32] = i8s.view(np.int16)
            out[base + 128 + 32 * v : base + 128 + 32 * v + 32] = i8u.view(np.int16)
            out[base + 256 + 64 * v : base + 256 + 64 * v + 64] = i16
    if ambient:
        out[TAIL] = rnd
        out[TAIL + 1] = sat
    return out


def coverage(data, rnd, sat):
    """Check coverage of rounding ties, saturation bounds and unsigned wrapping."""
    a = np.asarray(data[:64], dtype=np.int64)
    pos_tie = neg_tie = tie_differs = pos_sat = neg_sat = sym_min = False
    uneg = False
    for s in SHIFTS:
        r = round_ref(a, s, rnd)
        # a negative rounded value, which the unsigned outputs clip or wrap
        uneg |= bool((r < 0).any())
        if s > 0:
            tie = (a - ((a >> s) << s)) == (1 << (s - 1))
            pos_tie |= bool((tie & (a > 0)).any())
            neg_tie |= bool((tie & (a < 0)).any())
            tie_differs |= bool((tie & (r != round_ref(a, s, RND_FLOOR))).any())
        for w in (8, 16):
            lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
            pos_sat |= bool((r > hi).any())
            neg_sat |= bool((r < lo).any())
            # symmetric and asymmetric saturation differ exactly at r <= lo
            sym_min |= bool((r <= lo).any())
    missing = [
        name
        for name, ok in (
            ("positive tie", pos_tie),
            ("negative tie", neg_tie),
            ("tie that differs from floor", tie_differs),
            ("positive saturation or wrap", pos_sat),
            ("negative saturation or wrap", neg_sat),
            ("symmetric minimum", sym_min or sat != SAT_SYM),
            ("unsigned negative input", uneg),
        )
        if not ok
    ]
    return missing


def srs_design(dev, modes, ambient):
    i32 = IntegerType.get_signless(32)
    i16 = IntegerType.get_signless(16)

    @device(dev)
    def device_body():
        v64i32 = VectorType.get([64], i32)
        v32i32 = VectorType.get([32], i32)
        v32i16 = VectorType.get([32], i16)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        in_ty = np.ndarray[(IN_WORDS,), np.dtype[np.int32]]
        out_ty = np.ndarray[(OUT_WORDS,), np.dtype[np.int16]]

        if ambient:
            # External readback calls are allowed only in the ambient variant.
            get_rnd = external_func(
                "get_rnd", inputs=[], outputs=[np.int32], link_with="poison.o"
            )
            get_sat = external_func(
                "get_sat", inputs=[], outputs=[np.int32], link_with="poison.o"
            )

        shim = tile(0, 0)
        core_tile = tile(0, 2)
        in_fifo = object_fifo("in", shim, core_tile, 1, in_ty)
        out_fifo = object_fifo("out", core_tile, shim, 1, out_ty)

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                idx = lambda n: arith_d.constant(IndexType.get(), n)
                pad = arith_d.constant(i32, 0)

                def store16(val, at):
                    vector_d.transfer_write(
                        None, val, elem_out, [idx(at)], id_map, [True]
                    )

                for k, s in enumerate(SHIFTS):
                    base = SHIFT_WORDS * k
                    for v in range(N_VECS):
                        a = vector_d.transfer_read(
                            v64i32, elem_in, [idx(64 * v)], id_map, pad, [True]
                        )
                        y8s = srs(a, shift=s, signed=True, **modes)
                        y8u = srs(a, shift=s, signed=False, **modes)
                        store16(vector_d.bitcast(v32i16, y8s), base + 32 * v)
                        store16(vector_d.bitcast(v32i16, y8u), base + 128 + 32 * v)
                        for h in range(2):
                            b = vector_d.transfer_read(
                                v32i32,
                                elem_in,
                                [idx(64 * v + 32 * h)],
                                id_map,
                                pad,
                                [True],
                            )
                            y16 = srs(b, shift=s, signed=True, **modes)
                            store16(y16, base + 256 + 64 * v + 32 * h)
                # DMA copies the entire FIFO element, including the tail.
                # Initialize tile memory before any ambient mode readback.
                zero = arith_d.constant(i16, 0)
                store16(vector_d.broadcast(v32i16, zero), TAIL)
                if ambient:
                    memref_d.store(
                        arith_d.trunci(i16, get_rnd()), elem_out, [idx(TAIL)]
                    )
                    memref_d.store(
                        arith_d.trunci(i16, get_sat()), elem_out, [idx(TAIL + 1)]
                    )
                in_fifo.release(ObjectFifoPort.Consume, 1)
                out_fifo.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            a_task = shim_dma_single_bd_task(
                in_fifo, A, sizes=[1, 1, 1, IN_WORDS], issue_token=True
            )
            c_task = shim_dma_single_bd_task(
                out_fifo, C, sizes=[1, 1, 1, OUT_WORDS], issue_token=True
            )
            dma_start_task(a_task, c_task)
            dma_await_task(a_task, c_task)


def main():
    parser = argparse.ArgumentParser(description="AIE4 explicit-mode SRS")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("-o", "--output", required=True, help="MLIR output")
    parser.add_argument("--golden", required=True, help="golden output (i16)")
    parser.add_argument("--input", required=True, help="input data (i32)")
    args = parser.parse_args()

    self_check()
    modes, (rnd, sat) = VARIANTS[args.variant]
    ambient = args.variant == "ambient"
    data = make_input()
    missing = coverage(data, rnd, sat)
    if missing:
        sys.exit(f"coverage lost for {args.variant}: {', '.join(missing)}")
    data.astype("<i4").tofile(args.input)
    golden(data, rnd, sat, ambient).astype("<i2").tofile(args.golden)

    with mlir_mod_ctx() as ctx:
        srs_design(AIEDevice.npu3, modes, ambient)
        if not ambient:
            # Link the poison main() without adding an external call in the core.
            for op in ctx.module.body.operations[0].regions[0].blocks[0].operations:
                if op.operation.name == "aie.core":
                    op.operation.attributes["link_with"] = StringAttr.get("poison.o")
        res = ctx.module.operation.verify()
        if res is not True:
            sys.exit(str(res))

    with open(args.output, "w") as f:
        f.write(str(ctx.module))


if __name__ == "__main__":
    main()
