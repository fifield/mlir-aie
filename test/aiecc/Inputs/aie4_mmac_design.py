#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Emit an NPU3 design that exercises MMAC lowering for one format.

Used by test/aiecc/aie4-mmac-codegen.test (compile and disassembly checks)
and test/npu-xrt/npu3_mmac_aievec (hardware run).

One core receives
  in0: A0, A1 (M x K each, row-major), then Bt0, Bt1, Bt2 (64 x K each:
       B tiles stored column-major, i.e. B^T row-major)
  in1: C0 (M x 64 accumulator)
and returns three M x 64 results:
  R1 = matmul(A0, B0, C0)                      one tile, accumulator in
  R2 = matmul(A1, B1, R1)                      K = 2 tiles, C carried
  R3 = matrix_region(B<r3>) { mmac(A0[m]) }    row MMACs without acc
                                               (hardware zero_acc)
r3 = 0 restages B0 after B1; r3 = 2 stages a third tile.
With --contract, R1 and R2 are vector.contract over extended operands
(extsi/extui by --signs, extf with fastmath<afn> for floats) for the AIE4
Vector-to-AIEVec matcher.
Bi is vector.transpose(Bti), so the lowering's transpose to the column-major
staging layout cancels and B columns are loaded directly.
With --pair-b-loads, load B columns in 1024-bit pairs before extracting
each column. The staging lowering's extraction and concatenation can then
fold back to a direct load, exercising Peano's staging-load combine.
"""

import argparse

M = 4
N = 64

FORMATS = {
    # A/B element type, K, accumulator type, stored element type (int4 and
    # fp8 are stored as bytes and bitcast after loading).
    "bf16": dict(elt="bf16", k=32, acc="f32", store="bf16"),
    "int8": dict(elt="i8", k=64, acc="i32", store="i8"),
    "int4": dict(elt="i4", k=128, acc="i32", store="i8"),
    "fp8": dict(elt="f8E4M3FN", k=64, acc="f32", store="i8"),
}

# Integer signedness per case: (lhs_signed, rhs_signed). The zero-acc region
# uses the matmul signedness.
SIGNS = {"int8": (False, True), "int4": (True, True)}


def emit(
    fmt,
    r3_tile,
    m_rows=4,
    signs=None,
    contract=False,
    pair_b_loads=False,
):
    global M
    M = m_rows
    signs = signs or SIGNS.get(fmt)
    f = FORMATS[fmt]
    elt, k, acc, store = f["elt"], f["k"], f["acc"], f["store"]
    is_int = fmt in SIGNS
    # Stored elements per logical row of K elements.
    row_store = k // 2 if fmt == "int4" else k
    a_elems = M * row_store
    bt_elems = N * row_store
    in0_elems = 2 * a_elems + 3 * bt_elems
    in1_elems = M * N
    out_elems = 3 * M * N
    off_a = [0, a_elems]
    off_bt = [2 * a_elems + t * bt_elems for t in range(3)]

    lines = []
    w = lines.append
    consts = set()

    def c(v):
        consts.add(v)
        return f"%c{v}"

    counter = [0]

    def fresh(p):
        counter[0] += 1
        return f"%{p}{counter[0]}"

    def load_row(buf, off, count=1):
        """Load one or two logical rows of K elements."""
        lanes = count * k
        stored_lanes = count * row_store
        if fmt == "bf16" or fmt == "int8":
            v = fresh("v")
            w(
                f"      {v} = vector.load {buf}[{c(off)}] : memref<{in0_elems}x{store}>, vector<{lanes}x{elt}>"
            )
            return v
        raw = fresh("raw")
        w(
            f"      {raw} = vector.load {buf}[{c(off)}] : memref<{in0_elems}x{store}>, vector<{stored_lanes}xi8>"
        )
        v = fresh("v")
        w(
            f"      {v} = vector.bitcast {raw} : vector<{stored_lanes}xi8> to vector<{lanes}x{elt}>"
        )
        return v

    def load_matrix(buf, off, rows, pair_rows=False):
        cur = fresh("p")
        w(f"      {cur} = ub.poison : vector<{rows}x{k}x{elt}>")
        step = 2 if pair_rows else 1
        for r in range(0, rows, step):
            # Keep packed int4 pairs in bytes until after column extraction.
            # This preserves the load's register image and avoids asking the
            # generic vector lowering to reshape/extract sub-byte elements.
            packed_pair = pair_rows and fmt == "int4"
            pair_k, pair_elt = (row_store, store) if packed_pair else (k, elt)
            if packed_pair:
                loaded = fresh("raw")
                w(
                    f"      {loaded} = vector.load {buf}[{c(off + r * row_store)}] : memref<{in0_elems}x{store}>, vector<{2*pair_k}x{pair_elt}>"
                )
            else:
                loaded = load_row(buf, off + r * row_store, step)
            if pair_rows:
                pair = fresh("pair")
                w(
                    f"      {pair} = vector.shape_cast {loaded} : vector<{2*pair_k}x{pair_elt}> to vector<2x{pair_k}x{pair_elt}>"
                )
            for j in range(step):
                row = loaded
                if pair_rows:
                    row = fresh("v")
                    w(
                        f"      {row} = vector.extract {pair}[{j}] : vector<{pair_k}x{pair_elt}> from vector<2x{pair_k}x{pair_elt}>"
                    )
                    if packed_pair:
                        unpacked = fresh("v")
                        w(
                            f"      {unpacked} = vector.bitcast {row} : vector<{pair_k}x{pair_elt}> to vector<{k}x{elt}>"
                        )
                        row = unpacked
                nxt = fresh("m")
                w(
                    f"      {nxt} = vector.insert {row}, {cur} [{r+j}] : vector<{k}x{elt}> into vector<{rows}x{k}x{elt}>"
                )
                cur = nxt
        return cur

    body = []

    def sign_attrs(lhs=True, rhs=True):
        if not is_int:
            return ""
        ls, rs = signs
        parts = []
        if lhs:
            parts.append(f"lhs_signed = {str(ls).lower()}")
        if rhs:
            parts.append(f"rhs_signed = {str(rs).lower()}")
        return " {" + ", ".join(parts) + "}"

    accty = f"vector<{M}x{N}x{acc}>"
    aty = f"vector<{M}x{k}x{elt}>"
    bty = f"vector<{k}x{N}x{elt}>"

    # Collect the body first, then emit the constants it needs.
    saved = lines
    lines = body
    w = lines.append
    w(
        f"      %in0 = aie.objectfifo.acquire @in0(Consume, 1) : memref<{in0_elems}x{store}>"
    )
    w(
        f"      %in1 = aie.objectfifo.acquire @in1(Consume, 1) : memref<{in1_elems}x{acc}>"
    )
    w(
        f"      %out = aie.objectfifo.acquire @out(Produce, 1) : memref<{out_elems}x{acc}>"
    )
    a0 = load_matrix("%in0", off_a[0], M)
    a1 = load_matrix("%in0", off_a[1], M)
    for t in sorted({0, 1, r3_tile}):
        bt = load_matrix("%in0", off_bt[t], N, pair_b_loads)
        w(
            f"      %b{t} = vector.transpose {bt}, [1, 0] : vector<{N}x{k}x{elt}> to {bty}"
        )
    cur = fresh("p")
    w(f"      {cur} = ub.poison : {accty}")
    for r in range(M):
        v = fresh("v")
        w(
            f"      {v} = vector.load %in1[{c(r * N)}] : memref<{in1_elems}x{acc}>, vector<{N}x{acc}>"
        )
        nxt = fresh("m")
        w(
            f"      {nxt} = vector.insert {v}, {cur} [{r}] : vector<{N}x{acc}> into {accty}"
        )
        cur = nxt
    c0 = cur
    if contract:
        # R1/R2 as vector.contract over widened operands, for the
        # Vector-to-AIEVec matcher. Floating formats carry afn.
        def widen(v, ty, is_lhs):
            wide = ty.replace(f"x{elt}>", f"x{acc}>")
            if is_int:
                op = "arith.extsi" if signs[0 if is_lhs else 1] else "arith.extui"
            else:
                op = "arith.extf"
            r = fresh("w")
            w(f"      {r} = {op} {v} : {ty} to {wide}")
            return r, wide

        fm = "" if is_int else ", fastmath = #arith.fastmath<afn>"

        def contract_op(res, a, b, c):
            aw, awty = widen(a, aty, True)
            bw, bwty = widen(b, bty, False)
            w(
                f"      {res} = vector.contract {{indexing_maps = [affine_map<(m, n, k) -> (m, k)>, "
                f"affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], "
                f'iterator_types = ["parallel", "parallel", "reduction"], '
                f"kind = #vector.kind<add>{fm}}} {aw}, {bw}, {c} : {awty}, {bwty} into {accty}"
            )

        contract_op("%r1", a0, "%b0", c0)
        contract_op("%r2", a1, "%b1", "%r1")
    else:
        w(
            f"      %r1 = aievec.aie4.matmul {a0}, %b0, {c0}{sign_attrs()} : {aty}, {bty} into {accty}"
        )
        w(
            f"      %r2 = aievec.aie4.matmul {a1}, %b1, %r1{sign_attrs()} : {aty}, {bty} into {accty}"
        )
    region_attrs = ""
    if is_int:
        region_attrs = f" attributes {{rhs_signed = {str(signs[1]).lower()}}}"
    fmt_kw = {"bf16": "bf16", "int8": "int8", "int4": "int4", "fp8": "fp8"}[fmt]
    r3_b = f"%b{r3_tile}"
    w(
        f"      %r3 = aievec.aie4.matrix_region {fmt_kw} {r3_b} : {bty} -> {accty}{region_attrs} {{"
    )
    cur = fresh("p")
    w(f"        {cur} = ub.poison : {accty}")
    for r in range(M):
        a = fresh("a")
        w(f"        {a} = vector.extract {a0}[{r}] : vector<{k}x{elt}> from {aty}")
        x = fresh("x")
        w(
            f"        {x} = aievec.aie4.mmac {fmt_kw} {a}{sign_attrs()} : vector<{k}x{elt}> -> vector<{N}x{acc}>"
        )
        nxt = fresh("m")
        w(
            f"        {nxt} = vector.insert {x}, {cur} [{r}] : vector<{N}x{acc}> into {accty}"
        )
        cur = nxt
    w(f"        aievec.aie4.matrix_yield {cur} : {accty}")
    w("      }")
    for i, res in enumerate(["%r1", "%r2", "%r3"]):
        for r in range(M):
            v = fresh("o")
            w(f"      {v} = vector.extract {res}[{r}] : vector<{N}x{acc}> from {accty}")
            w(
                f"      vector.store {v}, %out[{c(i * M * N + r * N)}] : memref<{out_elems}x{acc}>, vector<{N}x{acc}>"
            )
    w("      aie.objectfifo.release @in0(Consume, 1)")
    w("      aie.objectfifo.release @in1(Consume, 1)")
    w("      aie.objectfifo.release @out(Produce, 1)")
    lines = saved
    w = lines.append

    w("module {")
    w("  aie.device(npu3) {")
    w("    %shim = aie.tile(0, 0)")
    w("    %t = aie.tile(0, 2)")
    w(
        f"    aie.objectfifo @in0(%shim, {{%t}}, 1 : i32) : !aie.objectfifo<memref<{in0_elems}x{store}>>"
    )
    w(
        f"    aie.objectfifo @in1(%shim, {{%t}}, 1 : i32) : !aie.objectfifo<memref<{in1_elems}x{acc}>>"
    )
    w(
        f"    aie.objectfifo @out(%t, {{%shim}}, 1 : i32) : !aie.objectfifo<memref<{out_elems}x{acc}>>"
    )
    w("    %core = aie.core(%t) {")
    for v in sorted(consts):
        w(f"      %c{v} = arith.constant {v} : index")
    lines.extend(body)
    w("      aie.end")
    w("    }")
    w(
        f"    aie.runtime_sequence(%a0: memref<{in0_elems}x{store}>, %a1: memref<{in1_elems}x{acc}>, %a2: memref<{out_elems}x{acc}>) {{"
    )
    for i, (fifo, arg, n) in enumerate(
        [
            ("in0", "%a0", in0_elems),
            ("in1", "%a1", in1_elems),
            ("out", "%a2", out_elems),
        ]
    ):
        ty = {
            "%a0": f"memref<{in0_elems}x{store}>",
            "%a1": f"memref<{in1_elems}x{acc}>",
            "%a2": f"memref<{out_elems}x{acc}>",
        }[arg]
        w(f"      %t{i} = aiex.dma_configure_task_for @{fifo} {{")
        w(
            f"        aie.dma_bd({arg} : {ty} offset = 0 len = {n} sizes = [1, 1, 1, {n}] strides = [0, 0, 0, 1])"
        )
        w("        aie.end")
        w("      } {issue_token = true}")
    w("      aiex.dma_start_task(%t0)")
    w("      aiex.dma_start_task(%t1)")
    w("      aiex.dma_start_task(%t2)")
    w("      aiex.dma_await_task(%t0)")
    w("      aiex.dma_await_task(%t1)")
    w("      aiex.dma_await_task(%t2)")
    w("    }")
    w("  }")
    w("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIE4 MMAC NPU3 design")
    parser.add_argument("format", choices=sorted(FORMATS))
    parser.add_argument("--r3-tile", type=int, choices=[0, 2], default=0)
    parser.add_argument("--m", type=int, default=4, help="rows per matmul")
    parser.add_argument(
        "--signs",
        choices=["uu", "us", "su", "ss"],
        help="integer lhs/rhs signedness (u/s); default per format",
    )
    parser.add_argument(
        "--contract",
        action="store_true",
        help="emit R1/R2 as vector.contract (needs --enable-aie4-vectorization)",
    )
    parser.add_argument(
        "--pair-b-loads",
        action="store_true",
        help="load B columns in 1024-bit pairs, then extract each column",
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    with open(args.output, "w") as out:
        signs = None
        if args.signs:
            signs = (args.signs[0] == "s", args.signs[1] == "s")
        out.write(
            emit(
                args.format,
                args.r3_tile,
                args.m,
                signs,
                args.contract,
                args.pair_b_loads,
            )
        )
