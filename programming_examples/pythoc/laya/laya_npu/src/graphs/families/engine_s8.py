"""device_s8 (M37): the B1-split dense engine on all 8 columns x 4 rows (32 cores), for the B1/L64 shapes that run
the engine on its own device (exact/b1_l64, the exact and fast B1/L64 pairs).

It is engine_cb.py (device_cb's engine half, M36) with COLS = 8 and A for core row r through shim/memtile 2r, as on
device_02. At B = 1 the two 64-row halves of the 128-row block (core rows 0-1 and 2-3) get different weight tiles:
each half takes half of its column's tiles (half_plan), so every core runs half of device_02's tile count (out/down 1,
qkvR 3 = Q K V, gate/up 3 pairs) instead of both halves duplicating batch row 0. Bit-identical to device_02 per output.

Streams per column c: B for both halves through one shim channel, shim c -> memtile c; the memtile object
[half-0 32 x 64 | half-1 32 x 64] is split into two 2-core broadcasts. Y is joined at memtile c. Shims 0, 2, 4, 6 carry
A and B (2 of 2 MM2S).

Host weights (src/packing/engine_cb.py, cols=8) are interleaved per column, per local tile slot i and per K object k:
[half 0 tile i object k | half 1 tile i object k]. Gate/up pads are stored in place: 96 tiles where device_02 holds 82.
The qkvR epilogue needs RoPE on local tiles j < 2 (Q, K): the s8 core builds compile dispatch.cc with the rule
3 * j < 2 * ntile (build_dense.py), which is j < 4 for device_02's ntile 6.

build() is the device alone with one op's sequence; the core-build template is
device_s8.core_template().
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects import memref
from aie.extras import types as T
from aie.extras.dialects import arith
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.runtime.taskgroup import TaskGroup

M, KK, NN = 32, 32, 64          # leaf tile
ROWS, COLS = 4, 8
HIDDEN, FFN = 1024, 2624
PAIRS = 41                      # gate/up tile pairs (2624 / 64)
OPS = ("out", "down", "gateup", "qkvR", "ln")
RECORDS = ("out", "down", "gateup", "qkvR", "ln@64", "ln@128", "ln@256", "ln@512")
HDR_SLOT = 6
QUEUE = 4                       # drain BDs per shim channel between syncs (L64 plans)
u16 = np.uint16
sys_maxsize = 9223372036854775807


# ------------------------------------------------------------------------------------------ model geometry
def column_blocks():
    """Gate/up column blocks: column c owns pairs c, c + 8, ...; its tiles start at `start` (2 per pair)."""
    out, start = [], 0
    for c in range(COLS):
        pairs = list(range(c, PAIRS, COLS))
        out.append(dict(pairs=pairs, start=start, tiles=2 * len(pairs)))
        start += 2 * len(pairs)
    return out


def half_plan(op, c, h):
    """Core half h of column c: B tiles in order (canonical tile ids; gate/up pairs padded by repeating the half's
    first pairs), the number of real tiles, and per output tile j its (operand, column offset in units, pitch)."""
    if op == "gateup":
        pairs = column_blocks()[c]["pairs"]
        half = 3                                        # ceil(6 / 2): every core runs 3 pairs
        mine = pairs[:half] if h == 0 else pairs[half:]
        real = len(mine)
        mine = mine + mine[: half - real]               # pads = the chunk's first pairs again (what the pad fill sends)
        return dict(tiles=[t for p in mine for t in (2 * p, 2 * p + 1)], real=2 * real,
                    outs=[("Y", p * NN, FFN) for p in mine])
    T = kn(op)[1] // NN // COLS                         # tiles of the column
    if op == "qkvR":
        js = [h, 2 + h, 4 + h]                          # Q j 0-1, K 2-3, V 4-5 of the column
        tiles = [c + COLS * j for j in js]
        return dict(tiles=tiles, real=len(tiles), outs=[("QKV"[t // 16], (t % 16) * 64 * 64, 64) for t in tiles])
    tiles = [c + COLS * j for j in range(T // 2 * h, T // 2 * (h + 1))]
    return dict(tiles=tiles, real=len(tiles), outs=[("Y", t * NN, kn(op)[1]) for t in tiles])


def weight_chunks(op):
    """Host layout of the split fast weights: (c, h) chunks of real tiles, c-major. {(c, h): (start tile, real)}"""
    out, start = {}, 0
    for c in range(COLS):
        for h in range(2):
            hp = half_plan(op, c, h)
            out[(c, h)] = (start, hp["real"])
            start += hp["real"]
    return out


GATEUP_W = 96 * KK * KK * NN                  # 8 columns x 6 slots x 2 halves of 1024 x 64 tiles = 96 tiles (82 + 14 pads)


def kn(op):
    return {"out": (HIDDEN, HIDDEN), "down": (FFN, HIDDEN), "gateup": (HIDDEN, 2 * FFN), "qkvR": (HIDDEN, 3 * HIDDEN)}[op]


def tiles_per_core(op):
    return len(half_plan(op, 0, 0)["tiles"])


def operands(op, L):
    """[(name, single-row elements, row stride or None (shared), dtype)] of a sequence, HDR in slot 6."""
    n = L * HIDDEN
    table = {
        "out": [("A", n, n, bfloat16), ("W", HIDDEN * HIDDEN, None, bfloat16), ("Y", n, 2 * n, u16)],
        "down": [("A", L * FFN, L * FFN, bfloat16), ("W", FFN * HIDDEN, None, bfloat16), ("Y", n, 2 * n, u16)],
        "gateup": [("A", n, 2 * n, bfloat16), ("W", GATEUP_W, None, bfloat16), ("Y", L * FFN, L * FFN, u16),
                   ("P", L * 512, L * 512, u16)],
        "qkvR": [("A", n, 2 * n, bfloat16), ("W", HIDDEN * 3 * HIDDEN, None, bfloat16), ("TPL", 16 * L * 128, None, bfloat16),
                 ("Q", n, 2 * n, u16), ("K", n, 2 * n, u16), ("V", n, n, u16)],
        "ln": [("R", 2 * n, 2 * n, u16), ("TPL", 3 * n, None, u16), ("X", 2 * n, 2 * n, u16)],
    }[op]
    dummies = [(f"D{i}", 16, None, u16) for i in range(len(table), HDR_SLOT)]
    return table + dummies + [("HDR", len(RECORDS) * M * KK, None, u16)]


def record_offset(op, L):
    return RECORDS.index(f"ln@{L}" if op == "ln" else op) * M * KK


def gelu_lut():
    """The GELU epilogue's table: 512 zero words, then bf16 GELU (exact erf form, as torch evaluates it in FP32)
    of every bf16 in [0x3B00, 0x4100] and of its negation."""
    import torch
    import torch.nn.functional as F

    raw = torch.cat((torch.arange(15104, 16641), torch.arange(15104, 16641) + 32768)).to(torch.uint16)
    lut = F.gelu(raw.view(torch.bfloat16).float(), approximate="none").to(torch.bfloat16).view(torch.uint16).numpy()
    return np.concatenate([np.zeros(512, u16), lut.astype(u16)])


# ------------------------------------------------------------------------------------------ the device
def topology():
    A_l2 = np.ndarray[(M * KK,), np.dtype[bfloat16]]
    B_l2 = np.ndarray[(KK * NN,), np.dtype[bfloat16]]
    Y_l2 = np.ndarray[(M * NN * ROWS,), np.dtype[u16]]
    A_l1 = np.ndarray[(M, KK), np.dtype[bfloat16]]
    B_l1 = np.ndarray[(KK, NN), np.dtype[bfloat16]]
    Y_l1 = np.ndarray[(M, NN), np.dtype[u16]]

    def blocked(rows, cols):  # row-major rows x cols -> [rows/8][cols/8][8][8] blocks
        return TensorAccessPattern((rows * cols,), 0, [rows // 8, cols // 8, 8, 8], [8 * cols, 8, cols, 1])

    A3, A2 = [], []
    for r in range(ROWS):
        A3.append(ObjectFifo(A_l2, name=f"A_L3L2_{r}", depth=2))
        A2.append(A3[r].cons().split([0], obj_types=[A_l1], names=[f"A_L2L1_{r}"], to_stream=[blocked(M, KK)],
                                     tile=Tile(2 * r, 1))[0])
    B3, B2, Y3, Y2 = [], [], [], [[None] * COLS for _ in range(ROWS)]
    # Y: the memtile walks each joined 32 x 64 core tile back out of its [4][8][8][8] blocks.
    unblock = TensorAccessPattern((M * NN,), 0, [M // 8, 8, NN // 8, 8], [8 * NN, 8, 64, 1])
    for c in range(COLS):
        B3.append(ObjectFifo(np.ndarray[(2 * KK * NN,), np.dtype[bfloat16]], name=f"B_L3L2_{c}", depth=2))
        B2.append(B3[c].cons().split([0, KK * NN], obj_types=[B_l1, B_l1], names=[f"B_L2L1_{c}_{h}" for h in range(2)],
                                     to_stream=[blocked(KK, NN)] * 2, tile=Tile(c, 1)))
        Y3.append(ObjectFifo(Y_l2, name=f"Y_L2L3_{c}", depth=2, to_stream=unblock))
        parts = Y3[c].prod().join([M * NN * r for r in range(ROWS)], obj_types=[Y_l1] * ROWS,
                                  names=[f"Y_L1L2_{c}_{r}" for r in range(ROWS)], depths=[1] * ROWS, tile=Tile(c, 1))
        for r in range(ROWS):
            Y2[r][c] = parts[r]
    return A3, A2, B3, B2, Y3, Y2, A_l1, B_l1, Y_l1


def parts():
    """Engine workers and the shim endpoints (Ap per core row, Bp per column, Yc per column)."""
    A3, A2, B3, B2, Y3, Y2, A_l1, B_l1, Y_l1 = topology()
    C_ty = np.ndarray[(M, NN), np.dtype[np.float32]]
    G_ty = np.ndarray[(M * NN,), np.dtype[u16]]
    S_ty = np.ndarray[(3586,), np.dtype[u16]]
    T_ty = np.ndarray[(M * 64,), np.dtype[u16]]
    H_ty = np.ndarray[(8,), np.dtype[np.int32]]
    i32 = np.int32
    zero = Kernel("bf16_zero_kernel", "bf16_zero_kernel.o", [C_ty])
    leaf = Kernel("laya_gemm_native", "laya_gemm_native.o", [A_l1, B_l1, C_ty])
    hk = Kernel("m9a_header", "engine.o", [A_l1, H_ty])
    tk = Kernel("m9a_table", "engine.o", [A_l1, T_ty, C_ty, H_ty, i32, i32])
    pk = Kernel("m9a_pass", "engine.o", [C_ty, G_ty, H_ty, i32])
    fk = Kernel("m9a_final", "engine.o", [C_ty, T_ty, G_ty, G_ty, S_ty, Y_l1, H_ty, i32])

    def make_core(col):
        def core(a, b, y, cbuf, tbuf, xbuf, gbuf, sbuf, hbuf, zk, lk, hk_, tk_, pk_, fk_):
            e = a.acquire(1)
            hk_(e, hbuf)
            a.release(1)

            def field(i):
                return arith.index_cast(memref.load(hbuf.op, [arith.constant(i, T.index())]), T.index())

            units, ntab, ntile, npass, kt, npre = field(0), field(1), field(2), field(3), field(4), field(7)
            for t in range_(0, npre, 1):     # section prelude (LN: gamma, beta)
                e = a.acquire(1)
                tk_(e, tbuf, cbuf, hbuf, arith.index_cast(t, T.i32()), arith.constant(-1, T.i32()))
                a.release(1)
            for _ in range_(0, units, 1):
                for t in range_(0, ntab, 1):  # table objects (qkvR: cos, sin; LN: the unit's rows)
                    e = a.acquire(1)
                    tk_(e, tbuf, cbuf, hbuf, arith.index_cast(t, T.i32()), arith.constant(col, T.i32()))
                    a.release(1)
                for j in range_(0, ntile, 1):
                    for p in range_(0, npass, 1):
                        zk(cbuf)
                        for _ in range_(0, kt, 1):
                            ea = a.acquire(1)
                            eb = b.acquire(1)
                            lk(ea, eb, cbuf)
                            a.release(1)
                            b.release(1)
                        pk_(cbuf, gbuf, hbuf, arith.index_cast(p, T.i32()))
                    o = y.acquire(1)
                    fk_(cbuf, tbuf, xbuf, gbuf, sbuf, o, hbuf, arith.index_cast(j, T.i32()))
                    y.release(1)

        return core

    lut = gelu_lut()
    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            scratch = Buffer(G_ty, name=f"x16_gate16_{c}_{r}")  # qkvR's x16, gate/up's gate16, LN's FP32 row
            bufs = [Buffer(C_ty, name=f"C_{c}_{r}"), Buffer(T_ty, name=f"table_{c}_{r}"), scratch, scratch,
                    Buffer(S_ty, initial_value=lut, name=f"gelu_stage_{c}_{r}"), Buffer(H_ty, name=f"hdr_{c}_{r}")]
            workers.append(Worker(make_core(c), [A2[r].cons(), B2[c][r // 2].cons(), Y2[r][c].prod()] + bufs + [zero, leaf, hk, tk, pk, fk],
                                  tile=Tile(c, 2 + r), stack_size=0xD00))

    Ap = [f.prod(tile=Tile(2 * r, 0)) for r, f in enumerate(A3)]
    Bp = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(B3)]
    Yc = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(Y3)]
    return workers, Ap, Bp, Yc


def seq(op, L, B, fast, Ap, Bp, Yc, empty=False):
    """(argument types, sequence function over the tensor arguments) of one op; empty: no transfers."""
    ops = operands(op, L)
    ext = [e if st is None else (B - 1) * st + e for _, e, st, _ in ops]
    plan = [] if empty else sections(op, L, B, fast)

    def sequence(*args):
        data = {nm: (a, e) for (nm, _, _, _), a, e in zip(ops, args, ext)}

        def tap(nm, sizes, strides, off):
            return TensorAccessPattern((data[nm][1],), off, sizes, strides)

        for sec in plan:
            for g, drains in enumerate(sec["drains"]):
                tg = TaskGroup()
                for c, nm, sz, sv, off in drains:
                    Yc[c].drain(data[nm][0], tap=tap(nm, sz, sv, off), wait=True, group=tg)
                if g == 0:
                    for kind, idx, nm, sz, sv, off in sec["fills"]:
                        (Ap if kind == "A" else Bp)[idx].fill(data[nm][0], tap=tap(nm, sz, sv, off), group=tg)
                tg.finish()

    arg_ty = [np.ndarray[(e,), np.dtype[dt]] for (_, _, _, dt), e in zip(ops, ext)]
    return arg_ty, sequence


def build(op=None, L=128, B=1, fast=False):
    """The engine half alone as a device (used to build its core ELFs)."""
    workers, Ap, Bp, Yc = parts()
    arg_ty, sequence = seq(op or "out", L, B, fast, Ap, Bp, Yc, empty=op is None)
    rt = Runtime(sequence, arg_ty + Yc + Bp + Ap)
    return str(Program(NPU2(), rt, workers=workers).resolve_program())


# ------------------------------------------------------------------------------------------ the runtime sequences
def header_fill(op, L, row):
    return ("A", row, "HDR", [1, 1, M, KK], [0, 0, KK, 1], record_offset(op, L))


def weight_fills(op, fast):
    """B fills: column c's weight tiles (T per column, KT K objects each).

    fast: the host stores each column's tiles contiguously, so one tile is a linear run; the outer size-1 dims keep
    the tile count as the BD's iteration dimension (a 2-D [count, tile] walk would fold into one long run)."""
    K, N = kn(op)
    KT, T = K // KK, tiles_per_core(op)
    tile = KT * KK * NN                     # elements of one compact tile
    assert fast
    span = 2 * tile                         # one slot: half 0's tile and half 1's, interleaved by K object
    return [("B", c, "W", [T, 1, 1, span], [span, 0, 0, 1], c * T * span) for c in range(COLS)]


def sections(op, L, B, fast=False):
    """[dict(fills=[(A|B, row|col, operand, sizes, strides, offset)], drains=[[(col, operand, sizes, strides, offset)] per group])]."""
    stride = {nm: st for nm, _, st, _ in operands(op, L)}
    if op == "ln":
        return ln_sections(L, B, stride)
    if L == 64:
        return l64_sections(op, B, stride, fast)
    K, N = kn(op)
    KT, T, RB, rows = K // KK, tiles_per_core(op), L // (ROWS * M), ROWS * M
    blocks = column_blocks()
    out = []
    for b in range(B):
        off = {nm: 0 if st is None else b * st for nm, st in stride.items()}
        for rb in range(RB):
            drains, fills = [], []
            for c in range(COLS):
                if op in ("out", "down"):
                    drains.append((c, "Y", [1, T, rows, NN], [0, COLS * NN, N, 1], off["Y"] + rb * rows * N + c * NN))
                elif op == "gateup":
                    real = len(blocks[c]["pairs"])
                    drains.append((c, "Y", [1, real, rows, NN], [0, COLS * NN, FFN, 1], off["Y"] + rb * rows * FFN + c * NN))
                    if T // 2 > real:
                        drains.append((c, "P", [1, T // 2 - real, rows, NN], [0, 0, NN, 1], off["P"] + (rb * COLS + c) * rows * NN))
                else:
                    for nm in ("Q", "K", "V"):
                        drains.append((c, nm, [1, 2, rows, NN], [0, 8 * L * 64, 64, 1], off[nm] + c * L * 64 + rb * rows * 64))
            for r in range(ROWS):
                fills.append(header_fill(op, L, r))
                if op == "qkvR":
                    pos = (rb * ROWS + r) * M
                    fills.append(("A", r, "TPL", [1, 2, M, KK], [0, KK, 128, 1], pos * 128 + 64))
                fills.append(("A", r, "A", [T, KT, M, KK], [0, KK, K, 1], off["A"] + (rb * ROWS + r) * M * K))
            fills += weight_fills(op, fast)
            out.append(dict(fills=fills, drains=[drains]))
    return out


def ln_sections(L, B, stride):
    """Column c's cores (rows r) hold token rows 32u + 8r + c: hidden -> X.A, normalized -> X.B."""
    UR = ROWS * COLS                 # token rows per unit (8-column engine: 32)
    n, units = L * HIDDEN, L // UR
    out = []
    for b in range(B):
        drains = [(c, "X", [units, ROWS, 2, HIDDEN], [UR * HIDDEN, COLS * HIDDEN, n, 1], b * stride["X"] + c * HIDDEN) for c in range(COLS)]
        fills = []
        for r in range(ROWS):
            fills.append(header_fill("ln", L, r))
            fills.append(("A", r, "TPL", [1, 1, 1, 2 * HIDDEN], [0, 0, 0, 1], HIDDEN))          # gamma, beta
            fills.append(("A", r, "R", [units, COLS, 2, HIDDEN], [UR * HIDDEN, HIDDEN, n, 1], b * stride["R"] + COLS * r * HIDDEN))
        out.append(dict(fills=fills, drains=[drains]))
    return out


def l64_sections(op, B, stride, fast):
    """L = 64: a 128-row block is two 64-row halves (core rows 0-1, 2-3); section s gives half h batch row 2s + h
    (B = 1: row 0 to both). Core row r reads rows (r % 2) * 32 .. + 32 of its half's row."""
    L, half = 64, 64
    K, N = kn(op)
    KT, T = K // KK, tiles_per_core(op)
    blocks = column_blocks()
    out = []
    for sec in ([(0, 0)] if B == 1 else [(2 * s, 2 * s + 1) for s in range(B // 2)]):
        fills = []
        for r in range(ROWS):
            h, sub = r // 2, r % 2
            fills.append(header_fill(op, L, r))
            if op == "qkvR":
                fills.append(("A", r, "TPL", [1, 2, M, KK], [0, KK, 128, 1], sub * M * 128 + 64))
            fills.append(("A", r, "A", [T, KT, M, KK], [0, KK, K, 1], sec[h] * stride["A"] + sub * M * K))
        fills += weight_fills(op, fast)

        def tiles(nm, offs, pitch):
            hs, base = (sec[1] - sec[0]) * stride[nm], sec[0] * stride[nm]
            step = offs[1] - offs[0] if len(offs) > 1 else 0
            if hs > 0 and all(o - offs[0] == i * step for i, o in enumerate(offs)) and (len(offs) == 1 or step > 0):
                return [(nm, [len(offs), 2, half, 64], [step, hs, pitch, 1], base + offs[0])]
            return [(nm, [2, 1, half, 64], [hs, 0, pitch, 1], base + o) for o in offs]

        assert B == 1, "split engine: B = 1 only"
        per_col = []
        for c in range(COLS):               # output j of the joined object: rows 0-63 = half 0's tile, 64-127 = half 1's
            h0, h1 = half_plan(op, c, 0), half_plan(op, c, 1)
            d = []
            for (n0, o0, p0), (n1, o1, p1) in zip(h0["outs"], h1["outs"]):
                assert n0 == n1 and p0 == p1 and o1 > o0, (op, c, o0, o1)
                d.append((n0, [2, 1, half, 64], [o1 - o0, 0, p0, 1], o0))
            per_col.append(d)
        groups = max((len(x) + QUEUE - 1) // QUEUE for x in per_col)
        out.append(dict(fills=fills, drains=[[(c, *d) for c in range(COLS) for d in per_col[c][g * QUEUE:(g + 1) * QUEUE]]
                                             for g in range(groups)]))
    return out
