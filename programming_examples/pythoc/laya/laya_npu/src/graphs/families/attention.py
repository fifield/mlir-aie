"""All-head attention (devices 05-10, 14, 15): 32 cores, one head-block unit per core per round.

Geometry (M7c S2, quad-major unit order of M9b B1). A unit is one head and a block of R query rows.
Each column has four cores (rows 2..5, roles 0..3) and per round takes column slot s = rnd*8 + col:
head s mod 16, rows 4*R*(s div 16) .. +4R (role j holds the j-th R-row block).

Per column the data flow is:

* ``kv_col{c}``: shim -> memtile (depth 2) -> ``kvb_col{c}`` broadcast to the column's four cores.
  One object is one K or V chunk (k x 64 BF16). The first object of a dispatch is the unit header
  ``[meta record (32 u16) | mask (L u16) | filler from K]``; then per round the head's K in 4C chunks
  and its V in 4C chunks (stage-descending: stages 3, 2, 1, then 0 after K has drained).
* ``inp_col{c}``: shim -> memtile (depth 1) -> split into four Q blocks (R x 64 BF16), ``inp_{c}_{j}``.
* ``out_col{c}``: role 0 (row 2) -> shim. Roles 3 -> 2 -> 1 -> 0 forward their finished rows over the
  cascade; role 0 writes its own unit and the three forwarded ones into consecutive output objects.

Variants (``VARIANTS``): L in 64/128/256/512; family backbone (Q^T staged in ``qk``) or decision
(Q read in place, ``qx`` staging); ``exp`` (vector exponential ``exp_v64`` with its table), ``skip``
(``s2_begin`` decides whether a unit sees any key: ``loops[1]`` gates the passes) and ``keys``
(``s2_merge`` also takes the exponential's buffers). The host ABI is the accepted 7-BO one
(Q, K, V, mask, metadata, statistics, result); statistics is never moved ("nostats").
A B-row sequence repeats the one-batch sequence per batch entry: Q and K entries sit 2n apart, V,
result n apart, mask L apart (n = L*1024).

Below high-level IRON:
* Worker bodies use ``aie.dialects.aie.put_cascade/get_cascade`` and ``vector.transfer_read/write``
  (16-lane f32) for the cascade forwarding of the output rows: IRON has CascadeFlow for the topology
  but no verb to move data over it. ``memref.load``/``arith`` read the chunk count and the skip flag
  that ``s2_header``/``s2_begin`` store in the core's ``loops`` buffer.
"""

import numpy as np

from aie.dialects import arith as _arith, memref, vector
from aie.dialects.aie import get_cascade, put_cascade
from aie.extras import types as T
from aie.extras.dialects import arith
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CascadeFlow, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import if_, range_
from aie.iron.device import NPU2, Tile
from aie.ir import AffineDimExpr, AffineMap

COLUMNS, ROLES, HEADS, META = 8, 4, 16, 32
RL = 16  # rows of the kernels' Q^T staging block
GEOM = {64: (16, 16, 1, 1), 128: (32, 32, 1, 1), 256: (32, 32, 2, 2), 512: (64, 64, 2, 2)}  # q, k, groups, chunks/stage
R_ROWS = {64: 16, 128: 16, 256: 16, 512: 8}  # rows per unit
KV_DEPTH = {64: 2, 128: 2, 256: 2, 512: 1}  # KV objects per core
REUSE_EXPS = {64: True, 128: True, 256: False, 512: False}  # pass 1 keeps exponentials for pass 2 (em)
EXP_SLOTS = 4
QX_WORDS = 1152  # decision Q staging (u32)
TABLE = 115  # exponential table: thresholds A[1..115], then T[0..115]

u16, u32, i32, f32 = np.uint16, np.uint32, np.int32, np.float32


def arr(n, dt):
    return np.ndarray[(n,), np.dtype[dt]]


def exp_table():
    """E1's per-core table (231 words): A[j] = least f32 magnitude a with fl(a*c1) >= j (j = 1..115),
    then T[k] = bits(fl(f32(-k) * c2)) (k = 0..115), with c1 = f32(1/ln 2), c2 = f32(ln 2)."""
    c1, c2, limit = np.float32(1.4426950408889634), np.float32(0.6931471805599453), 0x42A00000

    def ge(bits, j):
        return np.array([bits], u32).view(f32)[0] * c1 >= j

    A = []
    for j in range(1, TABLE + 1):
        lo, hi = 0, limit
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ge(mid, j) else (mid, hi)
        A.append(hi)
    Tt = [int(np.array([np.float32(-k) * c2 if k else np.float32(0.0)], f32).view(u32)[0]) for k in range(TABLE + 1)]
    return np.array(A + Tt, dtype=u32).view(i32)


class Variant:
    def __init__(self, L, family, skip, keys):
        self.L, self.family, self.skip, self.keys = L, family, skip, keys
        q, k, G, C = GEOM[L]
        self.k, self.C, self.R = k, C, R_ROWS[L]
        self.meta_words = 32 * G * C * 16 + 4 * q
        self.exp = family == "backbone" or L == 512  # T-EXP: backbone, decision L512
        self.em = REUSE_EXPS[L]
        self.kv_words = k * 64
        self.out_words = self.R * 32  # f32 words: R x 64 BF16
        self.blocks = L // self.R
        self.rounds = HEADS * self.blocks // 32
        self.n = L * 1024  # elements of one batch entry of Q/K/V/result

    def buffers(self):
        R, L = self.R, self.L
        b = []
        if self.family == "backbone":
            b.append(("qk", RL * 64 + 2 * self.k * RL, u16))
        b += [("unit", 8 * R + (L + META) // 2, f32), ("acc", 2 * R * 64, f32), ("sall", R * L, f32)]
        if self.em:
            b.append(("em", R * L + R * EXP_SLOTS, f32))
        b += [("ostage", self.out_words, f32), ("loops", 2, i32), ("mka", self.k, i32)]
        if self.exp:
            b += [("tab", 2 * TABLE + 1, i32), ("vx", 128, u32), ("vw", 384, u32), ("vz", 64, u32)]
        if self.family == "decision":
            b.append(("qx", QX_WORDS, u32))
        return b

    def unit(self, rnd, col):
        """(head, first row) of column ``col`` in round ``rnd`` (quad-major)."""
        s = rnd * COLUMNS + col
        return s % HEADS, 4 * (s // HEADS) * self.R


def kernels(v):
    """Kernel declarations in the archived order, with their signatures."""
    A = {name: arr(n, dt) for name, n, dt in v.buffers()}
    kv, q = arr(v.kv_words, u16), arr(v.R * 64, u16)
    qk = [A["qk"]] if v.family == "backbone" else []
    em = [A["em"]] if v.em else []
    ex = [A["tab"], A["vx"], A["vw"]] if v.exp else []
    qx = [A["qx"]] if v.family == "decision" else []
    sig = {
        "s2_header": [kv, A["unit"], A["loops"]],
        "s2_begin": [q] + qk + [A["unit"], i32, i32, i32] + ([A["acc"], A["loops"]] if v.skip else []),
        "s2_pass1": [kv] + (qk or [q]) + [A["unit"], A["sall"]] + em + [i32, A["mka"]] + ex + qx,
        "s2_merge": [A["unit"]] + (ex if v.keys else []),
        "s2_pass2": [kv, A["unit"], A["acc"], A["sall"]] + em + [i32, A["mka"]] + ex,
        "m6_finish_output": [A["acc"], A["ostage"]],
    }
    if v.exp:
        sig["exp_v64"] = [A["vz"], A["vx"], A["tab"], A["vw"], i32]
    return {name: Kernel(name, "attention.o", s) for name, s in sig.items()}


def _perm():
    return AffineMap.get(1, 0, [AffineDimExpr.get(0)])


def _value(x):
    return x.op if hasattr(x, "op") else x


def _send(buf, n):
    buf = _value(buf)
    zero = arith.constant(0.0, T.f32())
    for off in range_(0, n, 16):
        put_cascade(vector.transfer_read(T.vector(16, T.f32()), buf, [off], permutation_map=_perm(),
                                         padding=zero, in_bounds=[True]))


def _receive(buf, n):
    buf = _value(buf)
    for off in range_(0, n, 16):
        vector.transfer_write(None, get_cascade(T.vector(16, T.f32())), buf, [off], permutation_map=_perm(),
                              in_bounds=[True])


def _forward(n):
    for _ in range_(0, n, 16):
        put_cascade(get_cascade(T.vector(16, T.f32())))


def core_fn(v, col, j, names):
    def i32c(x):
        return arith.constant(x, T.i32())

    def body(kvb, inp, *rest):
        rest = list(rest)
        out = rest.pop(0) if j == 0 else None
        b = dict(zip(names, rest[: len(names)]))
        k = dict(zip(["s2_header", "s2_begin", "s2_pass1", "s2_merge", "s2_pass2", "m6_finish_output", "exp_v64"],
                     rest[len(names):]))
        qk = [b["qk"]] if v.family == "backbone" else []
        em = [b["em"]] if v.em else []
        ex = [b["tab"], b["vx"], b["vw"]] if v.exp else []
        qx = [b["qx"]] if v.family == "decision" else []
        loops = b["loops"].op

        h = kvb.acquire(1)
        k["s2_header"](h, b["unit"], b["loops"])
        kvb.release(1)
        # 4C (chunks per pass), stored by s2_header: a loaded trip count keeps the chunk loops rolled.
        chunks = _arith.index_cast(T.index(), memref.load(loops, [arith.constant(0, T.index())]))
        for rnd in range_(v.rounds):
            qo = inp.acquire(1)
            k["s2_begin"](qo, *qk, b["unit"], _arith.index_cast(T.i32(), rnd), i32c(col), i32c(j),
                          *([b["acc"], b["loops"]] if v.skip else []))
            if v.exp:
                k["exp_v64"](b["vz"], b["vx"], b["tab"], b["vw"], i32c(1))
            live = None
            if v.skip:  # s2_begin stores whether the unit sees any key
                live = _arith.cmpi(_arith.CmpIPredicate.ne,
                                   memref.load(loops, [arith.constant(1, T.index())]), i32c(0))

            def gated(fn):
                if live is None:
                    fn()
                else:
                    with if_(live):
                        fn()

            for i in range_(0, chunks, 1):  # pass 1: K chunks
                o = kvb.acquire(1)
                gated(lambda: k["s2_pass1"](o, *(qk or [qo]), b["unit"], b["sall"], *em,
                                            _arith.index_cast(T.i32(), i), b["mka"], *ex, *qx))
                kvb.release(1)
            gated(lambda: k["s2_merge"](b["unit"], *(ex if v.keys else [])))
            for i in range_(0, chunks, 1):  # pass 2: V chunks
                o = kvb.acquire(1)
                gated(lambda: k["s2_pass2"](o, b["unit"], b["acc"], b["sall"], *em,
                                            _arith.index_cast(T.i32(), i), b["mka"], *ex))
                kvb.release(1)
            inp.release(1)
            if j == 0:  # own unit, then roles 1, 2, 3 from the cascade
                o = out.acquire(1)
                k["m6_finish_output"](b["acc"], o)
                out.release(1)
                for _ in range_(3):
                    o = out.acquire(1)
                    _receive(o, v.out_words)
                    out.release(1)
            else:
                k["m6_finish_output"](b["acc"], b["ostage"])
                _send(b["ostage"], v.out_words)
                if j < 3:
                    for _ in range_(3 - j):
                        _forward(v.out_words)

    return body


def program(v, B=1):
    names = [name for name, _, _ in v.buffers()]
    table = exp_table() if v.exp else None
    kern = kernels(v)
    workers, inps, kvs, outs = [], [], [], []
    for col in range(COLUMNS):
        inp = ObjectFifo(arr(4 * v.R * 64, u16), depth=1, name=f"inp_col{col}")
        parts = inp.cons().split([j * v.R * 64 for j in range(ROLES)], tile=Tile(col, 1), depths=[1] * ROLES,
                                 obj_types=[arr(v.R * 64, u16)] * ROLES,
                                 names=[f"inp_{col}_{j}" for j in range(ROLES)])
        kv = ObjectFifo(arr(v.kv_words, u16), depth=2, name=f"kv_col{col}")
        kvb = kv.cons().forward(tile=Tile(col, 1), depth=2, name=f"kvb_col{col}")
        out = ObjectFifo(arr(v.out_words, f32), depth=2, name=f"out_col{col}")
        inps.append(inp.prod(tile=Tile(col, 0)))
        kvs.append(kv.prod(tile=Tile(col, 0)))
        outs.append(out.cons(tile=Tile(col, 0)))
        column = []
        for j in range(ROLES):
            args = [kvb.cons(depth=KV_DEPTH[v.L]), parts[j].cons(depth=1)] + ([out.prod()] if j == 0 else [])
            for name, n, dt in v.buffers():
                init = table if name == "tab" else None
                args.append(Buffer(arr(n, dt), initial_value=init, name=f"{name}_{col}_{j}"))
            args += list(kern.values())
            w = Worker(core_fn(v, col, j, names), args, tile=Tile(col, 2 + j), stack_size=4096)
            column.append(w)
        for j in range(ROLES - 1, 0, -1):
            CascadeFlow(column[j], column[j - 1])
        workers += column

    n, L = v.n, v.L
    qk_len, vo_len = (2 * B - 1) * n, B * n
    rows = 4 * v.R
    span = L * 64 // 4  # one V stage (C chunks)

    def tap(size, off, n_, sizes=None, strides=None):
        return TensorAccessPattern((size,), off, sizes or [n_], strides or [1])

    def sequence(Q, K, V, mask, meta, statistics, result, inps, kvs, outs):
        def fill(h, buf, size, off, n_, wait=False):
            return h.fill(buf, tap=tap(size, off, n_), wait=wait, managed=False)

        for b in range(B):
            qkb, vb = b * 2 * n, b * n
            # Header object per column: meta record 0, the mask, filler from K.
            head = [[fill(kvs[c], meta, v.meta_words, 0, META), fill(kvs[c], mask, B * L, b * L, L),
                     fill(kvs[c], K, qk_len, qkb, v.kv_words - META - L, wait=True)] for c in range(COLUMNS)]
            for t in head:
                t[2].await_()
            for t in head:
                for x in t[:2]:
                    x.free()
            for rnd in range(v.rounds):
                live = []
                for c in range(COLUMNS):
                    hd, qb = v.unit(rnd, c)
                    do = outs[c].drain(result, tap=tap(vo_len, vb + qb * 1024 + hd * 64, None, [rows, 64], [1024, 1]),
                                       wait=True, managed=False)
                    tq = fill(inps[c], Q, qk_len, qkb + (hd * L + qb) * 64, rows * 64)
                    tk = fill(kvs[c], K, qk_len, qkb + hd * L * 64, L * 64, wait=True)
                    tv = [fill(kvs[c], V, vo_len, vb + hd * L * 64 + s * span, span) for s in (3, 2, 1)]
                    live.append((do, tq, tk, tv, hd))
                for c, (do, tq, tk, tv, hd) in enumerate(live):
                    tk.await_()
                    tv.append(fill(kvs[c], V, vo_len, vb + hd * L * 64, span))
                for do, tq, tk, tv, hd in live:
                    do.await_()
                    for t in [tq] + tv:
                        t.free()

    rt = Runtime(sequence, [arr(qk_len, u16), arr(qk_len, u16), arr(vo_len, u16), arr(B * L, u16),
                            arr(v.meta_words, u16), arr(B * 32 * L, f32), arr(vo_len, u16), inps, kvs, outs])
    return str(Program(NPU2(), rt, workers=workers).resolve_program())


VARIANTS = {
    "device_05": Variant(64, "backbone", skip=False, keys=False),
    "device_06": Variant(128, "decision", skip=True, keys=False),
    "device_07": Variant(256, "decision", skip=True, keys=False),
    "device_08": Variant(64, "decision", skip=False, keys=False),
    "device_09": Variant(128, "backbone", skip=True, keys=True),
    "device_10": Variant(512, "backbone", skip=True, keys=True),
    "device_14": Variant(256, "backbone", skip=True, keys=True),
    "device_15": Variant(512, "decision", skip=True, keys=True),
}

DEVICES = {name: (lambda v=v: program(v)) for name, v in VARIANTS.items()}

# (device, rows B) -> laya sequence name (M18 export order).
_NAMES = {
    ("device_05", 1): "sequence_192", ("device_05", 2): "sequence_193", ("device_05", 4): "sequence_194",
    ("device_05", 8): "sequence_195", ("device_06", 1): "sequence_196", ("device_07", 1): "sequence_197",
    ("device_15", 1): "sequence_198", ("device_08", 1): "sequence_199", ("device_08", 2): "sequence_200",
    ("device_08", 4): "sequence_201", ("device_08", 8): "sequence_202", ("device_06", 2): "sequence_203",
    ("device_06", 4): "sequence_204", ("device_06", 8): "sequence_205", ("device_07", 2): "sequence_206",
    ("device_09", 1): "sequence_207", ("device_09", 2): "sequence_208", ("device_09", 4): "sequence_209",
    ("device_09", 8): "sequence_210", ("device_14", 1): "sequence_211", ("device_14", 2): "sequence_212",
    ("device_14", 4): "sequence_213", ("device_14", 8): "sequence_214", ("device_10", 1): "sequence_215",
    ("device_10", 2): "sequence_216",
}
SEQUENCES = [(name, "exact", lambda d=d, B=B: program(VARIANTS[d], B)) for (d, B), name in _NAMES.items()]
