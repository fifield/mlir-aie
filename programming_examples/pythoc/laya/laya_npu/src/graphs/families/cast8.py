"""Decision casts on 8 columns (device_03 cast8, device_04 castrelu8).

One core per column (row 2) on each of the 8 columns. A core forever acquires one
512-word FP32 gather object [256 values | 256 bias] and one 256-element BF16
output object, calls the ``cast_bias`` kernel, and (castrelu8 only) applies
``m13b_relu256`` to the output in place before releasing both. Each column has
its own shim (same column) for input and output.

Host data (n = L * 1024 elements of one token tensor):

* ``linear`` (cast8): per batch row, a 6n-word input holding Q, K, v gather
  objects back to back; Q and K land in outputs whose rows sit 2n apart, v in a
  dense B * n output.
* ``n`` (cast8): per row a 2n-word input; output rows 2n apart.
* ``castrelu`` (castrelu8): per row an 8n-word input (4n values); dense 4n
  output rows.

Every segment (one output of one row) is split into 8 contiguous column ranges
of whole objects, filled and drained on that column's shim, one TaskGroup per
row.

Everything is high-level IRON; nothing drops to lower-level bindings.
"""

import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.runtime.taskgroup import TaskGroup
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile

OBJ_IN, OBJ_OUT = 512, 256  # f32 words per gather object, BF16 elements per output object
COLS = 8
HIDDEN = 1024
f32, u16 = np.float32, np.uint16


def _dims(k, unit):
    """k objects of `unit` elements as BD sizes/strides (a BD dimension wraps at most 512)."""
    a, b = 1, k
    while b > 512:
        a, b = a * 2, b // 2
    assert a * b == k
    return [1, a, b, unit], [0, b * unit, unit, 1]


def _layout(kind, L, B):
    """(operand types, rows); rows = [[(in_word, out_arg, out_elem, objects)] per segment] per batch row."""
    n = L * HIDDEN
    if kind == "linear":
        ops = [(B * 6 * n, f32), ((B - 1) * 2 * n + n, u16), ((B - 1) * 2 * n + n, u16), (B * n, u16)]
        rows = [[(b * 6 * n + s * 2 * n, 1 + s, b * (n if s == 2 else 2 * n), n // OBJ_OUT) for s in range(3)]
                for b in range(B)]
    elif kind == "n":
        ops = [(B * 2 * n, f32), ((B - 1) * 2 * n + n, u16)]
        rows = [[(b * 2 * n, 1, b * 2 * n, n // OBJ_OUT)] for b in range(B)]
    elif kind == "castrelu":
        ops = [(B * 8 * n, f32), (B * 4 * n, u16)]
        rows = [[(b * 8 * n, 1, b * 4 * n, 4 * n // OBJ_OUT)] for b in range(B)]
    elif kind == "proto":  # the configuration build: 2048 objects, linear output
        ops = [(2048 * OBJ_IN, f32), (2048 * OBJ_OUT, u16)]
        rows = None
    return ops, rows


def program(relu, kind="proto", L=64, B=1):
    in_ty = np.ndarray[(OBJ_IN,), np.dtype[f32]]
    out_ty = np.ndarray[(OBJ_OUT,), np.dtype[u16]]
    obj = "kernels.o" if relu else "cast_bias.o"
    cast = Kernel("cast_bias", obj, [in_ty, out_ty])
    kernels = [cast, Kernel("m13b_relu256", obj, [out_ty])] if relu else [cast]

    def core(i, o, k, *r):
        for _ in range_(0x7FFFFFFF):
            x = i.acquire(1)
            y = o.acquire(1)
            k(x, y)
            if r:
                r[0](y)
            i.release(1)
            o.release(1)

    ins, outs, workers = [], [], []
    for col in range(COLS):
        fi = ObjectFifo(in_ty, depth=2, name=f"in{col}")
        fo = ObjectFifo(out_ty, depth=2, name=f"out{col}")
        workers.append(Worker(core, [fi.cons(), fo.prod(), *kernels], tile=Tile(col, 2)))
        ins.append(fi.prod(tile=Tile(col, 0)))
        outs.append(fo.cons(tile=Tile(col, 0)))

    ops, rows = _layout(kind, L, B)

    def tap(arg, off, k, unit):
        sizes, strides = _dims(k, unit)
        return TensorAccessPattern((ops[arg][0],), off, sizes, strides)

    def sequence(*args):
        data, hin, hout = args[:len(ops)], args[len(ops):len(ops) + COLS], args[len(ops) + COLS:]
        if rows is None:  # configuration build: all fills, then all drains
            per = 2048 // COLS
            for c in range(COLS):
                hin[c].fill(data[0], tap=TensorAccessPattern((ops[0][0],), c * per * OBJ_IN, [1, 1, per, OBJ_IN], [0, 0, OBJ_IN, 1]))
            for c in range(COLS):
                hout[c].drain(data[1], tap=TensorAccessPattern((ops[1][0],), c * per * OBJ_OUT, [1, 1, per, OBJ_OUT], [0, 0, OBJ_OUT, 1]), wait=True)
            return
        for segs in rows:
            tg = TaskGroup()
            for i0, oarg, o0, nobj in segs:
                k = nobj // COLS
                for c in range(COLS):
                    hin[c].fill(data[0], tap=tap(0, i0 + c * k * OBJ_IN, k, OBJ_IN), group=tg)
                    hout[c].drain(data[oarg], tap=tap(oarg, o0 + c * k * OBJ_OUT, k, OBJ_OUT), wait=True, group=tg)
            tg.finish()

    types = [np.ndarray[(e,), np.dtype[t]] for e, t in ops]
    rt = Runtime(sequence, types + ins + outs)
    return str(Program(NPU2(), rt, workers=workers).resolve_program())


DEVICES = {"device_03": lambda: program(False), "device_04": lambda: program(True)}

# Sequence names are laya_npu's (M18 export order): for each L in (64..512), B in (1, 2, 4, 8),
# three consecutive names: linear, n, castrelu, starting at sequence_217.
_LB = [(L, B) for L in (64, 128, 256, 512) for B in (1, 2, 4, 8)]
SEQUENCES = [(f"sequence_{217 + 3 * i + j:03d}", "exact",
              lambda kind=kind, L=L, B=B: program(kind == "castrelu", kind, L, B))
             for i, (L, B) in enumerate(_LB) for j, kind in enumerate(("linear", "n", "castrelu"))]
