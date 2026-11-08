"""LayerNorm + residual (device_13): eight cores, one per column, each LNr-ing L / 8 token rows.

Each host call works on B batch entries of L token rows (n = L * 1024 BF16 values per operand):

* R (in0): the two residual operands A | B stored back to back (2n per entry). The shim gathers, per
  token row, four 256-value pieces of A interleaved with the matching pieces of B: one 2,048-value object
  [A0 B0 .. A3 B3].
* TPL (in1): one template row [x | gamma | beta] (3,072 values) per token row, shared by all batch
  entries (3n, read only).
* H (out0) and N (out1): the hidden sum and the normalized row, 1,024 values per token row.

Core (c, 2) is fed and drained by the shim DMA of its own column (c, 0) (no memtile) and handles token
rows [c m, (c + 1) m), m = L / 8. Batch entries sit 2n apart in R, H and N; H and N end after the last
entry's n values, so they hold (B - 1) * 2n + n values.

Everything is high-level IRON; nothing drops to lower-level bindings.
"""

import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.runtime.taskgroup import TaskGroup
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile

COLS = 8
HIDDEN = 1024
PIECE = 256                       # residual gather granule
REPEAT = 16                       # rows per pass of the core's row loop
u16 = np.uint16
sys_maxsize = 9223372036854775807


def _ty(n):
    return np.ndarray[(n,), np.dtype[u16]]


def program(L=128, B=1):
    n = L * HIDDEN
    m = L // COLS                       # token rows per core
    in0_ty, in1_ty, out_ty = _ty(2 * HIDDEN), _ty(3 * HIDDEN), _ty(HIDDEN)
    lnr = Kernel("m8b_lnr_row", "m8b_lnr.o", [in0_ty, in1_ty, out_ty, out_ty])

    in0 = [ObjectFifo(in0_ty, depth=2, name=f"in0_{c}") for c in range(COLS)]
    in1 = [ObjectFifo(in1_ty, depth=2, name=f"in1_{c}") for c in range(COLS)]
    out0 = [ObjectFifo(out_ty, depth=2, name=f"out0_{c}") for c in range(COLS)]
    out1 = [ObjectFifo(out_ty, depth=2, name=f"out1_{c}") for c in range(COLS)]

    def core(r, t, h, y, kernel):
        for _ in range_(sys_maxsize):
            for _ in range_(REPEAT):
                a = r.acquire(1)
                b = t.acquire(1)
                c = h.acquire(1)
                d = y.acquire(1)
                kernel(a, b, c, d)
                r.release(1)
                t.release(1)
                h.release(1)
                y.release(1)

    workers = [Worker(core, [in0[c].cons(), in1[c].cons(), out0[c].prod(), out1[c].prod(), lnr],
                      tile=Tile(c, 2), stack_size=0xD00) for c in range(COLS)]

    r_len, t_len, o_len = B * 2 * n, 3 * n, (B - 1) * 2 * n + n
    pieces = m * HIDDEN // PIECE        # 256-value pieces per core per operand

    def sequence(R, T, H, N, *fifos):
        f_in0, f_in1, f_out0, f_out1 = (fifos[i * COLS:(i + 1) * COLS] for i in range(4))
        for b in range(B):
            tg = TaskGroup()
            for c in range(COLS):
                off = b * 2 * n + c * m * HIDDEN
                f_out0[c].drain(H, tap=TensorAccessPattern((o_len,), off, [m * HIDDEN], [1]), wait=True, group=tg)
                f_out1[c].drain(N, tap=TensorAccessPattern((o_len,), off, [m * HIDDEN], [1]), wait=True, group=tg)
            for c in range(COLS):
                off = b * 2 * n + c * m * HIDDEN
                f_in0[c].fill(R, tap=TensorAccessPattern((r_len,), off, [pieces, 2, PIECE], [PIECE, n, 1]), group=tg)
                f_in1[c].fill(T, tap=TensorAccessPattern((t_len,), c * m * 3 * HIDDEN, [m * 3 * HIDDEN], [1]),
                              group=tg)
            tg.finish()

    shim = lambda c: Tile(c, 0)
    rt = Runtime(sequence, [_ty(r_len), _ty(t_len), _ty(o_len), _ty(o_len)]
                 + [f.prod(tile=shim(c)) for c, f in enumerate(in0)]
                 + [f.prod(tile=shim(c)) for c, f in enumerate(in1)]
                 + [f.cons(tile=shim(c)) for c, f in enumerate(out0)]
                 + [f.cons(tile=shim(c)) for c, f in enumerate(out1)])
    return str(Program(NPU2(), rt, workers=workers).resolve_program())


DEVICES = {"device_13": program}

# Sequence names are laya_npu's (they come from the M18 export order).
_NAMES = {
    (64, 1): "sequence_006", (64, 2): "sequence_018", (64, 4): "sequence_030", (64, 8): "sequence_042",
    (128, 1): "sequence_054", (128, 2): "sequence_066", (128, 4): "sequence_078", (128, 8): "sequence_090",
    (256, 1): "sequence_102", (256, 2): "sequence_114", (256, 4): "sequence_126", (256, 8): "sequence_138",
    (512, 1): "sequence_150", (512, 2): "sequence_162", (512, 4): "sequence_174", (512, 8): "sequence_186",
}
SEQUENCES = [(name, "exact", lambda L=L, B=B: program(L, B)) for (L, B), name in _NAMES.items()]
