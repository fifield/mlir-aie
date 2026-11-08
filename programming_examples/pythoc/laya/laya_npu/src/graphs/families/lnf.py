"""Final LayerNorm (device_12): one core normalises one 1024-wide token row at a time.

Each host call normalises L rows of a (L, 1024) BF16 tensor for B batch entries.
``in0`` streams x rows; ``in1`` streams one 3072-element LayerNorm template row
[x | gamma | beta] per token (the host passes L copies, so the buffer is 3n long and
is re-read from offset 0 for every batch entry); ``out0`` returns the normalised rows.
Batch entries sit 2n apart in the x and output buffers (the arena layout), as in the
residual family.

Everything is high-level IRON; no lower-level bindings were needed.
"""

import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.runtime.taskgroup import TaskGroup
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile

HIDDEN = 1024
TEMPLATE = 3 * HIDDEN  # [x | gamma | beta]
u16 = np.uint16
sys_maxsize = 9223372036854775807


def program(L=128, B=1):
    n = L * HIDDEN  # elements in one batch entry of x / output
    row_ty = np.ndarray[(HIDDEN,), np.dtype[u16]]
    tmpl_ty = np.ndarray[(TEMPLATE,), np.dtype[u16]]
    x = ObjectFifo(row_ty, depth=2, name="in0")
    tmpl = ObjectFifo(tmpl_ty, depth=2, name="in1")
    out = ObjectFifo(row_ty, depth=2, name="out0")
    lnf = Kernel("m8b_lnf_row", "m8b_lnf.o", [row_ty, tmpl_ty, row_ty])

    def core(xs, ts, os, kernel):
        for _ in range_(sys_maxsize):
            for _ in range_(16):
                a = xs.acquire(1)
                t = ts.acquire(1)
                o = os.acquire(1)
                kernel(a, t, o)
                xs.release(1)
                ts.release(1)
                os.release(1)

    worker = Worker(core, [x.cons(), tmpl.cons(), out.prod(), lnf], tile=Tile(0, 2), stack_size=3328)

    io_len = (B - 1) * 2 * n + n
    tmpl_len = 3 * n

    def sequence(xa, ta, oa, xf, tf, of):
        for b in range(B):
            tg = TaskGroup()
            xf.fill(xa, tap=TensorAccessPattern((io_len,), b * 2 * n, [n], [1]), group=tg)
            tf.fill(ta, tap=TensorAccessPattern((tmpl_len,), 0, [tmpl_len], [1]), group=tg)
            of.drain(oa, tap=TensorAccessPattern((io_len,), b * 2 * n, [n], [1]), wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(io_len,), np.dtype[u16]], np.ndarray[(tmpl_len,), np.dtype[u16]],
                            np.ndarray[(io_len,), np.dtype[u16]],
                            x.prod(tile=Tile(0, 0)), tmpl.prod(tile=Tile(0, 0)), out.cons(tile=Tile(0, 0))])
    return str(Program(NPU2(), rt, workers=[worker]).resolve_program())


DEVICES = {"device_12": program}

# Sequence names are laya_npu's (they come from the M18 export order).
_NAMES = {
    (64, 1): "sequence_000", (64, 2): "sequence_012", (64, 4): "sequence_024", (64, 8): "sequence_036",
    (128, 1): "sequence_048", (128, 2): "sequence_060", (128, 4): "sequence_072", (128, 8): "sequence_084",
    (256, 1): "sequence_096", (256, 2): "sequence_108", (256, 4): "sequence_120", (256, 8): "sequence_132",
    (512, 1): "sequence_144", (512, 2): "sequence_156", (512, 4): "sequence_168", (512, 8): "sequence_180",
}
SEQUENCES = [(name, "exact", lambda L=L, B=B: program(L, B)) for (L, B), name in _NAMES.items()]
