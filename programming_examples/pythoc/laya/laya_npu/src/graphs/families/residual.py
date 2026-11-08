"""Residual add (device_11): one core adds two token rows into one.

Each host call adds two (L, 1024) BF16 tensors stored back to back, for B batch
entries. The shim gathers matching 256-element rows of both operands into one
512-element object; the core writes the 256-element sum.
"""

import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.runtime.taskgroup import TaskGroup
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile

ROW = 256
HIDDEN = 1024
u16 = np.uint16


def program(L=64, B=1):
    n = L * HIDDEN  # elements in one operand of one batch entry
    rows = n // ROW
    in_ty = np.ndarray[(2 * ROW,), np.dtype[u16]]
    out_ty = np.ndarray[(ROW,), np.dtype[u16]]
    source = ObjectFifo(in_ty, depth=2, name="input")
    output = ObjectFifo(out_ty, depth=2, name="output")
    residual = Kernel("residual", "residual.o", [in_ty, out_ty])

    def core(src, dst, kernel):
        for _ in range_(sys_maxsize):
            for _ in range_(256):
                a = src.acquire(1)
                b = dst.acquire(1)
                kernel(a, b)
                src.release(1)
                dst.release(1)

    worker = Worker(core, [source.cons(), output.prod(), residual], tile=Tile(0, 2), stack_size=3328)

    # Batch entries sit 2n apart in both buffers; the output buffer ends after the
    # last entry's sum, so it holds (B - 1) * 2n + n elements.
    in_len, out_len = B * 2 * n, (B - 1) * 2 * n + n

    def sequence(a, c, src, dst):
        for b in range(B):
            tg = TaskGroup()
            # A BD wraps at most 512 rows here; longer inputs repeat it with the
            # BD iteration (outermost) dimension.
            reps = max(1, rows // 512)
            sizes, strides = [reps, rows // reps, 2, ROW], [rows // reps * ROW, ROW, n, 1]
            src.fill(a, tap=TensorAccessPattern((in_len,), b * 2 * n, sizes, strides), group=tg)
            dst.drain(c, tap=TensorAccessPattern((out_len,), b * 2 * n, [n], [1]), wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(in_len,), np.dtype[u16]], np.ndarray[(out_len,), np.dtype[u16]],
                            source.prod(tile=Tile(0, 0)), output.cons(tile=Tile(0, 0))])
    return str(Program(NPU2(), rt, workers=[worker]).resolve_program())


sys_maxsize = 9223372036854775807

DEVICES = {"device_11": program}

# Sequence names are laya_npu's (they come from the M18 export order).
_NAMES = {
    (64, 1): "sequence_007", (64, 2): "sequence_019", (64, 4): "sequence_031", (64, 8): "sequence_043",
    (128, 1): "sequence_055", (128, 2): "sequence_067", (128, 4): "sequence_079", (128, 8): "sequence_091",
    (256, 1): "sequence_103", (256, 2): "sequence_115", (256, 4): "sequence_127", (256, 8): "sequence_139",
    (512, 1): "sequence_151", (512, 2): "sequence_163", (512, 4): "sequence_175", (512, 8): "sequence_187",
}
SEQUENCES = [(name, "exact", lambda L=L, B=B: program(L, B)) for (L, B), name in _NAMES.items()]
