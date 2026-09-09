"""Opaque-bit one-column packet aggregation ABI; no hardware imports.

Input/output [worker4,level4,pixel400,lane8] uint16. Workers transmit their
four planes in sequential grant order, XOR-tagging bits with 0x1111*(worker+1).
The diagnostic tag proves data traversed the intended compute worker. Output
keeps native worker-major order; this is not projection or pooling math.
"""
from pathlib import Path
import numpy as np

SHAPE = (4, 4, 400, 8)
ELEMENTS = 51200


def check_input(bits):
    if bits.dtype != np.uint16 or bits.shape != SHAPE:
        raise ValueError(f'expected uint16 input {SHAPE}')


def oracle(bits):
    check_input(bits)
    tags = (np.arange(1, 5, dtype=np.uint16) * 0x1111).reshape(4, 1, 1, 1)
    return np.bitwise_xor(bits, tags)


def frame_input(frame, seed=42):
    if frame < 0:
        raise ValueError('frame must be nonnegative')
    case = ('linear_index', 'random', 'zero', 'ones', 'edge_bits', 'random')[frame % 6]
    if case == 'linear_index':
        # All 51200 source identities fit uniquely in one uint16 sentinel.
        bits = np.arange(ELEMENTS, dtype=np.uint16)
    elif case == 'random':
        bits = np.random.default_rng(seed + frame).integers(0, 65536, ELEMENTS, dtype=np.uint16)
    elif case == 'edge_bits':
        # Include signed zeros, infinities and NaN encodings deliberately:
        # packet transport must preserve all 16 bits, not reinterpret values.
        bits = np.resize(np.array([0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xffc1, 1, 0xffff], np.uint16), ELEMENTS)
        bits = np.roll(bits, frame // 6)
    else:
        bits = np.full(ELEMENTS, 0 if case == 'zero' else 0xffff, np.uint16)
    return case, bits.reshape(SHAPE)


def validate_output(observed, expected):
    if observed.dtype != np.uint16 or observed.shape != SHAPE:
        raise RuntimeError('packet aggregate output has wrong dtype/shape')
    if not np.array_equal(observed, expected):
        coordinate = tuple(int(x) for x in np.argwhere(observed != expected)[0])
        raise RuntimeError(f'packet aggregate mismatch at worker/level/pixel/lane {coordinate}: '
                           f'got {int(observed[coordinate])}, expected {int(expected[coordinate])}')


class PacketAggregate:
    """One context, two persistent BOs, no retries after any runtime failure."""
    def __init__(self, build_dir, backend):
        self.backend, self.failed = backend, False
        root = Path(build_dir).resolve()
        self.xclbin, self.insts = root / 'packet_aggregate.xclbin', root / 'packet_aggregate.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing packet aggregation artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(ELEMENTS, dtype=np.uint16)

    def run(self, bits):
        if self.failed:
            raise RuntimeError('packet aggregation invalid after failure; recover and restart process')
        check_input(bits)
        try:
            self.input.data.reshape(-1)[:] = bits.reshape(-1)
            self.input._sync_to_device()
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.output])
            if not result.is_success():
                raise RuntimeError('packet aggregation runtime returned unsuccessful result')
            observed = self.output.numpy()
            if observed.dtype != np.uint16 or observed.size != ELEMENTS:
                raise RuntimeError('packet aggregation readback has wrong dtype/size')
            return observed.copy().reshape(SHAPE)
        except Exception:
            self.failed = True
            raise
