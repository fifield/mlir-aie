"""Host ABI for one-column, four-worker, reusable packet stripe joining.

One frame uploads uint16[25,4,16,16] and reads uint16[25,16,64]. The single
device submission processes all 25 stripes. Workers XOR their input words
with 0x1111*(worker+1); output concatenates workers in channel order per pixel.
All words are opaque bits, not bf16 arithmetic.
"""
from pathlib import Path
import numpy as np

INPUT_SHAPE = (25, 4, 16, 16)
OUTPUT_SHAPE = (25, 16, 64)
ELEMENTS = 25600


def check_input(bits):
    if bits.dtype != np.uint16 or bits.shape != INPUT_SHAPE:
        raise ValueError(f'expected uint16 input {INPUT_SHAPE}')


def oracle(bits):
    check_input(bits)
    tags = (np.arange(1, 5, dtype=np.uint16) * 0x1111).reshape(1, 4, 1, 1)
    return np.bitwise_xor(bits, tags).transpose(0, 2, 1, 3).copy().reshape(OUTPUT_SHAPE)


def frame_input(frame, seed=42):
    if frame < 0:
        raise ValueError('frame must be nonnegative')
    case = ('linear_index', 'stripe_worker_identity', 'random', 'zero', 'ones', 'edge_bits')[frame % 6]
    if case == 'linear_index':
        bits = np.arange(ELEMENTS, dtype=np.uint16).reshape(INPUT_SHAPE)
    elif case == 'stripe_worker_identity':
        stripe = np.arange(1, 26, dtype=np.uint16)[:, None, None, None]
        worker = np.arange(1, 5, dtype=np.uint16)[None, :, None, None]
        bits = np.broadcast_to((stripe << 8) | (worker << 4), INPUT_SHAPE).copy()
    elif case == 'random':
        bits = np.random.default_rng(seed + frame).integers(0, 65536, INPUT_SHAPE, dtype=np.uint16)
    elif case == 'edge_bits':
        words = np.array([0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xffc1, 1, 0xffff], np.uint16)
        bits = np.roll(np.resize(words, ELEMENTS), frame // 6).reshape(INPUT_SHAPE)
    else:
        bits = np.full(INPUT_SHAPE, 0 if case == 'zero' else 0xffff, np.uint16)
    return case, bits


def validate_output(observed, expected):
    if observed.dtype != np.uint16 or observed.shape != OUTPUT_SHAPE:
        raise RuntimeError('stripe join output has wrong dtype/shape')
    if not np.array_equal(observed, expected):
        c = tuple(int(v) for v in np.argwhere(observed != expected)[0])
        raise RuntimeError(f'stripe join mismatch at stripe/pixel/channel {c}: '
                           f'got {int(observed[c])}, expected {int(expected[c])}')


class PacketStripeJoin:
    """Persistent context/BOs, one submission per frame, no failure retries."""
    def __init__(self, build_dir, backend):
        self.backend, self.failed = backend, False
        root = Path(build_dir).resolve()
        self.xclbin, self.insts = root / 'packet_stripe_join.xclbin', root / 'packet_stripe_join.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing packet stripe join artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(ELEMENTS, dtype=np.uint16)

    def run(self, bits):
        if self.failed:
            raise RuntimeError('packet stripe join invalid after failure; recover and restart process')
        check_input(bits)
        try:
            self.input.data.reshape(-1)[:] = bits.reshape(-1)
            self.input._sync_to_device()
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.output])
            if not result.is_success():
                raise RuntimeError('packet stripe join runtime returned unsuccessful result')
            observed = self.output.numpy()
            if observed.dtype != np.uint16 or observed.size != ELEMENTS:
                raise RuntimeError('packet stripe join readback has wrong dtype/size')
            return observed.copy().reshape(OUTPUT_SHAPE)
        except Exception:
            self.failed = True
            raise
