"""Resident packet-to-gather host ABI and independent coordinate oracle.

I uint16[source4,worker4,level4,pixel400,lane8]. Each worker XORs its words
with 0x1111*(worker+1). O uint16[destination4,stripe25,pixel16,channel512]
replicates level-major concat across four destinations. Runtime uploads native
input only; tagging/gathering in this module is exclusively the CPU oracle.
"""
from pathlib import Path
import numpy as np

INPUT_SHAPE = (4, 4, 4, 400, 8)
OUTPUT_SHAPE = (4, 25, 16, 512)
INPUT_ELEMENTS, OUTPUT_ELEMENTS = 204800, 819200


def check_input(bits):
    if bits.dtype != np.uint16 or bits.shape != INPUT_SHAPE:
        raise ValueError(f'expected uint16 native packet-gather input {INPUT_SHAPE}')


def oracle(bits):
    """Decode every logical output channel explicitly, not via device DMA taps."""
    check_input(bits)
    channel = np.arange(512)
    level = channel // 128
    source = (channel % 128) // 32
    worker = (channel % 32) // 8
    lane = channel % 8
    pixel = np.arange(400)[:, None]
    gathered = bits[source[None], worker[None], level[None], pixel, lane[None]]
    gathered ^= (0x1111 * (worker + 1)).astype(np.uint16)[None]
    return np.broadcast_to(gathered.reshape(25, 16, 512), OUTPUT_SHAPE).copy()


def frame_input(frame, seed=42):
    if frame < 0:
        raise ValueError('frame must be nonnegative')
    case = ('index_low16', 'index_high16', 'random', 'zero', 'ones', 'edge_bits')[frame % 6]
    if case.startswith('index_'):
        identities = np.arange(INPUT_ELEMENTS, dtype=np.uint32)
        bits = (identities if case == 'index_low16' else identities >> 16).astype(np.uint16)
    elif case == 'random':
        bits = np.random.default_rng(seed + frame).integers(0, 65536, INPUT_ELEMENTS, dtype=np.uint16)
    elif case == 'edge_bits':
        words = np.array([0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xffc1, 1, 0xffff], np.uint16)
        bits = np.roll(np.resize(words, INPUT_ELEMENTS), frame // 6)
    else:
        bits = np.full(INPUT_ELEMENTS, 0 if case == 'zero' else 0xffff, np.uint16)
    return case, bits.reshape(INPUT_SHAPE)


def validate_output(observed, expected):
    if observed.dtype != np.uint16 or observed.shape != OUTPUT_SHAPE:
        raise RuntimeError('packet-gather output has wrong dtype/shape')
    if not np.array_equal(observed, expected):
        c = tuple(int(v) for v in np.argwhere(observed != expected)[0])
        k = c[-1]
        raise RuntimeError(f'packet-gather mismatch at destination/stripe/pixel/channel {c}, '
                           f'source={k % 128 // 32}, worker={k % 32 // 8}, level={k // 128}, lane={k % 8}: '
                           f'got {int(observed[c])}, expected {int(expected[c])}')


class PacketGather:
    """Two persistent BOs; one submission uploads native input and reads output.

    No host intermediate is repacked, uploaded or read between aggregation and
    gather. A failed runtime operation invalidates the object without retries.
    """
    def __init__(self, build_dir, backend):
        self.backend, self.failed = backend, False
        root = Path(build_dir).resolve()
        self.xclbin, self.insts = root / 'packet_gather.xclbin', root / 'packet_gather.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing resident packet-gather artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(INPUT_ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(OUTPUT_ELEMENTS, dtype=np.uint16)

    def run(self, bits):
        if self.failed:
            raise RuntimeError('packet-gather invalid after failure; recover and restart process')
        check_input(bits)
        try:
            self.input.data.reshape(-1)[:] = bits.reshape(-1)
            self.input._sync_to_device()
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.output])
            if not result.is_success():
                raise RuntimeError('packet-gather runtime returned unsuccessful result')
            observed = self.output.numpy()
            if observed.dtype != np.uint16 or observed.size != OUTPUT_ELEMENTS:
                raise RuntimeError('packet-gather readback has wrong dtype/size')
            return observed.copy().reshape(OUTPUT_SHAPE)
        except Exception:
            self.failed = True
            raise
