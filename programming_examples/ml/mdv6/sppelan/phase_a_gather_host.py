"""Numerical phase-A/gather ABI; CPU oracle never supplies device intermediates."""
from pathlib import Path
import numpy as np
from sppelan.phase_a_host import pack_shard_weights, pool_levels, values, exact_bits, RAW_WEIGHT_ELEMENTS

INPUT_SHAPE = (400, 256)
WEIGHT_SHAPE = (4, 4, 2, 1056)
OUTPUT_SHAPE = (4, 25, 16, 512)
METADATA_SHAPE = (4, 4, 32)
UPLOAD_BYTES, DOWNLOAD_BYTES = 272384, 1639424


def pack_weights(raw):
    return np.stack([pack_shard_weights(raw, shard) for shard in range(16)]).reshape(WEIGHT_SHAPE)


def frame_weights(trained, frame):
    """Rotate whole trained eight-channel shards, including their BN parameters."""
    if trained.dtype != np.uint16 or trained.shape != (RAW_WEIGHT_ELEMENTS,) or frame < 0:
        raise ValueError('expected trained raw SPP1 weights and nonnegative frame')
    shift = 8 * (frame % 16)
    return np.concatenate([np.roll(trained[:32768].reshape(128, 256), shift, axis=0).reshape(-1),
                           np.roll(trained[32768:32896], shift), np.roll(trained[32896:], shift)])


def frame_input(frame, raw, seed=42):
    if frame < 0:
        raise ValueError('frame must be nonnegative')
    case = ('random', 'negative', 'boundaries', 'cancellation', 'zero', 'signed_zero')[frame % 6]
    x = np.random.default_rng(seed + frame).standard_normal(INPUT_SHAPE).astype(np.float32)
    if case == 'negative':
        x.fill(-0.5)
    elif case in ('zero', 'signed_zero', 'boundaries', 'cancellation'):
        x.fill(0)
        if case == 'boundaries':
            for index, value in ((0, 1), (15, -1), (16, .5), (27, -.5), (28, 1), (383, -1), (384, .5), (399, -1)):
                x[index] = value
        elif case == 'cancellation':
            row = values(raw[:32768]).reshape(128, 256)[8 * ((frame // 6) % 16)]
            left, right = int(np.abs(row[:128]).argmax()), 128 + int(np.abs(row[128:]).argmax())
            if row[right] == 0:
                raise ValueError('cannot construct weighted K128 cancellation input')
            x[:, left], x[:, right] = 1, -row[left] / row[right]
        elif case == 'signed_zero':
            x.view(np.uint32).reshape(-1)[::2] = 0x80000000
    bits = exact_bits(x)
    if not np.isfinite(values(bits)).all():
        raise ValueError('input construction produced nonfinite bf16')
    return case, bits


def gather_oracle(f0):
    """Full-width reference projection → independent first-max pools → K mapping."""
    if f0.dtype != np.uint16 or f0.shape != (400, 128):
        raise ValueError('expected full-width uint16 projection [400,128]')
    levels = np.concatenate([pool_levels(f0[:, 8*s:8*(s+1)]) for s in range(16)], axis=2)
    # Explicit level/channel address decode, independent of device DMA taps.
    k = np.arange(512)
    gathered = levels[k[None] // 128, np.arange(400)[:, None], k[None] % 128]
    return np.broadcast_to(gathered.reshape(25, 16, 512), OUTPUT_SHAPE).copy()


def validate_output(observed, expected, metadata):
    if metadata.dtype != np.uint16 or metadata.shape != METADATA_SHAPE or np.any(metadata):
        coords = np.argwhere(metadata != 0)[:8].tolist()
        raise RuntimeError(f'phase-A gather requires floor0 and zero reserved metadata; nonzero={coords}')
    if observed.dtype != np.uint16 or observed.shape != OUTPUT_SHAPE:
        raise RuntimeError('phase-A gather wrong output dtype/shape')
    if not np.isfinite(values(observed)).all():
        raise RuntimeError('phase-A gather nonfinite output')
    if not np.array_equal(observed, expected):
        coord = tuple(int(v) for v in np.argwhere(observed != expected)[0])
        k = coord[-1]
        raise RuntimeError(f'phase-A gather mismatch destination/stripe/pixel/K={coord}, '
                           f'level={k//128}, source={k%128//32}, worker={k%32//8}, lane={k%8}: '
                           f'got {int(observed[coord])}, expected {int(expected[coord])}')


class PhaseAGather:
    def __init__(self, build_dir, backend):
        self.backend, self.failed = backend, False
        root = Path(build_dir).resolve()
        self.xclbin, self.insts = root / 'spp_phase_a_gather.xclbin', root / 'spp_phase_a_gather.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing phase-A gather artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input, self.weights, self.output, self.metadata = [backend.iron.zeros(int(np.prod(shape)), dtype=np.uint16)
            for shape in (INPUT_SHAPE, WEIGHT_SHAPE, OUTPUT_SHAPE, METADATA_SHAPE)]

    def run(self, bits, weights):
        if self.failed:
            raise RuntimeError('phase-A gather invalid after failure; restart process')
        if bits.dtype != np.uint16 or bits.shape != INPUT_SHAPE or weights.dtype != np.uint16 or weights.shape != WEIGHT_SHAPE:
            raise ValueError('phase-A gather input/weight ABI mismatch')
        try:
            for buffer, data in ((self.input, bits), (self.weights, weights)):
                buffer.data.reshape(-1)[:] = data.reshape(-1)
                buffer._sync_to_device()
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.weights, self.output, self.metadata])
            if not result.is_success():
                raise RuntimeError('phase-A gather runtime returned unsuccessful result')
            output, metadata = self.output.numpy().copy(), self.metadata.numpy().copy()
            if output.dtype != np.uint16 or output.size != np.prod(OUTPUT_SHAPE) or metadata.dtype != np.uint16 or metadata.size != np.prod(METADATA_SHAPE):
                raise RuntimeError('phase-A gather readback ABI mismatch')
            return output.reshape(OUTPUT_SHAPE), metadata.reshape(METADATA_SHAPE)
        except Exception:
            self.failed = True
            raise
