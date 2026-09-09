"""Phase-A single channel shard ABI, weight packing and independent pooling.

Normal args I[400,256], W[2,1056], O[4,400,8], M[32], all uint16.
Each physical weight chunk contains 1024 blocked matrix values, 8 BN scales,
8 biases and 16 zero padding elements, making the next chunk 64-byte aligned.
Pool-only args I[400,8], O[4,400,8], M[32]. Metadata must be all zero: floor
rounding mode followed by reserved zeros. No helper changes device rounding.
"""
from pathlib import Path
import numpy as np

INPUT_ELEMENTS, WEIGHT_ELEMENTS, OUTPUT_ELEMENTS = 102400, 2112, 12800
RAW_WEIGHT_ELEMENTS = 128 * 256 + 256


def values(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def exact_bits(floats):
    """Encode representable bf16 values; intended for maxima of bf16 inputs."""
    return (np.asarray(floats, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def pack_shard_weights(raw, shard):
    if raw.dtype != np.uint16 or raw.shape != (RAW_WEIGHT_ELEMENTS,) or type(shard) is not int or not 0 <= shard < 16:
        raise ValueError('expected full flat uint16 SPP1 weights and integer shard in [0,16)')
    channels = slice(shard * 8, (shard + 1) * 8)
    matrix = raw[:32768].reshape(128, 256)[channels]
    scale, bias = raw[32768:32896][channels], raw[32896:][channels]
    result = np.zeros((2, 1056), np.uint16)
    for block in range(2):
        part = matrix[:, block * 128:(block + 1) * 128]
        result[block, :1024] = part.reshape(8, 16, 8).transpose(1, 2, 0).reshape(-1)
        result[block, 1024:1032], result[block, 1032:1040] = scale, bias
    return result.reshape(-1)


def pool_levels(f0):
    if f0.dtype != np.uint16 or f0.shape != (400, 8):
        raise ValueError('expected uint16 f0 [400,8]')
    cur = values(f0).reshape(20, 20, 8)
    if not np.isfinite(cur).all():
        raise ValueError('pool oracle requires finite bf16 inputs')
    levels = [f0.copy()]
    for _ in range(3):
        padded = np.pad(cur, ((2, 2), (2, 2), (0, 0)), constant_values=-np.inf)
        # Window reduction has no shared indexing with the device kernel.
        windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5), axis=(0, 1))
        # First maximum in row-major window order matches torch max_pool2d's
        # signed-zero tie behavior; np.max may choose a different zero sign.
        flattened = windows.reshape(20, 20, 8, 25)
        cur = np.take_along_axis(flattened, flattened.argmax(axis=-1)[..., None], axis=-1)[..., 0]
        levels.append(exact_bits(cur).reshape(400, 8))
    return np.stack(levels)


def pool_cases():
    yield 'negative_constant', exact_bits(np.full((400, 8), -3, np.float32))
    border = np.full((20, 20, 8), -8, np.float32)
    border[[0, -1], :, :] = -2
    border[:, [0, -1], :] = -1
    yield 'negative_borders', exact_bits(border).reshape(400, 8)
    for index, (row, col) in enumerate(((0, 0), (0, 19), (19, 0), (19, 19))):
        x = np.full((20, 20, 8), -4, np.float32)
        x[row, col, index] = 1
        yield f'corner_{row}_{col}', exact_bits(x).reshape(400, 8)
    for channel in range(8):
        x = np.broadcast_to(-np.arange(1, 9, dtype=np.float32), (20, 20, 8)).copy()
        x[9, 10, channel] = 4
        yield f'channel_{channel}', exact_bits(x).reshape(400, 8)
    yield 'random', exact_bits(np.random.default_rng(42).integers(-16, 17, (400, 8)).astype(np.float32))
    yield 'zero', np.zeros((400, 8), np.uint16)
    row, col, channel = np.indices((20, 20, 8))
    signed_zero = np.where((row + col + channel) % 2, 0x8000, 0).astype(np.uint16)
    yield 'signed_zero_ties', signed_zero.reshape(400, 8)


class PhaseAShard:
    """Persistent synchronous arenas; runtime or metadata failure poisons use."""
    def __init__(self, build_dir, backend, pool_only=False):
        self.backend, self.pool_only, self.failed = backend, pool_only, False
        name = 'phase_a_pool' if pool_only else 'phase_a_shard'
        self.xclbin, self.insts = [Path(build_dir).resolve() / f'{name}.{ext}' for ext in ('xclbin', 'bin')]
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing phase-A artifacts under {build_dir}')
        self.handle = backend.DefaultNPURuntime.load(backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(3200 if pool_only else INPUT_ELEMENTS, dtype=np.uint16)
        self.weights = None if pool_only else backend.iron.zeros(WEIGHT_ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(OUTPUT_ELEMENTS, dtype=np.uint16)
        self.metadata = backend.iron.zeros(32, dtype=np.uint16)

    def run(self, bits, packed_weights=None):
        if self.failed:
            raise RuntimeError('phase-A invalid after failure; recover and restart process')
        shape = (400, 8 if self.pool_only else 256)
        if bits.dtype != np.uint16 or bits.shape != shape:
            raise ValueError(f'expected uint16 input {shape}')
        if self.pool_only:
            if packed_weights is not None:
                raise ValueError('pool-only does not accept weights')
        elif packed_weights is None or packed_weights.dtype != np.uint16 or packed_weights.shape != (WEIGHT_ELEMENTS,):
            raise ValueError('expected flat uint16 packed weights [2112]')
        try:
            uploads = [(self.input, bits)]
            if not self.pool_only:
                uploads.append((self.weights, packed_weights))
            for buffer, array in uploads:
                buffer.data.reshape(-1)[:] = array.reshape(-1)
                buffer._sync_to_device()
            args = [self.input] + ([] if self.pool_only else [self.weights]) + [self.output, self.metadata]
            result = self.backend.DefaultNPURuntime.run(self.handle, args)
            if not result.is_success():
                raise RuntimeError('phase-A runtime returned unsuccessful result')
            output = self.output.numpy().copy().reshape(4, 400, 8)
            metadata = self.metadata.numpy().copy()
            if metadata.dtype != np.uint16 or metadata.size != 32 or np.any(metadata):
                raise RuntimeError(f'phase-A expected floor rounding and zero reserved metadata, got {metadata}')
            if output.dtype != np.uint16 or not np.isfinite(values(output)).all():
                raise RuntimeError('phase-A output must be finite bf16 bits')
            return output
        except Exception:
            self.failed = True
            raise
