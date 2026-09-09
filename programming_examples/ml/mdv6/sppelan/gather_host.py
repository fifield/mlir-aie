"""Opaque uint16 gather ABI and independent CPU oracles; no NPU imports.

Input [source=4,row=4,level=4,pixel=400,lane=8]. Output duplicates the
pixel-major, level-then-neck-channel concat on all four destination columns:
[destination=4,stripe=25,pixel=16,K=512]. This is a gather-only diagnostic,
not a fused SPPELAN operator. Bit patterns may encode bf16 NaNs intentionally.
"""
from pathlib import Path
import numpy as np
from sppelan.fusion_schedule import gather_segments

INPUT_SHAPE = (4, 4, 4, 400, 8)
OUTPUT_SHAPE = (4, 25, 16, 512)
INPUT_ELEMENTS = 204800
OUTPUT_ELEMENTS = 819200


def check_input(bits):
    if bits.dtype != np.uint16 or bits.shape != INPUT_SHAPE:
        raise ValueError(f'expected uint16 input {INPUT_SHAPE}')


def transpose_oracle(bits):
    """Independent semantic definition: pixel, level, source, row, lane."""
    check_input(bits)
    concat = bits.transpose(3, 2, 0, 1, 4).reshape(25, 16, 512)
    return np.broadcast_to(concat, OUTPUT_SHAPE).copy()


def segment_oracle(bits):
    """Execute authored gather segment offsets, independently of transpose."""
    check_input(bits)
    sources = bits.reshape(4, -1)
    concat = np.empty((25, 16 * 512), np.uint16)
    for stripe in range(25):
        for segment in gather_segments(stripe):
            rows = np.arange(segment['rows'])[:, None]
            lanes = np.arange(segment['width'])[None, :]
            src = segment['source_offset'] + rows * segment['source_stride'] + lanes
            dst = segment['destination_offset'] + rows * segment['destination_stride'] + lanes
            concat[stripe, dst] = sources[segment['source_column'], src]
    return np.broadcast_to(concat.reshape(25, 16, 512), OUTPUT_SHAPE).copy()


def frame_input(frame, seed=42):
    if frame < 0:
        raise ValueError('frame must be nonnegative')
    case = ('index_low16', 'index_high16', 'random', 'zero', 'ones')[frame % 5]
    if case.startswith('index_'):
        indices = np.arange(INPUT_ELEMENTS, dtype=np.uint32)
        bits = (indices if case == 'index_low16' else indices >> 16).astype(np.uint16)
    elif case == 'random':
        bits = np.random.default_rng(seed + frame).integers(0, 65536, INPUT_ELEMENTS, dtype=np.uint16)
    else:
        bits = np.full(INPUT_ELEMENTS, 0 if case == 'zero' else 0xffff, np.uint16)
    return case, bits.reshape(INPUT_SHAPE)


class GatherProbe:
    """One persistent context and two BOs; no retry after any runtime failure."""
    def __init__(self, build_dir, backend):
        root = Path(build_dir).resolve()
        self.xclbin = root / 'gather_probe.xclbin'
        self.insts = root / 'gather_probe.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing gather-probe artifacts under {root}')
        self.backend = backend
        self.handle = backend.DefaultNPURuntime.load(backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(INPUT_ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(OUTPUT_ELEMENTS, dtype=np.uint16)
        self.failed = False

    def run(self, bits):
        if self.failed:
            raise RuntimeError('gather probe invalid after failure; recover and restart process')
        check_input(bits)
        try:
            self.input.data.reshape(-1)[:] = bits.reshape(-1)
            self.input._sync_to_device()
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.output])
            if not result.is_success():
                raise RuntimeError('gather runtime returned unsuccessful result')
            output = self.output.numpy()
            if output.dtype != np.uint16 or output.size != OUTPUT_ELEMENTS:
                raise RuntimeError('gather readback returned invalid dtype/size')
            return output.copy().reshape(OUTPUT_SHAPE)
        except Exception:
            self.failed = True
            raise
