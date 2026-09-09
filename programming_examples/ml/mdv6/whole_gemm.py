"""Opt-in 160x160 ELAN2 Conv4 GEMM, four spatial batches in one submission.

The ABI preserves the baseline's core-major/PPC-major tiling: uint16 bf16
input[4,32,2,104,128], output[4,32,2,104,64], and blocked weights[8320].
Partial slots have zero rows; unused slots repeat their batch's slot zero.
This is command batching, not on-chip fusion or cross-frame weight residency.
"""
from pathlib import Path
import numpy as np

BATCHES, CORES, PPC, TILE_M, IC, OC = 4, 32, 2, 104, 128, 64
PIXELS = 160 * 160
PIXELS_PER_BATCH = CORES * PPC * TILE_M
INPUT_ELEMENTS = BATCHES * PIXELS_PER_BATCH * IC
OUTPUT_ELEMENTS = BATCHES * PIXELS_PER_BATCH * OC
WEIGHT_ELEMENTS = IC * OC + 2 * OC


def pack_input(bits):
    if bits.dtype != np.uint16 or bits.shape != (160, 160, IC):
        raise ValueError('expected uint16 bf16 bits [160,160,128]')
    arena = np.zeros((BATCHES, CORES * PPC, TILE_M, IC), np.uint16)
    flat = bits.reshape(PIXELS, IC)
    for batch in range(BATCHES):
        start = batch * PIXELS_PER_BATCH
        rows = min(PIXELS_PER_BATCH, PIXELS - start)
        arena[batch].reshape(-1, IC)[:rows] = flat[start:start + rows]
        active = (rows + TILE_M - 1) // TILE_M
        arena[batch, active:] = arena[batch, 0]
    return arena.reshape(-1)


def pack_weights(bits):
    if bits.dtype != np.uint16 or bits.shape != (WEIGHT_ELEMENTS,):
        raise ValueError('expected flat uint16 OI weights plus BN scale/bias')
    packed = np.empty(WEIGHT_ELEMENTS, np.uint16)
    packed[:IC * OC] = bits[:IC * OC].reshape(OC // 8, 8, IC // 8, 8).transpose(2, 0, 3, 1).reshape(-1)
    packed[IC * OC:] = bits[IC * OC:]
    return packed


def unpack_output(bits):
    if bits.dtype != np.uint16 or bits.size != OUTPUT_ELEMENTS:
        raise ValueError('output arena has incorrect dtype/size')
    return bits.reshape(-1, OC)[:PIXELS].copy().reshape(160, 160, OC)


class WholeGemm:
    """Synchronous named persistent arenas, invalidated on runtime failure.

    Pack weights per call to avoid unsafe identity-keyed caches. No retries or
    context reloads are attempted after an upload, launch, or readback failure.
    """
    def __init__(self, build_dir, backend):
        self.backend = backend
        root = Path(build_dir).resolve()
        self.xclbin = root / 'whole_gemm.xclbin'
        self.insts = root / 'whole_gemm.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing whole-GEMM artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(
            backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(INPUT_ELEMENTS, dtype=np.uint16)
        self.weights = backend.iron.zeros(WEIGHT_ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(OUTPUT_ELEMENTS, dtype=np.uint16)
        self.failed = False

    def run(self, x, weights):
        if self.failed:
            raise RuntimeError('whole-GEMM invalid after failed run; recover and restart process')
        torch = self.backend.torch
        if x.dtype != torch.bfloat16 or x.device.type != 'cpu':
            raise ValueError('whole-GEMM input must be CPU bf16')
        inp = pack_input(x.contiguous().view(torch.uint16).numpy())
        wt = pack_weights(weights)
        try:
            self.backend._fill_and_sync(self.input, inp)
            self.backend._fill_and_sync(self.weights, wt)
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.weights, self.output])
            if not result.is_success():
                raise RuntimeError('whole-GEMM runtime returned unsuccessful result')
            return torch.from_numpy(unpack_output(self.output.numpy())).view(torch.bfloat16)
        except Exception:
            self.failed = True
            raise
