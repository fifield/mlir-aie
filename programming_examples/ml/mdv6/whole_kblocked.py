"""Three 80x80 IC256/OC128 K-blocked GEMM batches in one submission.

Input ABI [3,32,1,68,256], output [3,32,1,68,128], both uint16 bf16.
Weights contain 16 independently blocked K16 chunks with repeated BN fields.
The final batch has 30 full slots, eight valid rows in slot 30, and slot 31
repeats slot zero. This is command batching, not inter-operator fusion.
"""
from pathlib import Path
import numpy as np

BATCHES, CORES, PPC, TILE_M, IC, OC, K_BLOCK = 3, 32, 1, 68, 256, 128, 16
PIXELS = 80 * 80
PIXELS_PER_BATCH = CORES * PPC * TILE_M
INPUT_ELEMENTS = BATCHES * PIXELS_PER_BATCH * IC
OUTPUT_ELEMENTS = BATCHES * PIXELS_PER_BATCH * OC
RAW_WEIGHT_ELEMENTS = IC * OC + 2 * OC
CHUNK_ELEMENTS = K_BLOCK * OC + 2 * OC
WEIGHT_ELEMENTS = (IC // K_BLOCK) * CHUNK_ELEMENTS


def pack_input(bits):
    if bits.dtype != np.uint16 or bits.shape != (80, 80, IC):
        raise ValueError('expected uint16 bf16 bits [80,80,256]')
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
    if bits.dtype != np.uint16 or bits.shape != (RAW_WEIGHT_ELEMENTS,):
        raise ValueError('expected flat uint16 OI weights plus BN scale/bias')
    packed = np.empty((IC // K_BLOCK, CHUNK_ELEMENTS), np.uint16)
    matrix = bits[:IC * OC].reshape(OC, IC)
    for block in range(IC // K_BLOCK):
        tile = matrix[:, block * K_BLOCK:(block + 1) * K_BLOCK]
        packed[block, :K_BLOCK * OC] = tile.reshape(OC // 8, 8, K_BLOCK // 8, 8).transpose(2, 0, 3, 1).reshape(-1)
        packed[block, K_BLOCK * OC:] = bits[IC * OC:]
    return packed.reshape(-1)


def unpack_output(bits):
    if bits.dtype != np.uint16 or bits.size != OUTPUT_ELEMENTS:
        raise ValueError('output arena has incorrect dtype/size')
    return bits.reshape(-1, OC)[:PIXELS].copy().reshape(80, 80, OC)


class WholeKBlocked:
    """Persistent synchronous arenas; runtime failures invalidate without retry.

    Weights are repacked on every call, including when the caller mutates the
    same array. All sixteen BN copies are refreshed along with convolution data.
    """
    def __init__(self, build_dir, backend):
        self.backend = backend
        root = Path(build_dir).resolve()
        self.xclbin = root / 'whole_kblocked.xclbin'
        self.insts = root / 'whole_kblocked.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing whole-K-blocked artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(
            backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(INPUT_ELEMENTS, dtype=np.uint16)
        self.weights = backend.iron.zeros(WEIGHT_ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(OUTPUT_ELEMENTS, dtype=np.uint16)
        self.failed = False

    def run(self, x, weights):
        if self.failed:
            raise RuntimeError('whole-K-blocked invalid after failed run; recover and restart process')
        torch = self.backend.torch
        if x.dtype != torch.bfloat16 or x.device.type != 'cpu':
            raise ValueError('whole-K-blocked input must be CPU bf16')
        inp = pack_input(x.contiguous().view(torch.uint16).numpy())
        wt = pack_weights(weights)
        try:
            self.backend._fill_and_sync(self.input, inp)
            self.backend._fill_and_sync(self.weights, wt)
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.weights, self.output])
            if not result.is_success():
                raise RuntimeError('whole-K-blocked runtime returned unsuccessful result')
            return torch.from_numpy(unpack_output(self.output.numpy())).view(torch.bfloat16)
        except Exception:
            self.failed = True
            raise
