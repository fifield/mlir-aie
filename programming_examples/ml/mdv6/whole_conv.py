"""Opt-in whole-operator schedule for re8/re21 bottleneck Conv2 only.

External ABI (uint16 bf16 bits): input[32,10,10,64], weights[4,9248],
output[4,32,8,8,16]. Nine real patches are row-major; remaining cores repeat
patch zero to match the baseline. Outputs crop partial tiles at 20x20.
Host patch construction and final materialization remain explicit. The device
replays the input arena for four OC blocks, in one synchronous submission.
No retained on-chip state across submissions or concurrent use is promised.
"""
from pathlib import Path
import numpy as np

INPUT_ELEMENTS = 32 * 10 * 10 * 64
WEIGHT_SLOT = 16 * 64 * 9 + 32
OUTPUT_ELEMENTS = 4 * 32 * 8 * 8 * 16


def pack_input(bits):
    if bits.shape != (20, 20, 64) or bits.dtype != np.uint16:
        raise ValueError('expected uint16 bf16 bits [20,20,64]')
    padded = np.pad(bits, ((1, 5), (1, 5), (0, 0)))
    patches = np.empty((32, 10, 10, 64), dtype=np.uint16)
    for i in range(9):
        row, col = divmod(i, 3)
        patches[i] = padded[row * 8:row * 8 + 10, col * 8:col * 8 + 10]
    patches[9:] = patches[0]
    return patches.reshape(-1)


def pack_weights(bits):
    if bits.dtype != np.uint16 or bits.shape != (64 * 64 * 9 + 128,):
        raise ValueError('expected full uint16 OIHW weights plus BN scale/bias')
    conv = bits[:64 * 64 * 9].reshape(64, 64, 9)
    scale = bits[64 * 64 * 9:64 * 64 * 9 + 64]
    bias = bits[-64:]
    result = np.empty((4, WEIGHT_SLOT), dtype=np.uint16)
    for block in range(4):
        packed = conv[block * 16:(block + 1) * 16].reshape(2, 8, 8, 8, 9)
        result[block, :-32] = packed.transpose(0, 2, 4, 3, 1).reshape(-1)
        result[block, -32:-16] = scale[block * 16:(block + 1) * 16]
        result[block, -16:] = bias[block * 16:(block + 1) * 16]
    return result.reshape(-1)


def unpack_output(bits):
    if bits.dtype != np.uint16 or bits.size != OUTPUT_ELEMENTS:
        raise ValueError('output arena has incorrect dtype/size')
    tiles = bits.reshape(4, 32, 8, 8, 16)
    result = np.empty((20, 20, 64), dtype=np.uint16)
    for block in range(4):
        for i in range(9):
            row, col = divmod(i, 3)
            h, w = min(8, 20 - row * 8), min(8, 20 - col * 8)
            result[row * 8:row * 8 + h, col * 8:col * 8 + w,
                   block * 16:(block + 1) * 16] = tiles[block, i, :h, :w]
    return result


class WholeConv:
    """Named persistent arenas; failure invalidates this process-local executor.

    Weights are packed on each call: this bounded first version avoids id-keyed
    cache lifetime risks across the twelve distinct model uses. Future island
    ownership should pack/bind each immutable weight tensor once.
    """
    def __init__(self, build_dir, backend):
        self.backend = backend
        root = Path(build_dir).resolve()
        self.xclbin = root / 'whole_conv.xclbin'
        self.insts = root / 'whole_conv.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing whole-conv artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(
            backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(INPUT_ELEMENTS, dtype=np.uint16)
        self.weights = backend.iron.zeros(4 * WEIGHT_SLOT, dtype=np.uint16)
        self.output = backend.iron.zeros(OUTPUT_ELEMENTS, dtype=np.uint16)
        self.failed = False

    def run(self, x, weights):
        if self.failed:
            raise RuntimeError('whole-conv invalid after failed run; recover and restart process')
        torch = self.backend.torch
        if x.dtype != torch.bfloat16 or x.device.type != 'cpu':
            raise ValueError('whole-conv input must be CPU bf16')
        inp = pack_input(x.contiguous().view(torch.uint16).numpy())
        wt = pack_weights(weights)
        try:
            self.backend._fill_and_sync(self.input, inp)
            self.backend._fill_and_sync(self.weights, wt)
            self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.weights, self.output])
            result = unpack_output(self.output.numpy().copy())
            return torch.from_numpy(result).view(torch.bfloat16)
        except Exception:
            self.failed = True
            raise
