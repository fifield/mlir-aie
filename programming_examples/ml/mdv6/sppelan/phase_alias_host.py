"""Opaque finite A→25B→A phase sentinel; no host phase/stripe operations."""
from pathlib import Path
import numpy as np

A_ELEMENTS = 12800
B_ELEMENTS = 8192
STRIPES = 25
ELEMENTS = A_ELEMENTS + STRIPES * B_ELEMENTS
SHAPE = (ELEMENTS,)
FRAME_BYTES = ELEMENTS * 2


def check_input(bits):
    if bits.dtype != np.uint16 or bits.shape != SHAPE:
        raise ValueError(f'expected native uint16 phase-alias input {SHAPE}')


def oracle(bits):
    """Decode each flat output address independently of device descriptors."""
    check_input(bits)
    address = np.arange(ELEMENTS)
    stripe = (address - A_ELEMENTS) // B_ELEMENTS
    tags = np.where(address < A_ELEMENTS, 0xA5A5, 0x5A00 | (stripe + 1))
    return bits ^ tags.astype(np.uint16)


def frame_input(frame, seed=42):
    if frame < 0:
        raise ValueError('frame must be nonnegative')
    case = ('index_low16', 'index_high16', 'random', 'zero', 'ones', 'edge_bits')[frame % 6]
    if case.startswith('index_'):
        identities = np.arange(ELEMENTS, dtype=np.uint32)
        bits = (identities if case == 'index_low16' else identities >> 16).astype(np.uint16)
    elif case == 'random':
        bits = np.random.default_rng(seed + frame).integers(0, 65536, ELEMENTS, dtype=np.uint16)
    elif case == 'edge_bits':
        words = np.array([0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xffc1, 1, 0xffff], np.uint16)
        bits = np.roll(np.resize(words, ELEMENTS), frame // 6)
    else:
        bits = np.full(ELEMENTS, 0 if case == 'zero' else 0xffff, np.uint16)
    return case, bits


def validate_output(observed, expected):
    if observed.dtype != np.uint16 or observed.shape != SHAPE:
        raise RuntimeError('phase-alias output has wrong dtype/shape')
    if not np.array_equal(observed, expected):
        address = int(np.flatnonzero(observed != expected)[0])
        phase = 'A' if address < A_ELEMENTS else 'B'
        stripe = None if phase == 'A' else (address - A_ELEMENTS) // B_ELEMENTS
        lane = address if phase == 'A' else (address - A_ELEMENTS) % B_ELEMENTS
        raise RuntimeError(f'phase-alias mismatch at address={address}, phase={phase}, '
                           f'stripe={stripe}, lane={lane}: got {int(observed[address])}, '
                           f'expected {int(expected[address])}')


class PhaseAlias:
    """One persistent context and two BOs; any runtime failure poisons the object."""
    def __init__(self, build_dir, backend):
        self.backend, self.failed = backend, False
        root = Path(build_dir).resolve()
        self.xclbin, self.insts = root / 'phase_alias.xclbin', root / 'phase_alias.bin'
        if not self.xclbin.is_file() or not self.insts.is_file():
            raise FileNotFoundError(f'missing phase-alias artifacts under {root}')
        self.handle = backend.DefaultNPURuntime.load(backend.NPUKernel(str(self.xclbin), str(self.insts)))
        self.input = backend.iron.zeros(ELEMENTS, dtype=np.uint16)
        self.output = backend.iron.zeros(ELEMENTS, dtype=np.uint16)

    def run(self, bits):
        if self.failed:
            raise RuntimeError('phase-alias invalid after failure; recover and restart process')
        check_input(bits)
        try:
            self.input.data.reshape(-1)[:] = bits
            self.input._sync_to_device()
            result = self.backend.DefaultNPURuntime.run(self.handle, [self.input, self.output])
            if not result.is_success():
                raise RuntimeError('phase-alias runtime returned unsuccessful result')
            observed = self.output.numpy()
            if observed.dtype != np.uint16 or observed.size != ELEMENTS:
                raise RuntimeError('phase-alias readback has wrong dtype/size')
            return observed.copy().reshape(SHAPE)
        except Exception:
            self.failed = True
            raise
