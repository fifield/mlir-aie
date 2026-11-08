"""The B1-split engine weight layout (src/packing/engine_cb.py) gives each core half exactly its tiles:
device_cb (4 columns, M36) and device_s8 (8 columns, M37)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "laya_npu_engine_cb", Path(__file__).resolve().parents[1] / "src/packing/engine_cb.py"
)
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)
OBJ = cb.KK * cb.NN * 2  # bytes of one 32 x 64 bf16 object


@pytest.mark.parametrize("cols", [4, 8])
@pytest.mark.parametrize("op", ["qkvR", "out", "gateup", "down"])
def test_interleave_streams_each_half_its_tiles(op, cols):
    K, N = cb.KN[op]
    kt, ntiles = K // cb.KK, (2 * cb.PAIRS if op == "gateup" else N // cb.NN)
    slot = cb.compact_slot(op)
    assert sorted(slot) == list(range(ntiles))
    # compact buffer whose every object holds its (canonical tile, K step) label
    label = np.zeros((ntiles, kt, OBJ // 4), np.uint32)
    for t in range(ntiles):
        label[slot[t], :, 0] = t
        label[slot[t], :, 1] = np.arange(kt)
    out = np.frombuffer(cb.interleave(label.tobytes(), op, cols), np.uint32).reshape(-1, OBJ // 4)
    per_core = len(cb.half_tiles(op, 0, 0, cols))
    assert out.shape[0] == cols * per_core * kt * 2
    stream = out.reshape(cols, per_core, kt, 2, -1)
    real = []
    for c in range(cols):
        for h in range(2):
            got = stream[c, :, :, h]
            assert (got[:, :, 1] == np.arange(kt)).all()
            tiles = list(got[:, 0, 0])
            assert tiles == cb.half_tiles(op, c, h, cols)
            assert (got[:, :, 0] == np.array(tiles)[:, None]).all()
            unique = list(dict.fromkeys(tiles))
            assert tiles[: len(unique)] == unique  # pads only repeat the half's first tiles
            real += unique
    assert sorted(real) == list(range(ntiles))  # every tile on exactly one core half
    if op == "qkvR":  # local order Q.. K.. V..: RoPE on local tiles 3 j < 2 ntile (= device_02's j < 4 on 4 columns)
        k = per_core // 3
        assert all([t // 16 for t in cb.half_tiles(op, c, h, cols)] == [0] * k + [1] * k + [2] * k
                   for c in range(cols) for h in range(2))
    # the header's per-core tile counts (out, down, gateup pairs, qkvR) match the streams
    ntile = dict(zip(["out", "down", "gateup", "qkvR"], [f[3] for f in cb.header_fields(cols)[:4]]))
    assert per_core == ntile[op] * (2 if op == "gateup" else 1)
