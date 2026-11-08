"""Weight layout and header of device_cb's B1-split engine (src/graphs/families/engine_cb.py, M36).

fast/b1_l64's engine weights were packed for device_02 ("compact": 8 columns, each column's tiles c, c + 8, ...
contiguous; a tile is K/32 consecutive 32 x 64 objects). device_cb runs the engine on 4 columns, and gives the two
64-row halves of each column (core rows 0-1 and 2-3) different tiles (half_plan). One shim channel per column
streams both halves. Its memtile splits each object [half 0 | half 1], so the host layout is, per column c, per
local slot i, per K object k: half 0's tile i object k, then half 1's. Gate/up halves are padded to 6 pairs by
repeating the half's first pairs, stored in place (96 tiles where the compact layout has 82).

M37: the same layout with cols = 8 is device_s8's (src/graphs/families/engine_s8.py): the B1 split on all 8 columns,
for exact/b1_l64 and the B1/L64 pairs. Each core then runs half of device_02's tiles (gate/up 3 pairs, qkvR Q K V).

The geometry here mirrors engine_cb.half_plan / engine_s8.half_plan; tests/test_engine_cb_layout.py checks it.
"""

import numpy as np

HIDDEN, FFN, PAIRS, NN, KK = 1024, 2624, 41, 64, 32
COLS8, COLS = 8, 4
GATEUP_HALF = {4: 6, 8: 3}             # pairs per core: ceil(11 / 2) on 4 columns, ceil(6 / 2) on 8
KN = {"out": (HIDDEN, HIDDEN), "down": (FFN, HIDDEN), "gateup": (HIDDEN, 2 * FFN), "qkvR": (HIDDEN, 3 * HIDDEN)}


def column_pairs(cols):
    return [list(range(c, PAIRS, cols)) for c in range(cols)]


def compact_slot(op):
    """canonical tile id -> its tile slot in the 8-column compact layout."""
    if op == "gateup":
        pos, start = {}, 0
        for pairs in column_pairs(COLS8):
            for i, p in enumerate(pairs):
                pos[2 * p], pos[2 * p + 1] = start + 2 * i, start + 2 * i + 1
            start += 2 * len(pairs)
        return [pos[t] for t in range(2 * PAIRS)]
    n = KN[op][1] // NN
    T = n // COLS8
    return [(t % COLS8) * T + t // COLS8 for t in range(n)]


def half_tiles(op, c, h, cols=COLS):
    """Canonical B tiles (in stream order) of column c, half h on device_cb (cols 4) or device_s8 (cols 8)."""
    if op == "gateup":
        half = GATEUP_HALF[cols]
        pairs = column_pairs(cols)[c]
        mine = pairs[:half] if h == 0 else pairs[half:]
        mine = mine + mine[: half - len(mine)]
        return [t for p in mine for t in (2 * p, 2 * p + 1)]
    T = KN[op][1] // NN // cols
    if op == "qkvR":                   # per half Q.. K.. V..: RoPE on the local Q and K tiles (2/3 of them)
        g, k = T // 3, T // 6          # tiles per Q/K/V group of the column, per half
        return [c + cols * (q * g + h * k + i) for q in range(3) for i in range(k)]
    return [c + cols * j for j in range(T // 2 * h, T // 2 * (h + 1))]


def interleave(compact, op, cols=COLS):
    """compact (8-column) weight bytes -> device_cb's (cols 4) or device_s8's (cols 8) layout."""
    kt = KN[op][0] // KK
    tiles = np.frombuffer(compact, np.uint8).reshape(-1, kt, KK * NN * 2)
    slot = compact_slot(op)
    parts = []
    for c in range(cols):
        for a, b in zip(half_tiles(op, c, 0, cols), half_tiles(op, c, 1, cols)):
            parts.append(np.stack([tiles[slot[a]], tiles[slot[b]]], axis=1).reshape(-1))
    return np.concatenate(parts).tobytes()


def header_fields(cols=COLS):
    """Engine header records (op, units, ntab, ntile, npass, kt, npre) for out, down, gateup, qkvR, then LN at
    L = 64..512. cols 4 (device_cb): per-core tile counts as on device_02; LN in 16-row units with 8 table objects.
    cols 8 (device_s8): half of device_02's tile counts; LN as on device_02 (32-row units, 16 table objects)."""
    if cols == 8:
        return [(1, 1, 0, 1, 1, 32, 0), (1, 1, 0, 1, 1, 82, 0), (2, 1, 0, 3, 2, 32, 0), (3, 1, 2, 3, 1, 32, 0)] + [
            (5, L // 32, 16, 1, 0, 0, 2) for L in (64, 128, 256, 512)]
    return [(1, 1, 0, 2, 1, 32, 0), (1, 1, 0, 2, 1, 82, 0), (2, 1, 0, 6, 2, 32, 0), (3, 1, 2, 6, 1, 32, 0)] + [
        (5, L // 16, 8, 1, 0, 0, 2) for L in (64, 128, 256, 512)]
