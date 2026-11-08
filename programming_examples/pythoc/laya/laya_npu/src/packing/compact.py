"""Generate a compact weight permutation from declared shim descriptor geometry."""

import numpy as np


def word_addresses(p):
    """Word addresses of one push relative to its operand (independent expansion of the shim BD)."""
    b = p["words"]
    length = b[0]
    base = p["base"] // 4
    d0, d1, d2, it = b[3], b[4], b[5], b[6]
    w0, s0 = (d0 >> 20) & 0x3FF, (d0 & 0xFFFFF) + 1
    w1, s1 = (d1 >> 20) & 0x3FF, (d1 & 0xFFFFF) + 1
    s2 = (d2 & 0xFFFFF) + 1
    iw, istep = ((it >> 20) & 0x3F) + 1, (it & 0xFFFFF) + 1
    n = np.arange(length, dtype=np.int64)
    if w0 == 0:
        one = n
    else:
        assert w1 > 0
        one = (n % w0) * s0 + ((n // w0) % w1) * s1 + (n // (w0 * w1)) * s2
    return np.concatenate([base + (r % iw) * istep + one for r in range(p["rep"])])


def mapping(records, nbytes):
    unique = {}
    for record in records:
        original = record["original"]
        wrap = ((original["words"][6] >> 20) & 63) + 1
        current = dict(original, rep=min(original["rep"], wrap))
        key = (current["base"], current["rep"], tuple(current["words"]))
        if key not in unique:
            unique[key] = word_addresses(current).astype(np.uint32)
    extents = []
    for vector in sorted(unique.values(), key=lambda x: -len(x)):
        if not any(
            len(vector) <= len(e) and np.array_equal(vector, e[: len(vector)])
            for e in extents
        ):
            extents.append(vector)
    result = np.concatenate(extents)
    if len(result) * 4 != nbytes or not np.array_equal(
        np.sort(result), np.arange(nbytes // 4, dtype=np.uint32)
    ):
        raise RuntimeError("compact mapping is not a whole-weight bijection")
    return result
