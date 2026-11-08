import struct


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


NAMES = {
    "_ZL8m8b_up16": 4096,
    "_ZL12m8b_gelu_out": 1024,
    "_ZL14m8b_cast_stage": 2048,
    "_ZL14m8d_cast_stage": 2048,
    "_ZL14m8d_rope_stage": 256,
    "_ZL12m8d_cast_out": 512,
}


def inspect(raw):
    require(raw[:6] == b"\x7fELF\x01\x01", "ELF32 little endian required")
    phoff, shoff = struct.unpack_from("<II", raw, 28)
    phent, phnum, shent, shnum, shstr = struct.unpack_from("<HHHHH", raw, 42)
    secs = [struct.unpack_from("<10I", raw, shoff + i * shent) for i in range(shnum)]
    names = raw[secs[shstr][4] : secs[shstr][4] + secs[shstr][5]]
    name = lambda blob, i: blob[i : blob.index(b"\0", i)].decode()
    b_index = next(i for i, s in enumerate(secs) if name(names, s[0]) == ".bss")
    b = secs[b_index]
    sym = next(s for s in secs if s[1] == 2)
    st = secs[sym[6]]
    strings = raw[st[4] : st[4] + st[5]]
    symbols = {}
    for off in range(sym[4], sym[4] + sym[5], sym[9]):
        n, v, z, info, other, ndx = struct.unpack_from("<IIIBBH", raw, off)
        n = name(strings, n)
        if n in NAMES:
            symbols[n] = (v, z)
    require({n: z for n, (a, z) in symbols.items()} == NAMES, "scratch symbol mismatch")
    start = b[3]
    end = start + b[5]
    require(end - start == 9984, "scratch extent")
    cursor = start
    for a, z in sorted(symbols.values()):
        require(a == cursor, "scratch padding/overlap")
        cursor += z
    require(cursor == end, "scratch coverage")
    ph = [struct.unpack_from("<8I", raw, phoff + i * phent) for i in range(phnum)]
    matches = [
        i
        for i, s in enumerate(ph)
        if s[0] == 1 and s[2] == start and s[5] == 9984 and s[4] == 0
    ]
    require(len(matches) == 1, "BSS must be its own load segment")
    return (
        {"e_phoff": phoff, "e_phentsize": phent},
        {"sh_addr": b[3], "sh_size": b[5], "sh_entry": shoff + b_index * shent},
        symbols,
        matches[0],
    )
