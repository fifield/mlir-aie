"""Apply the inherited BFP leaf replacement using only freshly compiled inputs."""

import struct


def sections(raw):
    if raw[:6] != b"\x7fELF\x01\x01":
        raise RuntimeError("requires little-endian ELF32")
    start = struct.unpack_from("<I", raw, 32)[0]
    size, count = struct.unpack_from("<HH", raw, 46)
    if size != 40:
        raise RuntimeError("unexpected section header size")
    return [struct.unpack_from("<10I", raw, start + i * size) for i in range(count)]


def symbol(raw, name):
    headers = sections(raw)
    for table in headers:
        if table[1] != 2:
            continue
        strings = headers[table[6]]
        names = raw[strings[4] : strings[4] + strings[5]]
        for offset in range(table[4], table[4] + table[5], table[9]):
            name_index, value, size, info, other, index = struct.unpack_from(
                "<IIIBBH", raw, offset
            )
            end = names.index(b"\0", name_index)
            if names[name_index:end].decode() == name:
                if not 0 < index < len(headers):
                    raise RuntimeError("undefined leaf")
                section = headers[index]
                if not section[2] & 4:
                    raise RuntimeError("leaf must be executable")
                local = value - section[3]
                if not 0 <= local or local + size > section[5]:
                    raise RuntimeError("leaf out of section bounds")
                return section[4] + local, size, index, local
    raise RuntimeError("leaf symbol missing")


def address(raw):
    _, _, index, start = symbol(raw, "laya_gemm_native")
    return sections(raw)[index][3] + start


def patch(native, object_bytes, leaf_size):
    target, old_size, _, _ = symbol(native, "laya_gemm_native")
    source, new_size, index, start = symbol(object_bytes, "laya_gemm_native")
    if address(native) != address(object_bytes):
        raise RuntimeError("replacement leaf was linked for a different address")
    if new_size != leaf_size or new_size > old_size:
        raise RuntimeError("leaf sizes changed")
    for table in sections(object_bytes):
        if table[1] in (4, 9) and table[7] == index:
            for offset in range(table[4], table[4] + table[5], table[9]):
                relocation = struct.unpack_from("<I", object_bytes, offset)[0]
                if start <= relocation < start + new_size:
                    raise RuntimeError("replacement leaf has unresolved relocations")
    result = bytearray(native)
    result[target : target + new_size] = object_bytes[source : source + new_size]
    # The historical patch retained the old symbol extent and unreachable tail.
    # Preserve every byte outside the new source-compiled function body.
    return bytes(result)
