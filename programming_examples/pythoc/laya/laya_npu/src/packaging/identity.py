"""Full-ELF identity allowing only validated PDI revoke-ID metadata and its UID.

The inherited bootgen wrapper leaves revoke_id unspecified in many parents.
The builder explicitly requests zero. All other ELF bytes remain compared.
"""

import hashlib, struct


def normalized(raw, require_revoke_zero=False):
    if raw[:6] != b"\x7fELF\x01\x01":
        raise RuntimeError("expected little-endian ELF32")
    start = struct.unpack_from("<I", raw, 32)[0]
    size, count, index = struct.unpack_from("<HHH", raw, 46)
    if size != 40 or start + size * count > len(raw) or index >= count:
        raise RuntimeError("bad ELF sections")
    headers = [struct.unpack_from("<10I", raw, start + i * size) for i in range(count)]
    strings = headers[index]
    names = raw[strings[4] : strings[4] + strings[5]]
    result = bytearray(raw)
    uid = None
    pdis = 0
    allocated = hashlib.md5()
    for h in headers:
        offset, length = h[4:6]
        if h[1] != 8 and offset + length > len(raw):
            raise RuntimeError("section outside ELF")
        body = raw[offset : offset + length]
        name = names[h[0] : names.index(b"\0", h[0])].decode()
        if h[2] & 2:
            allocated.update(body)
        if name.startswith(".pdi."):
            if length < 336:
                raise RuntimeError("short PDI")
            w = struct.unpack_from("<84I", body)
            if not (
                w[6] == 36
                and w[8] == 52
                and w[36] == 52
                and w[60] == 84
                and w[5] == w[7] == w[37] == 1
            ):
                raise RuntimeError("unsupported PDI header layout")
            if any(w[i] for i in (11, 20, 39, 69)):
                raise RuntimeError("encrypted or attributed PDI unsupported")
            if any(
                sum(w[lo:hi]) & 0xFFFFFFFF != 0xFFFFFFFF
                for lo, hi in ((4, 36), (36, 52), (52, 84))
            ):
                raise RuntimeError("PDI checksum mismatch")
            if require_revoke_zero and w[38] != 0:
                raise RuntimeError("generated PDI revoke ID is not zero")
            struct.pack_into("<I", result, offset + 152, 0)
            struct.pack_into("<I", result, offset + 204, (w[51] + w[38]) & 0xFFFFFFFF)
            pdis += 1
        elif name == ".note.xrt.UID":
            if (
                length != 32
                or struct.unpack_from("<III", body) != (4, 16, 4)
                or body[12:16] != b"XRT\0"
                or uid is not None
            ):
                raise RuntimeError("unsupported UID note")
            uid = (offset, body[16:])
            result[offset + 16 : offset + 32] = bytes(16)
    if not pdis or uid is None or uid[1] != allocated.digest():
        raise RuntimeError("full ELF allocated-data UID mismatch")
    return {
        "sha256": hashlib.sha256(result).hexdigest(),
        "pdi_sections": pdis,
        "uid_allocated_md5_verified": True,
        "allowed_metadata": "PDI word38 revoke_id, its word51 checksum, allocated-data MD5 UID only",
    }
