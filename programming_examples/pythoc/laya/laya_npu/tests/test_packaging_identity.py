"""Reject payload/checksum changes while allowing only the documented wrapper fields."""

import hashlib, importlib.util, json, struct
from pathlib import Path
import pytest

pytestmark = pytest.mark.build
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "packaging_identity", ROOT / "src/packaging/identity.py"
)
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


@pytest.fixture(scope="module")
def elf(build_dir, build_record):
    plan = json.loads(
        (build_dir / build_record["graphs"]["exact/b1_l64"]["plan"]).read_text()
    )
    return (build_dir / plan["elf"]).read_bytes()


def sections(raw):
    start = struct.unpack_from("<I", raw, 32)[0]
    size, count, index = struct.unpack_from("<HHH", raw, 46)
    headers = [struct.unpack_from("<10I", raw, start + i * size) for i in range(count)]
    string = headers[index]
    names = raw[string[4] : string[4] + string[5]]
    return [(names[h[0] : names.index(b"\0", h[0])].decode(), h) for h in headers]


def repair_uid(raw):
    digest = hashlib.md5()
    uid = None
    for name, h in sections(raw):
        if h[2] & 2:
            digest.update(raw[h[4] : h[4] + h[5]])
        if name == ".note.xrt.UID":
            uid = h[4] + 16
    raw[uid : uid + 16] = digest.digest()


def test_revoke_id_metadata_is_the_only_normalized_payload(elf):
    changed = bytearray(elf)
    _, h = next((n, h) for n, h in sections(elf) if n.startswith(".pdi."))
    offset = h[4]
    old = struct.unpack_from("<I", changed, offset + 152)[0]
    checksum = struct.unpack_from("<I", changed, offset + 204)[0]
    struct.pack_into("<I", changed, offset + 152, (old + 17) & 0xFFFFFFFF)
    struct.pack_into("<I", changed, offset + 204, (checksum - 17) & 0xFFFFFFFF)
    repair_uid(changed)
    assert identity.normalized(changed)["sha256"] == identity.normalized(elf)["sha256"]
    with pytest.raises(RuntimeError, match="not zero"):
        identity.normalized(changed, require_revoke_zero=True)


def test_bad_pdi_checksum_rejected(elf):
    changed = bytearray(elf)
    _, h = next((n, h) for n, h in sections(elf) if n.startswith(".pdi."))
    changed[h[4] + 204] ^= 1
    with pytest.raises(RuntimeError, match="checksum"):
        identity.normalized(changed)


def test_control_payload_change_never_normalizes_away(elf):
    changed = bytearray(elf)
    _, h = next((n, h) for n, h in sections(elf) if n.startswith(".ctrltext."))
    changed[h[4] + 24] ^= 1
    repair_uid(changed)
    assert identity.normalized(changed)["sha256"] != identity.normalized(elf)["sha256"]
