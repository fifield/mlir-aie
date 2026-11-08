"""A source rebuild reproduces every shipped binary exactly.

Expected identities are in tests/expected/. Graph ELFs are compared after
normalizing only the PDI revoke_id wrapper fields (src/packaging/identity.py);
the control streams are part of the ELF.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.build
EXPECTED = Path(__file__).resolve().parent / "expected"
CORES = json.loads((EXPECTED / "cores.json").read_text())
PACKING = json.loads((EXPECTED / "packing.json").read_text())
GRAPHS = json.loads((EXPECTED / "graphs.json").read_text())
_spec = importlib.util.spec_from_file_location(
    "packaging_identity", EXPECTED.parents[1] / "src/packaging/identity.py"
)
_identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_identity)
normalized = _identity.normalized


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("target", sorted(CORES))
def test_core_identity(build_dir, build_record, target):
    expected = CORES[target]
    record = build_record["core_builds"][target]
    cores = record["cores"]
    if isinstance(cores, list):
        cores = {str(i): item for i, item in enumerate(cores)}
    actual = {
        name: sha(build_dir / "cores" / target / item["path"])
        for name, item in cores.items()
    }
    assert actual == expected["cores"]
    if target == "residual":
        assert record["leaf"]["sha256"] == expected["leaf_object"]
        return
    if "native_cores" in expected:
        native = {name: item["sha256"] for name, item in record["native_cores"].items()}
        assert (
            native == expected["native_cores"]
        ), "native cores differ before the dense transform"
    shared = {
        name: h
        for name, h in expected["shared_objects"].items()
        if name in record["objects"]
    }
    assert {name: record["objects"][name] for name in shared} == shared


def test_packed_blob_identity(build_dir, build_record):
    blobs = build_record["packing"]["blobs"]
    assert set(blobs) == set(PACKING)
    mismatched = [
        k
        for k, item in blobs.items()
        if sha(build_dir / "data" / item["path"]) != PACKING[k]
    ]
    assert not mismatched, mismatched[:10]


@pytest.mark.parametrize("target", sorted(GRAPHS))
def test_graph_identity(build_dir, build_record, target):
    built = build_record["graphs"][target]
    plan = json.loads((build_dir / built["plan"]).read_text())
    elf = (build_dir / plan["elf"]).read_bytes()
    assert (
        normalized(elf, require_revoke_zero=True)["sha256"]
        == GRAPHS[target]["elf_normalized_sha256"]
    )
