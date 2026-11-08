"""Check replacement bounds and address guards of the dense fast-leaf patch on rebuilt cores."""

import importlib.util
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest

pytestmark = pytest.mark.build
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "leaf_patch", ROOT / "src/dense/leaf_patch.py"
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


LEAF_BYTES = {"dense_fast": 576, "dense_m2n2": 752}


@pytest.fixture(scope="module", params=sorted(LEAF_BYTES))
def dense(build_dir, request):
    output = build_dir / "cores" / request.param
    native = next(output.glob("elfs_*/*.elf")).read_bytes()
    leaf = next(
        p.read_bytes()
        for p in (output / "linked_bfp").glob("*/leaf.elf")
        if patch.address(p.read_bytes()) == patch.address(native)
    )
    return output, native, leaf, LEAF_BYTES[request.param]


def test_only_function_body_changes(dense):
    _, native, leaf, size = dense
    result = patch.patch(native, leaf, size)
    start, _, _, _ = patch.symbol(native, "laya_gemm_native")
    replacement, size, _, _ = patch.symbol(leaf, "laya_gemm_native")
    assert result[:start] == native[:start]
    assert result[start + size :] == native[start + size :]
    assert result[start : start + size] == leaf[replacement : replacement + size]
    assert result != native


def test_wrong_link_address_rejected(dense):
    _, native, leaf, size = dense
    with mock_patch.object(patch, "address", side_effect=[4096, 4112]):
        with pytest.raises(RuntimeError, match="different address"):
            patch.patch(native, leaf, size)


def test_unlinked_object_rejected(dense):
    output, native, _, size = dense
    with pytest.raises(RuntimeError):
        patch.patch(native, (output / "bfp_leaf.o").read_bytes(), size)
