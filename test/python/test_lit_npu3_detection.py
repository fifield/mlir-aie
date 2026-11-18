# RUN: %pytest %s
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""How LitConfigHelper detects an npu3 (AIE4) device and an AIE4-capable Peano.

xrt-smi, llc and clang are stand-in scripts, so no hardware or Peano is needed.
"""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the stand-in tools are shell scripts"
)


class _Fatal(Exception):
    pass


class _LLVMConfig:
    """The parts of lit.llvm.config.LLVMConfig that detect_xrt uses."""

    def __init__(self):
        self.config = SimpleNamespace(environment={})
        self.lit_config = SimpleNamespace(fatal=self._fatal)

    @staticmethod
    def _fatal(message):
        raise _Fatal(message)

    def with_environment(self, name, value, append_path=False):
        self.config.environment[name] = value


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(_REPO_ROOT / "python"))
    monkeypatch.delenv("AIE_EXPECTED_NPU", raising=False)
    from aie_lit_utils.lit_config_helpers import LitConfigHelper

    return LitConfigHelper


def _tool(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _xrt_bin(tmp_path: Path, device_name: str) -> Path:
    bin_dir = tmp_path / "xrt_bin"
    bin_dir.mkdir()
    _tool(
        bin_dir,
        "xrt-smi",
        "cat <<'EOF'\n"
        "Device(s) Present\n"
        "|BDF             |Name          |Architecture  |Topology  |\n"
        "|----------------|--------------|--------------|----------|\n"
        f"|[0000:83:00.1]  |{device_name}  |aie4          |6x3       |\n"
        "EOF",
    )
    return bin_dir


def _detect(helper, bin_dir, **kwargs):
    return helper.detect_xrt(
        "xrt_lib",
        "xrt_include",
        str(bin_dir),
        str(_REPO_ROOT),
        _LLVMConfig(),
        [],
        **kwargs,
    )


@pytest.mark.parametrize("name", ["RyzenAI-npu9", "RyzenAI-npu3"])
def test_npu3_with_aie4_peano(helper, tmp_path, name):
    config = _detect(helper, _xrt_bin(tmp_path, name), can_use_peano_aie4=True)
    # Only the npu3 feature: the generic ryzen_ai tests target AIE2 and AIE2P.
    assert config.features == ["ryzen_ai_npu3"]
    run_on_npu3 = config.substitutions["%run_on_npu3%"]
    assert "run_on_npu.py" in run_on_npu3 and run_on_npu3.endswith('"npu3"')
    assert config.substitutions["%run_on_npu1%"] == "echo"
    assert config.substitutions["%run_on_npu2%"] == "echo"
    assert config.substitutions["%aie_cmake_device%"] == "npu"


def test_npu3_without_aie4_backend(helper, tmp_path):
    # An AIE2-only Peano (the generic gate) does not enable npu3 tests.
    config = _detect(
        helper, _xrt_bin(tmp_path, "RyzenAI-npu9"), can_use_peano_feature_gate=True
    )
    assert config.features == []
    assert config.substitutions["%run_on_npu3%"] == "echo"


def test_npu3_with_aie4_vitis_component(helper, tmp_path):
    bin_dir = _xrt_bin(tmp_path, "RyzenAI-npu9")
    config = helper.detect_xrt(
        "xrt_lib", "xrt_include", str(bin_dir), str(_REPO_ROOT), _LLVMConfig(), ["AIE4"]
    )
    assert config.features == ["ryzen_ai_npu3"]


def test_npu2_is_unchanged(helper, tmp_path):
    config = _detect(
        helper,
        _xrt_bin(tmp_path, "RyzenAI-npu4"),
        can_use_peano_feature_gate=True,
        can_use_peano_aie4=True,
    )
    assert config.features == ["ryzen_ai", "ryzen_ai_npu2"]
    assert config.substitutions["%run_on_npu3%"] == "echo"
    assert config.substitutions["%aie_cmake_device%"] == "npu2"


def test_no_xrt_still_substitutes_npu3(helper):
    config = helper.detect_xrt("", "", "", str(_REPO_ROOT), _LLVMConfig())
    assert config.substitutions["%run_on_npu3%"] == "echo"


def test_expected_npu3(helper, tmp_path, monkeypatch):
    monkeypatch.setenv("AIE_EXPECTED_NPU", "npu3")
    bin_dir = _xrt_bin(tmp_path, "RyzenAI-npu9")
    config = _detect(helper, bin_dir, can_use_peano_aie4=True)
    assert config.features == ["ryzen_ai_npu3"]
    with pytest.raises(_Fatal, match="AIE_EXPECTED_NPU=npu3"):
        _detect(helper, bin_dir)


def _peano(tmp_path, llc_rc, clang_rc) -> Path:
    bin_dir = tmp_path / "peano_bin"
    bin_dir.mkdir()
    # Each stand-in accepts only the AIE4 triple, as a Peano with AIE4 does.
    _tool(
        bin_dir,
        "llc",
        'case "$*" in *-mtriple=aie4*) cat >/dev/null; exit '
        f"{llc_rc};; esac\nexit 2",
    )
    _tool(
        bin_dir,
        "clang",
        'case "$*" in *--target=aie4-none-unknown-elf*) cat >/dev/null; exit '
        f"{clang_rc};; esac\nexit 2",
    )
    return bin_dir


@pytest.mark.parametrize(
    "llc_rc, clang_rc, expected",
    [(0, 0, True), (1, 0, False), (0, 1, False)],
)
def test_peano_supports_aie4(helper, tmp_path, llc_rc, clang_rc, expected):
    bin_dir = _peano(tmp_path, llc_rc, clang_rc)
    assert helper.peano_supports_aie4(str(bin_dir)) is expected


def test_peano_supports_aie4_without_peano(helper, tmp_path):
    assert helper.peano_supports_aie4(str(tmp_path / "missing")) is False
