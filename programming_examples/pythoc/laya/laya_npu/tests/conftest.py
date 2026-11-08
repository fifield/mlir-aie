"""Shared fixtures for the laya_npu tests.

Markers (deselected by default, see pytest.ini):
  build  rebuild every core, blob and graph from source (about 30 minutes), or reuse
         an existing rebuild.py output named by LAYA_NPU_BUILD.
  npu    run on the NPU. Uses the staged runtime named by LAYA_NPU_STAGE, or stages one
         from the build fixture.
"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEVICE_LOCK = Path("/tmp/npu-dev.lock")


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="session")
def build_dir(tmp_path_factory):
    given = os.environ.get("LAYA_NPU_BUILD")
    if given:
        path = Path(given).resolve()
        if not (path / "build.json").is_file():
            pytest.fail(f"LAYA_NPU_BUILD={given} is not a rebuild.py output directory")
        return path
    if not (ROOT / read(ROOT / "manifests/model.json")["path"]).is_file():
        pytest.fail("the raw model checkpoint is absent; run fetch_model.py")
    out = tmp_path_factory.mktemp("laya_npu") / "build"
    subprocess.run(
        [sys.executable, str(ROOT / "rebuild.py"), "--output", str(out)], check=True
    )
    return out


@pytest.fixture(scope="session")
def build_record(build_dir):
    return read(build_dir / "build.json")


@pytest.fixture(scope="session")
def stage_dir(request, tmp_path_factory):
    given = os.environ.get("LAYA_NPU_STAGE")
    if given:
        return Path(given).resolve()
    from stage_research import stage_research
    from stage_runtime import stage

    build = request.getfixturevalue("build_dir")
    out = tmp_path_factory.mktemp("laya_npu") / "stage"
    stage(build, out)
    stage_research(build, out)
    return out


@pytest.fixture(scope="session")
def backend(stage_dir):
    """The staged laya_npu_runtime package, imported once per session."""
    sys.path.insert(0, str(stage_dir))
    import laya_npu_runtime.api
    import laya_npu_runtime.infer

    return laya_npu_runtime


def amdxdna_journal():
    r = subprocess.run(
        ["journalctl", "-k", "--no-pager", "-o", "short-iso"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [line for line in r.stdout.splitlines() if "amdxdna" in line]


@pytest.fixture(scope="session")
def journal():
    """Call the returned function after every NPU call. Any new amdxdna kernel log
    line stops the whole session: the device may be wedged and further hardware
    work must not run until someone looks."""
    baseline = amdxdna_journal()

    def check():
        if amdxdna_journal() != baseline:
            pytest.exit(
                "amdxdna driver logged new kernel messages; stopping all NPU tests",
                returncode=3,
            )

    yield check
    check()


@pytest.fixture(scope="module")
def device_lock():
    """Exclusive NPU ownership for tests that construct Inference(use_lock=False).
    Client tests must not use this: Client takes the same lock itself."""
    with DEVICE_LOCK.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
