"""Inherited ownership regression tests plus laya_npu graph and scheduling contracts."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "laya_npu_runtime_source",
    ROOT / "src/runtime/__init__.py",
    submodule_search_locations=[str(ROOT / "src/runtime")],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from laya_npu_runtime_source import runtime
from laya_npu_runtime_source.common import validate_plan
from laya_npu_runtime_source.mixed_contract import validate_mixed_structure
from laya_npu_runtime_source.mixed_schedule import eligibility
from laya_npu_runtime_source.scheduler import pair_eligibility


def test_all_declared_plan_bounds():
    for path in (ROOT / "src/graphs/plans").glob("*.json"):
        plan = json.loads(path.read_text())
        (validate_mixed_structure if plan["batch"] == 3 else validate_plan)(plan)


def test_mixed_admission_preserves_natural_length_and_cache_bound():
    prepared = [
        {"bucket": 64, "valid_lengths": [50, 62]},
        {"bucket": 64, "valid_lengths": [60]},
    ]
    record = {"arena_bytes": 100}
    assert eligibility(prepared, "exact", record, 100)[0]
    assert not eligibility(prepared, "fast", record, 100)[0]
    assert not eligibility(prepared, "exact", record, 99)[0]
    padded = copy.deepcopy(prepared)
    padded[0]["bucket"] = 128
    assert not eligibility(padded, "exact", record, 100)[0]
    assert not eligibility(list(reversed(prepared)), "exact", record, 100)[0]


def test_pair_remains_opt_in():
    prepared = [
        {"bucket": 64, "valid_lengths": [50]},
        {"bucket": 64, "valid_lengths": [60]},
    ]
    assert not pair_eligibility(prepared, False, {})[0]
    assert pair_eligibility(prepared, True, {})[0]


def test_no_historical_imports_in_runtime():
    for path in (ROOT / "src/runtime").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom) and not node.level
                    else []
                )
            )
            assert not any(
                __import__("re").match(r"(m\d|laya_local)", n) for n in names
            ), path


def fake_runtime(states):
    import types
    from laya_npu_runtime_source.runtime import Runtime

    class Run:
        def set_arg(self, *args):
            pass

        def start(self):
            pass

        def state(self):
            return next(states)

    r = Runtime.__new__(Runtime)
    r.x = types.SimpleNamespace(run=lambda kernel: Run())
    r.use_lock = False
    r.closed = r.failed = False
    r.lock_wait_seconds = 0.0
    r.pending = None
    r.retaining_nonterminal = False
    r.kernels = {"test": object()}
    r.bos = [object()]
    r.hosts = [object()]
    r.context = object()
    r.elf = object()
    r.device = object()
    return r


def test_terminal_failure_refuses_reuse():
    r = fake_runtime(iter([5]))
    with pytest.raises(RuntimeError, match="state=5"):
        r.submit(dict(instance="test", deadline=1.5))
    with pytest.raises(RuntimeError, match="failed command"):
        r.submit(dict(instance="test", deadline=1.5))
    r.close()
    r.close()


def test_running_is_not_terminal():
    r = fake_runtime(iter([3, 4]))
    assert r.submit(dict(instance="test", deadline=1.5)) >= 0
    assert not r.failed
    r.close()


def test_timeout_retains_buffers_until_later_terminal(monkeypatch):
    from laya_npu_runtime_source import runtime

    clock = iter([0.0, 0.0, 6.0, 6.0])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(clock))
    r = fake_runtime(iter([999, 3, 4]))
    bo = r.bos[0]
    context = r.context
    observed = []

    def waiting(_):
        observed.append(
            (
                r.pending is not None,
                r.retaining_nonterminal,
                r.bos[0] is bo,
                r.context is context,
            )
        )
        with pytest.raises(RuntimeError, match="nonterminal"):
            r.close()

    monkeypatch.setattr(runtime.time, "sleep", waiting)
    with pytest.raises(RuntimeError, match="state=None"):
        r.submit(dict(instance="test", deadline=1.5))
    assert observed[-1] == (True, True, True, True)
    assert r.pending is None and not r.retaining_nonterminal
    r.close()
    assert not r.bos and r.context is None


def test_corrupt_blob_refused(tmp_path, monkeypatch):
    import zlib
    from laya_npu_runtime_source import runtime

    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / ("a" * 64 + ".z")).write_bytes(zlib.compress(b"changed"))
    monkeypatch.setattr(runtime, "DATA", tmp_path)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        runtime.load_blob("a" * 64)
