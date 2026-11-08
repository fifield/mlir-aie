"""Dedicated laya_npu performance measurements with validation outside timed regions.

Waits for the shared locks. Uses a staged runtime (stage_runtime.py) without modifying it.
Stops if the amdxdna driver logs anything new during the run.
"""

import argparse, copy, fcntl, hashlib, json, math, os, platform, shutil, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def journal():
    r = subprocess.run(
        ["journalctl", "-k", "--no-pager", "-o", "short-iso"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [x for x in r.stdout.splitlines() if "amdxdna" in x]


def hashes(r):
    h = hashlib.sha256()
    for i, n in enumerate(r["prepared"]["valid_lengths"]):
        h.update(r["output"][i, :n].tobytes())
    return {
        "output": h.hexdigest(),
        "raw": {
            k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in r["raw"].items()
        },
        "formatted": hashlib.sha256(
            json.dumps(r["result"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def stats(values):
    x = sorted(values)
    position = 0.95 * (len(x) - 1)
    lo = int(position)
    hi = math.ceil(position)
    return {
        "n": len(x),
        "median": statistics.median(x),
        "min": x[0],
        "max": x[-1],
        "p95": x[lo] + (x[hi] - x[lo]) * (position - lo),
        "stdev": statistics.stdev(x) if len(x) > 1 else 0.0,
    }


def summarize(item):
    warm = [r for r in item["calls"] if r["phase"] == "measured"]
    cold = [r for r in item["calls"] if r["phase"] == "cold"]
    if not warm:
        return
    item["summary"] = {
        "wall_ms": stats([1000 * r["wall_seconds"] for r in warm]),
        "device_submit_ms": stats([1000 * r["device_submit_seconds"] for r in warm]),
        "cold_wall_ms": stats([1000 * r["wall_seconds"] for r in cold]),
        "launches": sorted(set(r["launches"] for r in warm)),
        "units_per_second": item["units"]
        / statistics.median(r["wall_seconds"] for r in warm),
        "ms_per_unit": statistics.median(r["wall_seconds"] for r in warm)
        * 1000
        / item["units"],
        "median_by_block_ms": {
            str(b): statistics.median(
                r["wall_seconds"] for r in warm if r["block"] == b
            )
            * 1000
            for b in sorted({r["block"] for r in warm})
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=Path, default=ROOT / "staged/runtime_01")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--lock", action="append", default=["/tmp/npu-dev.lock"])
    a = p.parse_args()
    if a.repeats < 3:
        raise RuntimeError("at least three samples per block required")
    stage = a.stage.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixtures = ROOT / "inputs/qualification/fixtures.json"
    report = {
        "status": "waiting_for_locks",
        "stage": str(stage),
        "bundle_sha256": sha(stage / "laya_npu_runtime/data/bundle.json"),
        "method": {
            "blocks": 2,
            "order": "forward then reverse",
            "measured_calls_per_block": a.repeats,
            "unmeasured_calls_per_block": 2,
            "cold": "first call after closing graph cache; model load excluded",
            "warmup": "one additional call after cold",
            "timed_region": "Inference.rows, answer_batch, or answer only; hashing, journal checks and report I/O excluded",
            "device_submit": "runtime submit/wait durations; not pure kernel cycles",
            "cpu_threads": 8,
            "memo_entries": 0,
            "cache_entries": 2,
            "cache_bytes": 2 * 1024**3,
            "p95": "linear sample percentile; only 10 samples at default settings",
            "service_unit": "independent request for pair; question for mixed",
            "instrumentation": "harness wraps Runtime.execute to collect returned submission durations; staged files unchanged",
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "regular": {},
        "services": {},
    }

    def save():
        temp = out / "benchmark.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(out / "benchmark.json")

    baseline = None

    def check():
        if journal() != baseline:
            raise RuntimeError("driver journal changed; stop hardware")

    locks = []
    save()
    waiting = time.monotonic()
    try:
        while True:
            try:
                for name in sorted(set(a.lock)):
                    f = open(name, "a")
                    locks.append(f)
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                for f in reversed(locks):
                    f.close()
                locks = []
                time.sleep(2)
        baseline = journal()
        report.update(status="running", lock_wait_seconds=time.monotonic() - waiting)
        save()
        sys.path.insert(0, str(stage))
        from laya_npu_runtime.infer import Inference
        from laya_npu_runtime.runtime import Runtime

        submitted = []
        original = Runtime.execute

        def execute(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            submitted.extend(result[1])
            return result

        Runtime.execute = execute
        data = read(fixtures)
        cases = data["cases"]

        def measure(model, fn, validate, item, block):
            for i in range(a.repeats + 2):
                phase = "cold" if i == 0 else ("warmup" if i == 1 else "measured")
                submitted.clear()
                started = time.perf_counter()
                result = fn()
                elapsed = time.perf_counter() - started
                # Validation and all filesystem/journal operations are outside this timer.
                row = {
                    "block": block,
                    "phase": phase,
                    "iteration": i,
                    "wall_seconds": elapsed,
                    "device_submit_seconds": sum(submitted),
                    "launches": len(submitted),
                }
                if isinstance(result, dict) and "timings" in result:
                    row["internal_timings"] = result["timings"]
                item["calls"].append(row)
                validate(result)
                if model.incidents or model.fallback:
                    raise RuntimeError("runtime incident or fallback")
                check()
                row["validated"] = True
                summarize(item)
                save()

        for block in range(2):
            ordered = cases if block == 0 else list(reversed(cases))
            modes = ["exact", "fast"] if block == 0 else ["fast", "exact"]
            for mode in modes:
                with Inference(
                    use_lock=False, arithmetic=mode, cache_entries=2, memo_entries=0
                ) as model:
                    for case in [
                        c for c in ordered if c["target"].startswith(mode + "/")
                    ]:
                        check()
                        model.cache.close()
                        batch = len(case["rows"])
                        item = report["regular"].setdefault(
                            case["target"],
                            {
                                "units": batch,
                                "unit": "row",
                                "length": case["length"],
                                "calls": [],
                            },
                        )

                        def validate(result):
                            if hashes(result) != case["expected"]:
                                raise RuntimeError("output mismatch: " + case["target"])

                        measure(
                            model,
                            lambda: model.rows(case["rows"], bucket=case["length"]),
                            validate,
                            item,
                            block,
                        )
                        print(
                            "regular",
                            block,
                            case["target"],
                            round(item["summary"]["wall_ms"]["median"], 2),
                            "ms",
                            flush=True,
                        )
            specifications = [
                ("pair", mode, enabled)
                for mode in ["exact", "fast"]
                for enabled in [False, True]
            ] + [("mixed", "exact", enabled) for enabled in [False, True]]
            if block:
                specifications.reverse()
            references = {}
            for kind, mode, enabled in specifications:
                with Inference(
                    use_lock=False,
                    arithmetic=mode,
                    cache_entries=2,
                    memo_entries=0,
                    phase_major=enabled,
                ) as model:
                    key = kind + "/" + mode + "/" + ("shared" if enabled else "serial")
                    item = report["services"].setdefault(
                        key,
                        {
                            "units": 2 if kind == "pair" else 3,
                            "unit": "request" if kind == "pair" else "question",
                            "calls": [],
                        },
                    )
                    if kind == "pair":
                        row = next(c for c in cases if c["target"] == mode + "/b1_l64")[
                            "rows"
                        ][0]
                        states = [row["state"], row["state"].replace("blue", "red")]
                        questions = {row["id"]: row["question"]}

                        def fn():
                            return model.answer_batch(states, questions)

                        def validate(result):
                            value = [r["result"] for r in result]
                            if model.last_schedule["events"][0]["paired"] != enabled:
                                raise RuntimeError("pair route mismatch")
                            identity = ("pair", mode)
                            if identity in references and references[identity] != value:
                                raise RuntimeError("pair output drift")
                            references[identity] = copy.deepcopy(value)

                    else:
                        if not enabled:
                            model.bundle["mixed_requests"] = {}

                        def fn():
                            return model.answer(
                                data["mixed"]["state"], data["mixed"]["questions"]
                            )

                        def validate(result):
                            if (
                                model.last_response["scheduling"]
                                == "phase_major_b2_b1_l64"
                            ) != enabled:
                                raise RuntimeError("mixed route mismatch")
                            identity = ("mixed", mode)
                            value = result["result"]
                            if identity in references and references[identity] != value:
                                raise RuntimeError("mixed output drift")
                            references[identity] = copy.deepcopy(value)

                    measure(model, fn, validate, item, block)
                    print(
                        "service",
                        block,
                        key,
                        round(item["summary"]["wall_ms"]["median"], 2),
                        "ms",
                        flush=True,
                    )
        Runtime.execute = original
        check()
        report.update(status="performance_benchmark_passed", journal_unchanged=True)
        save()
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        try:
            if baseline is not None:
                report["journal_unchanged"] = journal() == baseline
                if not report["journal_unchanged"]:
                    report.update(
                        status="failed", journal_error="driver journal changed"
                    )
            save()
        finally:
            for f in reversed(locks):
                f.close()
    if report["status"] != "performance_benchmark_passed":
        raise RuntimeError("benchmark failed; inspect report")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
