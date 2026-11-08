"""Render measured laya_npu performance without altering raw evidence."""

import argparse, csv, hashlib, json, os, statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    a = p.parse_args()
    d = json.loads(a.report.read_text())
    if d["status"] != "performance_benchmark_passed":
        raise RuntimeError("benchmark must complete successfully")
    rows = []
    for group in ["regular", "services"]:
        for key, v in d[group].items():
            s = v["summary"]
            cold = [r for r in v["calls"] if r["phase"] == "cold"]
            setup = [
                r["internal_timings"]["bucket_setup"] * 1000
                for r in cold
                if "internal_timings" in r
            ]
            rows.append(
                {
                    "kind": group,
                    "configuration": key,
                    "samples": s["wall_ms"]["n"],
                    "median_ms": s["wall_ms"]["median"],
                    "p95_ms": s["wall_ms"]["p95"],
                    "min_ms": s["wall_ms"]["min"],
                    "max_ms": s["wall_ms"]["max"],
                    "device_submit_median_ms": s["device_submit_ms"]["median"],
                    "cold_median_ms": s["cold_wall_ms"]["median"],
                    "units": v["units"],
                    "unit": v["unit"],
                    "units_per_second": s["units_per_second"],
                    "ms_per_unit": s["ms_per_unit"],
                    "launches": ",".join(map(str, s["launches"])),
                    "block0_median_ms": s["median_by_block_ms"]["0"],
                    "block1_median_ms": s["median_by_block_ms"]["1"],
                    "cold_context_setup_median_ms": (
                        statistics.median(setup) if setup else None
                    ),
                    "cpu_preparation_median_ms": (
                        statistics.median(
                            r["internal_timings"]["cpu_preparation"] * 1000
                            for r in v["calls"]
                            if r["phase"] == "measured"
                        )
                        if group == "regular"
                        else None
                    ),
                    "cpu_heads_format_median_ms": (
                        statistics.median(
                            r["internal_timings"]["cpu_heads_calibration_format"] * 1000
                            for r in v["calls"]
                            if r["phase"] == "measured"
                        )
                        if group == "regular"
                        else None
                    ),
                }
            )
    with a.csv.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# laya_npu performance benchmark",
        "",
        f"Status: **{d['status']}**. {d['method']['blocks']} opposite-order blocks, {d['method']['measured_calls_per_block']} measured calls per block, one cold call and one additional warmup per configuration/block. CPU threads: 8; memoization disabled; two-context, 2 GiB cache.",
        "",
        "Wall time covers tokenization/preparation, graph execution, output readback, and CPU heads/formatting. Hash validation, journal checks, and evidence writes occur outside the timer. Model construction is excluded. Device submission covers runtime submit/wait, not pure kernel cycles. Cold context calls include context/buffer initialization; OS caches are not flushed. Pair time is for both independent requests; mixed time is for all three questions. Service timing uses the Inference API, not HTTP or the public Client wrapper.",
        "",
        f"All measurements hold the shared device and compiler locks. Every call validates outputs or serial/shared equivalence; the frozen driver journal is unchanged. Reported throughput is units divided by median resident-call latency, not sustained server throughput. P95 is a descriptive sample percentile from only {2*d['method']['measured_calls_per_block']} measurements.",
        "",
        "## Eligible regular configurations",
        "",
        "| Configuration | Median ms | P95 ms | Range ms | Device submit ms | Cold ms | Rows/s | Launches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["kind"] == "regular":
            lines.append(
                f"| {r['configuration']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['min_ms']:.2f}–{r['max_ms']:.2f} | {r['device_submit_median_ms']:.2f} | {r['cold_median_ms']:.2f} | {r['units_per_second']:.2f} | {r['launches']} |"
            )
    lines += [
        "",
        "## Pair and mixed service comparisons",
        "",
        "| Configuration | Median ms | P95 ms | Device submit ms | Cold ms | ms/unit | Launches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["kind"] == "services":
            lines.append(
                f"| {r['configuration']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['device_submit_median_ms']:.2f} | {r['cold_median_ms']:.2f} | {r['ms_per_unit']:.2f} | {r['launches']} |"
            )
    lines += ["", "## Resident-call speedups", ""]
    for b in [1, 2, 4]:
        exact = d["regular"][f"exact/b{b}_l64"]["summary"]["wall_ms"]["median"]
        fast = d["regular"][f"fast/b{b}_l64"]["summary"]["wall_ms"]["median"]
        lines.append(f"- Fast B{b}/L64 versus exact: **{exact/fast:.3f}×**.")
    for kind, mode in [("pair", "exact"), ("pair", "fast"), ("mixed", "exact")]:
        serial = d["services"][f"{kind}/{mode}/serial"]["summary"]["wall_ms"]["median"]
        shared = d["services"][f"{kind}/{mode}/shared"]["summary"]["wall_ms"]["median"]
        lines.append(
            f"- {kind.capitalize()} {mode}, shared versus serial: **{serial/shared:.3f}×**."
        )
    lines += [
        "",
        f"[Raw evidence]({os.path.relpath(a.report.resolve(),a.markdown.resolve().parent)}) (SHA-256 `{hashlib.sha256(a.report.read_bytes()).hexdigest()}`). The CSV retains both block medians to expose order effects. The 13 rejected fast shapes are outside this eligible-performance matrix.",
        "",
    ]
    with a.markdown.open("x") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
