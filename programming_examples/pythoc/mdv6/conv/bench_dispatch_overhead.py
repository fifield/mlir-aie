#!/usr/bin/env python3
"""Isolate per-dispatch overhead: resident hw_context vs alternating contexts.

Measures xrt.run wall time for merged ELFs under three schedules:

  A-only:   one hw_context, run kernel A back to back        -> dispatch floor
  A/A2:     two hw_contexts of the SAME ELF, alternating     -> ctx switch cost
  A/B:      two hw_contexts of DIFFERENT ELFs, alternating   -> ctx + cfg cost

If A/A2 == A/B >> A-only, the cost is hw_context switching (PDI swap) and
collapsing distinct ELFs would not help; if A/B >> A/A2, array reconfig
dominates. Compare A-only with model profile (~4.5 ms NPU/dispatch).

Data is garbage (zeros); we only time the launch. BOs sync once up front.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import pyxrt as xrt

BUILD = Path(__file__).resolve().parent / "build_merged"

# elf -> dispatcher @main arg sizes in uint16 (wt, in, out)
ELFS = {
    "ocb_re8_rn3_x1": (20832, 221184, 98304),
    "ocb_re6_rn3_x1": (36992, 73728, 32768),
    "merged_gemm_t188_ic64_oc64_p2_x1": (4224, 770048, 770048),
    "merged_gemm_t44_ic128_oc128_p1_x1": (16640, 180224, 180224),
    "ocb_aconv7_x1": (295424, 331776, 131072),
    "merged_ftconv0_x1": (2368, 430336, 409600),
    "merged_ftconv0_x8": (2368,) + (430336, 409600) * 8,
    # floor probes built in /tmp/nortp_build
    "/tmp/nortp_build/empty": (8,),
    "/tmp/nortp_build/ocb_re8_rn3_nortp": (20832, 221184, 98304),
}


def load(device, name):
    path = f"{name}.elf" if name.startswith("/") else str(BUILD / f"{name}.elf")
    elf = xrt.elf(path)
    ctx = xrt.hw_context(device, elf)
    kern = xrt.ext.kernel(ctx, "main")
    bos = []
    for n_u16 in ELFS[name]:
        bo = xrt.ext.bo(device, n_u16 * 2)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bos.append(bo)
    return (elf, ctx, kern, bos)


def dispatch(entry):
    _, _, kern, bos = entry
    run = xrt.run(kern)
    for i, bo in enumerate(bos):
        run.set_arg(i, bo)
    run.start()
    run.wait2()


def bench(label, schedule, iters, warmup=10):
    n = len(schedule)
    for i in range(warmup):
        dispatch(schedule[i % n])
    ts = []
    for i in range(iters):
        t0 = time.perf_counter()
        dispatch(schedule[i % n])
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    print(
        f"{label:34s} n={iters:4d}  min={ts[0]:8.3f}  p50={ts[len(ts)//2]:8.3f}"
        f"  p90={ts[int(len(ts)*0.9)]:8.3f}  max={ts[-1]:8.3f}  mean={statistics.mean(ts):8.3f} ms"
    )
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    device = xrt.device(0)
    t0 = time.perf_counter()
    a = load(device, "ocb_re8_rn3_x1")
    t_load_a = (time.perf_counter() - t0) * 1e3
    a2 = load(device, "ocb_re8_rn3_x1")
    b = load(device, "ocb_re6_rn3_x1")
    g = load(device, "merged_gemm_t188_ic64_oc64_p2_x1")
    print(f"hw_context+kernel load (re8_rn3): {t_load_a:.1f} ms\n")

    bench("A-only (re8_rn3 resident)", [a], args.iters)
    bench("A/A2 (same ELF, two ctx)", [a, a2], args.iters)
    bench("A/B (re8_rn3 / re6_rn3)", [a, b], args.iters)
    bench("A/G (re8_rn3 / gemm)", [a, g], args.iters)
    bench("A-only again (recheck)", [a], args.iters)
    bench("G-only (gemm resident)", [g], args.iters)
    bench("B-only (re6_rn3 resident)", [b], args.iters)

    # Resident payload sweep: per-dispatch wall vs total BO bytes.
    print("\npayload sweep (resident, p50 vs MB):")
    for name in ELFS:
        entry = load(device, name)
        mb = sum(ELFS[name]) * 2 / 1e6
        ts = bench(f"  {name}", [entry], max(30, args.iters // 3))
        p50 = ts[len(ts) // 2]
        print(f"    -> {mb:6.2f} MB  {p50:7.3f} ms  {mb / p50 * 1000:7.0f} MB/s")
        del entry


if __name__ == "__main__":
    main()
