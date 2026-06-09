#!/usr/bin/env python3
"""Step-1 dispatch overhead suite: floor probes + payload sweep + ctx switch."""
import statistics
import time
from pathlib import Path

import pyxrt as xrt

BUILD = Path(__file__).resolve().parent / "build_merged"

ELFS = {
    "empty (no DMA, no cores)": ("/tmp/nortp_build/empty.elf", (8,)),
    "re8_rn3 (no RTP writes)": ("/tmp/nortp_build/ocb_re8_rn3_nortp.elf", (20832, 221184, 98304)),
    "re6_rn3": (str(BUILD / "ocb_re6_rn3_x1.elf"), (36992, 73728, 32768)),
    "re8_rn3": (str(BUILD / "ocb_re8_rn3_x1.elf"), (20832, 221184, 98304)),
    "gemm_t44_ic128_oc128": (str(BUILD / "merged_gemm_t44_ic128_oc128_p1_x1.elf"), (16640, 180224, 180224)),
    "aconv7_ocb": (str(BUILD / "ocb_aconv7_x1.elf"), (295424, 331776, 131072)),
    "ftconv0_x1": (str(BUILD / "merged_ftconv0_x1.elf"), (2368, 430336, 409600)),
    "gemm_t188_ic64_oc64": (str(BUILD / "merged_gemm_t188_ic64_oc64_p2_x1.elf"), (4224, 770048, 770048)),
    "ftconv0_x8": (str(BUILD / "merged_ftconv0_x8.elf"), (2368,) + (430336, 409600) * 8),
}

device = xrt.device(0)


def load(name):
    path, sizes = ELFS[name]
    elf = xrt.elf(path)
    ctx = xrt.hw_context(device, elf)
    kern = xrt.ext.kernel(ctx, "main")
    bos = []
    for n in sizes:
        bo = xrt.ext.bo(device, n * 2)
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


def bench(label, schedule, iters=100, warmup=10):
    try:
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
            f"{label:34s} min={ts[0]:8.3f}  p50={ts[len(ts)//2]:8.3f}"
            f"  p90={ts[int(len(ts)*0.9)]:8.3f}  mean={statistics.mean(ts):8.3f} ms",
            flush=True,
        )
        return ts[len(ts) // 2]
    except Exception as e:
        print(f"{label:34s} FAILED: {e}", flush=True)
        return None


print("== resident payload sweep ==", flush=True)
cache = {}
for name in ELFS:
    try:
        cache[name] = load(name)
    except Exception as e:
        print(f"{name}: load failed: {e}", flush=True)
        continue
    mb = sum(ELFS[name][1]) * 2 / 1e6
    p50 = bench(name, [cache[name]])
    if p50:
        print(f"    payload {mb:6.2f} MB -> {mb/p50*1000:7.0f} MB/s effective", flush=True)

print("\n== context switching ==", flush=True)
a = cache.get("re8_rn3")
if a:
    a2 = load("re8_rn3")
    for other in ("re6_rn3", "gemm_t188_ic64_oc64", "ftconv0_x1"):
        if other in cache:
            bench(f"alt re8_rn3 / {other}", [a, cache[other]])
    bench("alt re8_rn3 / re8_rn3 (2 ctx)", [a, a2])
    bench("re8_rn3 resident recheck", [a])
