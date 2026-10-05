#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Validate the initial DDR fill and report median launch time for sweep.sh.

import argparse
import statistics
import sys
import time

import numpy as np
import pyxrt as xrt

SENTINEL = 0xA5
FILL = 0x5A

p = argparse.ArgumentParser()
p.add_argument("--elf", required=True)
p.add_argument("--wb", type=int, required=True)
p.add_argument("--rows", type=int, required=True)
p.add_argument("--L", type=int, required=True)
p.add_argument("--cols", type=int, required=True)
p.add_argument("--rd", type=float, default=0.0)
p.add_argument("--warmups", type=int, default=10)
p.add_argument("--samples", type=int, default=20)
a = p.parse_args()

nbytes = a.wb * a.rows
dev = xrt.device(0)
ctx = xrt.hw_context(dev, xrt.elf(a.elf))
kern = xrt.ext.kernel(ctx, "main:sequence")
bo = xrt.ext.bo(dev, nbytes)
buf = np.frombuffer(bo.map(), dtype=np.uint8, count=nbytes)
buf[:] = SENTINEL
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
# Second argument: source of the optional concurrent reads.
bo_rd = xrt.ext.bo(dev, nbytes)
bo_rd.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def launch():
    t0 = time.perf_counter_ns()
    r = kern(bo, bo_rd)
    r.wait()
    return (time.perf_counter_ns() - t0) / 1e3


launch()
bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
# Every byte of the cols*S strip region is written (the run grid tiles it).
written = buf.reshape(a.rows, a.wb)[:, : a.wb // a.cols * a.cols]
bad = int(np.count_nonzero(written != FILL))
if bad:
    print(f"FAIL: {bad} of {written.size} bytes not written with the fill")
    sys.exit(1)
for _ in range(a.warmups):
    launch()
ts = sorted(launch() for _ in range(a.samples))
print(
    f"RESULT bytes={nbytes} median_us={statistics.median(ts):.1f} best_us={ts[0]:.1f}"
)
print("PASS")
