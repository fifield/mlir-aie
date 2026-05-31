#!/usr/bin/env python3
"""Bytewise correctness check for merged_gemm_re6_rn1_pair_x1.elf.

Dispatches the pair ELF with one input + two weights/outputs, then compares
each output against the single-kernel ELF (merged_gemm_t164_ic96_oc48_p1_x1)
run twice with the same weights. They must be bytewise identical — same
input, same weights, same kernel.

Run from the mdv6 dir:
  source env.sh && source venv/bin/activate
  flock /tmp/npu-dev.lock python3 conv/test_pair_rn1.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyxrt as xrt

# Shape: gemm_t164_ic96_oc48_p1
N_CORES = 32
TILE_M = 164
IC = 96
OC = 48
PPC = 1

TOTAL_SLOTS = N_CORES * PPC  # 32
IN_NELEM = TOTAL_SLOTS * TILE_M * IC      # 503808
OUT_NELEM = TOTAL_SLOTS * TILE_M * OC     # 251904
WT_NELEM = IC * OC + 2 * OC               # 4704


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _bo_read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def main():
    bd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_merged")
    pair_elf = os.path.join(bd, "merged_gemm_re6_rn1_pair_x1.elf")
    single_elf = os.path.join(bd, "merged_gemm_t164_ic96_oc48_p1_x1.elf")
    for p in (pair_elf, single_elf):
        if not os.path.exists(p):
            print(f"FAIL: missing {p}")
            return 1

    device = xrt.device(0)

    # Synthetic inputs (mid-range bf16 values to avoid NaNs/INFs).
    rng = np.random.default_rng(seed=42)
    in_arr = rng.integers(0x3c00, 0x4000, size=IN_NELEM, dtype=np.uint16)
    wt_a = rng.integers(0x3c00, 0x4000, size=WT_NELEM, dtype=np.uint16)
    wt_b = rng.integers(0x3c00, 0x4000, size=WT_NELEM, dtype=np.uint16)

    # --- Pair ELF: one xrt.run, two outputs ---
    print(f"Loading {os.path.basename(pair_elf)}...")
    pair = xrt.elf(pair_elf)
    pair_ctx = xrt.hw_context(device, pair)
    pair_kernel = xrt.ext.kernel(pair_ctx, "main")

    pair_in = xrt.ext.bo(device, IN_NELEM * 2)
    pair_wt0 = xrt.ext.bo(device, WT_NELEM * 2)
    pair_out0 = xrt.ext.bo(device, OUT_NELEM * 2)
    pair_wt1 = xrt.ext.bo(device, WT_NELEM * 2)
    pair_out1 = xrt.ext.bo(device, OUT_NELEM * 2)
    _bo_fill(pair_in, in_arr)
    _bo_fill(pair_wt0, wt_a)
    _bo_fill(pair_wt1, wt_b)

    print("Dispatching pair ELF...")
    import time as _t
    t0 = _t.perf_counter()
    run = xrt.run(pair_kernel)
    run.set_arg(0, pair_in)
    run.set_arg(1, pair_wt0)
    run.set_arg(2, pair_out0)
    run.set_arg(3, pair_wt1)
    run.set_arg(4, pair_out1)
    run.start()
    run.wait2()
    pair_dt = (_t.perf_counter() - t0) * 1000
    pair_out0_data = _bo_read(pair_out0, OUT_NELEM)
    pair_out1_data = _bo_read(pair_out1, OUT_NELEM)
    print(f"  pair wall: {pair_dt:.2f} ms; out0 nonzero: "
          f"{int(np.count_nonzero(pair_out0_data))}/{OUT_NELEM}; "
          f"out1 nonzero: {int(np.count_nonzero(pair_out1_data))}/{OUT_NELEM}")

    # --- Single ELF, two sequential calls ---
    print(f"Loading {os.path.basename(single_elf)}...")
    single = xrt.elf(single_elf)
    single_ctx = xrt.hw_context(device, single)
    single_kernel = xrt.ext.kernel(single_ctx, "main")

    # Single ELF uses (wt, in, out) arg order (share_arg_idxs={1} convention).
    single_in = xrt.ext.bo(device, IN_NELEM * 2)
    single_wt = xrt.ext.bo(device, WT_NELEM * 2)
    single_out = xrt.ext.bo(device, OUT_NELEM * 2)
    _bo_fill(single_in, in_arr)

    refs = []
    for wt in (wt_a, wt_b):
        _bo_fill(single_wt, wt)
        t0 = _t.perf_counter()
        r = xrt.run(single_kernel)
        r.set_arg(0, single_wt)
        r.set_arg(1, single_in)
        r.set_arg(2, single_out)
        r.start()
        r.wait2()
        dt = (_t.perf_counter() - t0) * 1000
        refs.append(_bo_read(single_out, OUT_NELEM))
        print(f"  single wall: {dt:.2f} ms")

    # --- Compare ---
    diff0 = int(np.sum(pair_out0_data != refs[0]))
    diff1 = int(np.sum(pair_out1_data != refs[1]))
    print(f"\nbytewise diff: out0 = {diff0}/{OUT_NELEM} ; out1 = {diff1}/{OUT_NELEM}")
    if diff0 == 0 and diff1 == 0:
        print("PASS — pair ELF outputs match standalone-twice baseline.")
        return 0
    # Diagnostic: show first few mismatches.
    if diff0:
        mask = pair_out0_data != refs[0]
        idxs = np.flatnonzero(mask)[:8]
        for i in idxs:
            print(f"  out0 [{i}]: pair={pair_out0_data[i]:04x} ref={refs[0][i]:04x}")
    if diff1:
        mask = pair_out1_data != refs[1]
        idxs = np.flatnonzero(mask)[:8]
        for i in idxs:
            print(f"  out1 [{i}]: pair={pair_out1_data[i]:04x} ref={refs[1][i]:04x}")
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
