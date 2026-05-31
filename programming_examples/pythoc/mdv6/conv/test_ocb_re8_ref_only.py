#!/usr/bin/env python3
"""Smoke test the n_ocb=1 reference ELF in isolation.

Quick check: does the new aie2_multicore_ocb.py with n_ocb=1 even run?
If this hangs, the bug is in the new module's structure, not in OCB
unrolling specifically.
"""
import os, sys, time
import numpy as np
import pyxrt as xrt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF_ELF = os.path.join(_HERE, "build_merged", "ocb_re8_rn3_ref_x1.elf")

HOST_INPUT = 73728
WEIGHT_SLOT = 9248
HOST_OUTPUT = 8192


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def main():
    if not os.path.exists(_REF_ELF):
        print(f"missing {_REF_ELF}")
        return 1
    device = xrt.device(0)
    rng = np.random.default_rng(seed=1)
    in_arr = rng.integers(0x3c00, 0x4000, size=HOST_INPUT, dtype=np.uint16)
    wt = rng.integers(0x3c00, 0x4000, size=WEIGHT_SLOT, dtype=np.uint16)

    print(f"Loading {_REF_ELF}...")
    ref = xrt.elf(_REF_ELF)
    ctx = xrt.hw_context(device, ref)
    kernel = xrt.ext.kernel(ctx, "main")

    in_bo = xrt.ext.bo(device, in_arr.nbytes)
    wt_bo = xrt.ext.bo(device, wt.nbytes)
    out_bo = xrt.ext.bo(device, HOST_OUTPUT * 2)
    _bo_fill(in_bo, in_arr)
    _bo_fill(wt_bo, wt)

    print("Dispatching n_ocb=1 ref...")
    # Merged-x1 ELF arg order (share_arg_idxs={1}): arg0=wt, arg1=in, arg2=out
    t0 = time.perf_counter()
    r = xrt.run(kernel)
    r.set_arg(0, wt_bo)
    r.set_arg(1, in_bo)
    r.set_arg(2, out_bo)
    r.start()
    r.wait2()
    dt = (time.perf_counter() - t0) * 1000
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = np.frombuffer(out_bo.map(), dtype=np.uint16, count=HOST_OUTPUT).copy()
    nz = int(np.count_nonzero(out))
    print(f"  wall: {dt:.2f} ms; nonzero outputs: {nz}/{HOST_OUTPUT}")
    if nz == 0:
        print("FAIL: all zeros")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
