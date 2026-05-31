#!/usr/bin/env python3
"""Bytewise correctness check for ocb_re8_rn3.elf (Phase E v1 prototype).

Tests the OCB-unrolled ELF (n_ocb=4) against n_ocb=1 reference run 4 times,
each with a different OCB's weights. Each slice of the unrolled ELF's
output BO must match the corresponding single-OCB run bytewise.

Run from mdv6 dir:
  cd /home/jfifield/npu-dev-pythoc && source env.sh
  flock /tmp/npu-dev.lock python3 mlir-aie/programming_examples/pythoc/mdv6/conv/test_ocb_re8_rn3.py
"""
import os
import sys
import time
import numpy as np

import pyxrt as xrt

_HERE = os.path.dirname(os.path.abspath(__file__))
_BD = os.path.join(_HERE, "build_merged")
_OCB_ELF = os.path.join(_BD, "ocb_re8_rn3_x1.elf")
_REF_ELF = os.path.join(_BD, "ocb_re8_rn3_ref_x1.elf")

# Shape (matches build_ocb.py _LAYERS["re8_rn3"]):
#   tile 4×4, ic=64, oc_block=16, n_ocb=4, ppc=1
TILE_H = TILE_W = 4
IC = 64
OC_BLOCK = 16
N_OCB = 4
PPC = 1
KS = 3
PADDING = 1
N_CORES = 32

# Derived sizes (must match aie2_multicore_ocb.py)
PATCH_H = (TILE_H - 1) * 1 + KS  # 6
PATCH_W = (TILE_W - 1) * 1 + KS  # 6
PATCH_RAW = PATCH_H * PATCH_W * IC  # 2304
PATCH_SIZE = PATCH_RAW + (PATCH_RAW % 2)  # 2304 (already even)
CORE_INPUT = PPC * PATCH_SIZE
HOST_INPUT = N_CORES * CORE_INPUT  # 32 * 2304 = 73728
WEIGHT_SLOT = OC_BLOCK * IC * KS * KS + 2 * OC_BLOCK  # 9216 + 32 = 9248
OUTPUT_TILE = TILE_H * TILE_W * OC_BLOCK  # 256
CORE_OUTPUT = PPC * OUTPUT_TILE
HOST_OUTPUT = N_CORES * CORE_OUTPUT  # 32 * 256 = 8192


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _bo_read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def main():
    for p in (_OCB_ELF, _REF_ELF):
        if not os.path.exists(p):
            print(f"FAIL: missing {p}")
            return 1

    print(f"shapes: tile={TILE_H}x{TILE_W} ic={IC} oc_block={OC_BLOCK} "
          f"n_ocb={N_OCB} ppc={PPC}")
    print(f"  host_input={HOST_INPUT}u16 host_output={HOST_OUTPUT}u16 "
          f"weight_slot={WEIGHT_SLOT}u16")
    print(f"  ocb_elf: W={N_OCB*WEIGHT_SLOT}u16, O={N_OCB*HOST_OUTPUT}u16")

    device = xrt.device(0)

    # Synthetic input + per-OCB weights. Use mid-range bf16 (~1.0..2.0) to
    # exercise the kernel without NaNs.
    rng = np.random.default_rng(seed=2026)
    in_arr = rng.integers(0x3c00, 0x4000, size=HOST_INPUT, dtype=np.uint16)
    wts = [rng.integers(0x3c00, 0x4000, size=WEIGHT_SLOT, dtype=np.uint16)
           for _ in range(N_OCB)]

    # --- OCB-unrolled ELF: one xrt.run, big W = [wt0, wt1, wt2, wt3] ---
    print(f"\nLoading {os.path.basename(_OCB_ELF)}...")
    ocb = xrt.elf(_OCB_ELF)
    ocb_ctx = xrt.hw_context(device, ocb)
    ocb_kernel = xrt.ext.kernel(ocb_ctx, "main")

    big_W = np.concatenate(wts)
    big_W_bo = xrt.ext.bo(device, big_W.nbytes)
    in_bo = xrt.ext.bo(device, in_arr.nbytes)
    big_O_bo = xrt.ext.bo(device, N_OCB * HOST_OUTPUT * 2)

    _bo_fill(in_bo, in_arr)
    _bo_fill(big_W_bo, big_W)

    # Merged-x1 ELF arg order (share_arg_idxs={1}): arg0=wt, arg1=in, arg2=out
    print("Dispatching OCB-unrolled ELF (one xrt.run, 4 OCBs)...")
    t0 = time.perf_counter()
    r = xrt.run(ocb_kernel)
    r.set_arg(0, big_W_bo)
    r.set_arg(1, in_bo)
    r.set_arg(2, big_O_bo)
    r.start()
    r.wait2()
    ocb_ms = (time.perf_counter() - t0) * 1000
    big_O = _bo_read(big_O_bo, N_OCB * HOST_OUTPUT)
    nz = int(np.count_nonzero(big_O))
    print(f"  ocb wall: {ocb_ms:.2f} ms; nonzero outputs: {nz}/{len(big_O)}")

    # --- Reference: n_ocb=1 ELF run 4 times with different weights ---
    print(f"\nLoading {os.path.basename(_REF_ELF)}...")
    ref = xrt.elf(_REF_ELF)
    ref_ctx = xrt.hw_context(device, ref)
    ref_kernel = xrt.ext.kernel(ref_ctx, "main")

    ref_in_bo = xrt.ext.bo(device, in_arr.nbytes)
    ref_W_bo = xrt.ext.bo(device, WEIGHT_SLOT * 2)
    ref_O_bo = xrt.ext.bo(device, HOST_OUTPUT * 2)
    _bo_fill(ref_in_bo, in_arr)

    refs = []
    total_ref_ms = 0.0
    for ocb_idx, wt in enumerate(wts):
        _bo_fill(ref_W_bo, wt)
        t0 = time.perf_counter()
        r = xrt.run(ref_kernel)
        # Merged-x1: arg0=wt, arg1=in, arg2=out
        r.set_arg(0, ref_W_bo)
        r.set_arg(1, ref_in_bo)
        r.set_arg(2, ref_O_bo)
        r.start()
        r.wait2()
        dt = (time.perf_counter() - t0) * 1000
        total_ref_ms += dt
        ref_out = _bo_read(ref_O_bo, HOST_OUTPUT)
        nz = int(np.count_nonzero(ref_out))
        print(f"  ref OCB{ocb_idx}: {dt:.2f} ms; nonzero: {nz}/{HOST_OUTPUT}")
        refs.append(ref_out)

    # --- Compare slice-by-slice ---
    print("\nBytewise comparison:")
    all_match = True
    for i in range(N_OCB):
        slice_start = i * HOST_OUTPUT
        slice_end = slice_start + HOST_OUTPUT
        ocb_slice = big_O[slice_start:slice_end]
        diff = int(np.sum(ocb_slice != refs[i]))
        print(f"  OCB{i}: diff = {diff}/{HOST_OUTPUT}")
        if diff != 0:
            all_match = False
            mask = ocb_slice != refs[i]
            idxs = np.flatnonzero(mask)[:6]
            for j in idxs:
                print(f"    [{j}]: ocb={ocb_slice[j]:04x} ref={refs[i][j]:04x}")

    print(f"\nWall:")
    print(f"  OCB-unrolled (1 call):      {ocb_ms:.2f} ms")
    print(f"  Reference  ({N_OCB} calls):  {total_ref_ms:.2f} ms")
    print(f"  Speedup factor:             {total_ref_ms/ocb_ms:.2f}x")

    if all_match:
        print("\nPASS — OCB-unrolled output matches per-OCB reference bytewise.")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
