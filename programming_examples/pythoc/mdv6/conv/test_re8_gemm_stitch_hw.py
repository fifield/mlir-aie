#!/usr/bin/env python3
"""B2a HW test — re8-shape device-resident GEMM->GEMM dispatcher merge.

Proves the CONTEXT-NEGATIVE dispatcher merge on a real re8 GEMM shape (rnm:
1x1 conv IC=128 -> OC=128, tile_m=44, ppc=1, 32 cores):

  1. live hw_context count DROPS 2 -> 1 (two single-sub ELFs collapse into one
     2-sub-device merged ELF).
  2. host dispatches/op-pair DROP 2 -> 1 (the on-device PDI swap replaces the
     2nd host dispatch + the host bounce of the intermediate).
  3. BIT-EXACT vs the two ops run as two separate host dispatches with the
     intermediate bounced through host DDR.

Run from the mdv6 dir:
  source env.sh
  flock /tmp/npu-dev.lock python3 conv/test_re8_gemm_stitch_hw.py
"""
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyxrt as xrt
from build_re8_gemm_stitch import (
    build_all, _resolve_build_dir,
    STITCH_ELF, PROD_ELF, CONS_ELF,
    N_CORES, TILE_M, IC, OC, PPC,
)

# Element counts (u16) for the rnm GEMM shape.
N_SLOTS = N_CORES * PPC                 # 32
PROD_IN_N = N_SLOTS * TILE_M * IC       # 32*44*128 = 180224
INTER_N = N_SLOTS * TILE_M * OC         # producer out == consumer in = 180224
# Weight envelope from aie2_gemm_conv1x1: IC*OC + 2*OC (conv + BN scale/bias).
WT_N = IC * OC + 2 * OC                 # 16640
CONS_OUT_N = N_SLOTS * TILE_M * OC      # 180224

WARMUP = 5
ITERS = 50


def _elf_path(name):
    return os.path.join(_resolve_build_dir(), f"{name}.elf")


def _fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def _bf16_pos(rng, n):
    """Random small positive bf16 bit-patterns (avoid denormals/NaN/Inf)."""
    return rng.integers(0x3c00, 0x4000, size=n, dtype=np.uint16)


def main():
    if not build_all():
        print("FAIL: build failed")
        return 1

    device = xrt.device(0)
    rng = np.random.default_rng(seed=4242)
    in_arr = _bf16_pos(rng, PROD_IN_N)
    wt_prod = _bf16_pos(rng, WT_N)
    wt_cons = _bf16_pos(rng, WT_N)

    # ------------------------------------------------------------------
    # BEFORE — two separate single-sub ELFs => 2 hw_context, 2 host dispatches.
    # Producer writes the intermediate; host reads it back and pushes it into
    # the consumer's input BO; consumer runs. This is the path the merge
    # eliminates.
    # ------------------------------------------------------------------
    prod = xrt.elf(_elf_path(PROD_ELF))
    prod_ctx = xrt.hw_context(device, prod)          # hw_context #1
    prod_k = xrt.ext.kernel(prod_ctx, "main")
    p_in = xrt.ext.bo(device, PROD_IN_N * 2)
    p_wt = xrt.ext.bo(device, WT_N * 2)
    p_out = xrt.ext.bo(device, INTER_N * 2)
    _fill(p_in, in_arr)
    _fill(p_wt, wt_prod)

    cons = xrt.elf(_elf_path(CONS_ELF))
    cons_ctx = xrt.hw_context(device, cons)          # hw_context #2
    cons_k = xrt.ext.kernel(cons_ctx, "main")
    c_in = xrt.ext.bo(device, INTER_N * 2)
    c_wt = xrt.ext.bo(device, WT_N * 2)
    c_out = xrt.ext.bo(device, CONS_OUT_N * 2)
    _fill(c_wt, wt_cons)

    baseline_contexts = 2  # one per ELF

    def run_baseline():
        # dispatch 1: producer  (in,wt,out)
        r = xrt.run(prod_k)
        r.set_arg(0, p_in); r.set_arg(1, p_wt); r.set_arg(2, p_out)
        r.start(); r.wait2()
        # host bounce of the intermediate
        inter = _read(p_out, INTER_N)
        _fill(c_in, inter)
        # dispatch 2: consumer  (in,wt,out)
        r = xrt.run(cons_k)
        r.set_arg(0, c_in); r.set_arg(1, c_wt); r.set_arg(2, c_out)
        r.start(); r.wait2()
        return _read(c_out, CONS_OUT_N)

    ref_out = run_baseline()
    baseline_dispatches = 2

    # ------------------------------------------------------------------
    # AFTER — one merged ELF, ONE hw_context, producer->consumer chain_link.
    # @main(in0, wt0, INTER, wt1, out1). Host fills in0 + both weights, runs
    # ONCE (producer config -> on-device PDI swap -> consumer config), reads
    # only out1. The INTER BO is allocated but NEVER filled/read by host
    # between ops — it is device-resident across the swap.
    # ------------------------------------------------------------------
    stitch = xrt.elf(_elf_path(STITCH_ELF))
    stitch_ctx = xrt.hw_context(device, stitch)      # ONE hw_context
    stitch_k = xrt.ext.kernel(stitch_ctx, "main")
    merged_contexts = 1

    s_in = xrt.ext.bo(device, PROD_IN_N * 2)
    s_wt0 = xrt.ext.bo(device, WT_N * 2)
    s_inter = xrt.ext.bo(device, INTER_N * 2)        # device-resident; host never touches it
    s_wt1 = xrt.ext.bo(device, WT_N * 2)
    s_out = xrt.ext.bo(device, CONS_OUT_N * 2)
    _fill(s_in, in_arr)
    _fill(s_wt0, wt_prod)
    _fill(s_wt1, wt_cons)
    # Poison the intermediate and DO NOT sync it after the producer runs. If the
    # consumer silently read host data instead of the on-device producer output,
    # the result would differ.
    _fill(s_inter, np.full(INTER_N, 0xdead, dtype=np.uint16))

    s_run = xrt.run(stitch_k)
    s_run.set_arg(0, s_in)
    s_run.set_arg(1, s_wt0)
    s_run.set_arg(2, s_inter)
    s_run.set_arg(3, s_wt1)
    s_run.set_arg(4, s_out)

    def run_stitched():
        s_run.start(); s_run.wait2()
        return _read(s_out, CONS_OUT_N)

    stitch_out = run_stitched()
    merged_dispatches = 1

    # ------------------------------------------------------------------
    max_diff = int(np.max(np.abs(stitch_out.astype(np.int32)
                                 - ref_out.astype(np.int32))))
    ndiff = int(np.sum(stitch_out != ref_out))
    exact = (ndiff == 0)
    one_context = isinstance(stitch_ctx, xrt.hw_context)

    # ------------------------------------------------------------------
    def time_fn(fn, warmup, iters):
        for _ in range(warmup):
            fn()
        s = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            s.append((time.perf_counter() - t0) * 1e6)  # us
        return s

    def run_2dispatch_nobounce():
        r = xrt.run(prod_k)
        r.set_arg(0, p_in); r.set_arg(1, p_wt); r.set_arg(2, p_out)
        r.start(); r.wait2()
        r = xrt.run(cons_k)
        r.set_arg(0, c_in); r.set_arg(1, c_wt); r.set_arg(2, c_out)
        r.start(); r.wait2()

    def run_1dispatch():
        r = xrt.run(prod_k)
        r.set_arg(0, p_in); r.set_arg(1, p_wt); r.set_arg(2, p_out)
        r.start(); r.wait2()

    st_stitch = time_fn(run_stitched, WARMUP, ITERS)
    st_2disp = time_fn(run_2dispatch_nobounce, WARMUP, ITERS)
    st_1disp = time_fn(run_1dispatch, WARMUP, ITERS)
    st_base = time_fn(run_baseline, WARMUP, ITERS)

    med = statistics.median
    stitch_us = med(st_stitch)
    two_us = med(st_2disp)
    one_us = med(st_1disp)
    base_us = med(st_base)
    swap_cost_us = stitch_us - one_us
    host_2nd_dispatch_us = two_us - one_us

    print("\n============ B2a: re8 GEMM->GEMM dispatcher merge ============")
    print(f"shape: 1x1 GEMM IC={IC}->OC={OC} tile_m={TILE_M} {N_CORES}c ppc={PPC}")
    print(f"       intermediate = {INTER_N} u16 ({INTER_N*2} B), device-resident")
    print()
    print(f"[hw_context count]   BEFORE (2 ELFs) = {baseline_contexts}    "
          f"AFTER (1 merged ELF) = {merged_contexts}    "
          f"DROP = {baseline_contexts - merged_contexts}")
    print(f"                     merged ELF loaded under 1 xrt.hw_context: "
          f"{'YES' if one_context else 'NO'} (entry='main', 2 aiex.configure = 1 PDI swap)")
    print(f"[host dispatches]    BEFORE = {baseline_dispatches}    "
          f"AFTER = {merged_dispatches}    "
          f"DROP = {baseline_dispatches - merged_dispatches}")
    print()
    print(f"[bit-exact]          stitched vs 2-dispatch host-bounce baseline:")
    print(f"                     max_diff={max_diff}  ndiff={ndiff}/{CONS_OUT_N}  "
          f"-> {'PASS (bit-exact)' if exact else 'FAIL'}")
    if not exact:
        idxs = np.flatnonzero(stitch_out != ref_out)[:6]
        print("                     first: " + ", ".join(
            f"[{j}] s={stitch_out[j]:04x} r={ref_out[j]:04x}" for j in idxs))
    print()
    print(f"[latency, median over {ITERS} iters]")
    print(f"  1 host dispatch (producer only)        : {one_us:8.1f} us")
    print(f"  2 host dispatches (no host bounce)      : {two_us:8.1f} us")
    print(f"  2 host dispatches + host bounce (BEFORE): {base_us:8.1f} us")
    print(f"  stitched 1 dispatch (1 on-device swap)  : {stitch_us:8.1f} us")
    print(f"  marginal 2nd HOST dispatch (replaced)   : {host_2nd_dispatch_us:8.1f} us")
    print(f"  on-device PDI swap (stitched - 1 disp)  : {swap_cost_us:8.1f} us")
    print(f"  net win vs host-bounce baseline         : {base_us - stitch_us:8.1f} us")
    print("==============================================================")

    ok = exact and one_context and (merged_contexts < baseline_contexts) \
        and (merged_dispatches < baseline_dispatches)
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{'context-NEGATIVE merge bit-exact on real re8 shape' if ok else 'see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
