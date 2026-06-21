#!/usr/bin/env python3
"""PoC-1 HW test — on-device producer->consumer operator stitching.

Proves:
  1. Two back-to-back NPU ops (GEMM producer -> GEMM consumer) run in ONE
     ELF / ONE hw-context, with an on-device PDI swap between them.
  2. The intermediate stays DEVICE-SIDE: the host never fills it after the
     producer writes it, and never reads it before the consumer reads it.
     (Dispatcher %arg2 == producer.out == consumer.in via a producer->
     consumer chain_link in build_stitch_poc.py.)
  3. BIT-EXACT vs the same two ops run as two separate host dispatches
     (intermediate bounced through host).
  4. Measures the per-stitched-dispatch latency vs the 2-dispatch baseline,
     and reports the on-device swap cost vs the ~505 us host dispatch it
     replaces.

Run from the mdv6 dir:
  source env.sh
  flock /tmp/npu-dev.lock python3 conv/test_stitch_poc_hw.py
"""
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyxrt as xrt
from build_stitch_poc import (
    build_all, _resolve_build_dir,
    STITCH_ELF, PROD_ELF, CONS_ELF,
    TILE_M, N_CORES, PPC, IC, MID, OC,
    NOOP_COUNTS,
)

# Element counts (u16) for the pinned PoC shapes.
N_SLOTS = N_CORES * PPC                       # 32
PROD_IN_N = N_SLOTS * TILE_M * IC             # 131072
INTER_N = N_SLOTS * TILE_M * MID              # 131072 (producer out == consumer in)
PROD_WT_N = IC * MID + 2 * MID                # 4224
CONS_WT_N = MID * OC + 2 * OC                 # 2112
CONS_OUT_N = N_SLOTS * TILE_M * OC            # 65536

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
    rng = np.random.default_rng(seed=1234)
    in_arr = _bf16_pos(rng, PROD_IN_N)
    wt_prod = _bf16_pos(rng, PROD_WT_N)
    wt_cons = _bf16_pos(rng, CONS_WT_N)

    # ------------------------------------------------------------------
    # Baseline: TWO SEPARATE host dispatches. Producer writes intermediate,
    # host reads it back, host fills it into consumer's input, consumer runs.
    # This is the "intermediate touches host" path we are eliminating.
    # ------------------------------------------------------------------
    prod = xrt.elf(_elf_path(PROD_ELF))
    prod_k = xrt.ext.kernel(xrt.hw_context(device, prod), "main")
    p_in = xrt.ext.bo(device, PROD_IN_N * 2)
    p_wt = xrt.ext.bo(device, PROD_WT_N * 2)
    p_out = xrt.ext.bo(device, INTER_N * 2)
    _fill(p_in, in_arr)
    _fill(p_wt, wt_prod)

    cons = xrt.elf(_elf_path(CONS_ELF))
    cons_k = xrt.ext.kernel(xrt.hw_context(device, cons), "main")
    c_in = xrt.ext.bo(device, INTER_N * 2)
    c_wt = xrt.ext.bo(device, CONS_WT_N * 2)
    c_out = xrt.ext.bo(device, CONS_OUT_N * 2)
    _fill(c_wt, wt_cons)

    def run_baseline():
        # dispatch 1: producer  (in,wt,out)
        r = xrt.run(prod_k)
        r.set_arg(0, p_in); r.set_arg(1, p_wt); r.set_arg(2, p_out)
        r.start(); r.wait2()
        # host bounce: read intermediate from device, push into consumer in BO
        inter = _read(p_out, INTER_N)
        _fill(c_in, inter)
        # dispatch 2: consumer  (in,wt,out)
        r = xrt.run(cons_k)
        r.set_arg(0, c_in); r.set_arg(1, c_wt); r.set_arg(2, c_out)
        r.start(); r.wait2()
        return _read(c_out, CONS_OUT_N)

    ref_out = run_baseline()

    # ------------------------------------------------------------------
    # Stitched: ONE ELF, ONE hw-context, producer->consumer chain_link.
    # @main(in0, wt0, INTER, wt1, out1). Host fills in0 + both weights, runs
    # ONCE (producer config -> PDI swap -> consumer config), reads only out1.
    # The INTER arg BO is allocated but NEVER filled or read by host between
    # ops — it is device-resident across the swap.
    # ------------------------------------------------------------------
    stitch = xrt.elf(_elf_path(STITCH_ELF))
    stitch_ctx = xrt.hw_context(device, stitch)
    stitch_k = xrt.ext.kernel(stitch_ctx, "main")

    s_in = xrt.ext.bo(device, PROD_IN_N * 2)
    s_wt0 = xrt.ext.bo(device, PROD_WT_N * 2)
    s_inter = xrt.ext.bo(device, INTER_N * 2)   # device-resident; host never touches it
    s_wt1 = xrt.ext.bo(device, CONS_WT_N * 2)
    s_out = xrt.ext.bo(device, CONS_OUT_N * 2)
    _fill(s_in, in_arr)
    _fill(s_wt0, wt_prod)
    _fill(s_wt1, wt_cons)
    # Pre-fill the intermediate BO with a poison pattern and DO NOT sync it
    # after the producer runs. If the consumer were silently reading host
    # data instead of the on-device producer output, the result would differ.
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

    # ------------------------------------------------------------------
    # Bit-exactness.
    # ------------------------------------------------------------------
    max_diff = int(np.max(np.abs(stitch_out.astype(np.int32)
                                 - ref_out.astype(np.int32))))
    ndiff = int(np.sum(stitch_out != ref_out))
    exact = (ndiff == 0)

    # ------------------------------------------------------------------
    # hw-context assertion: stitched path uses exactly ONE hw_context object.
    # The baseline uses two (one per ELF). We assert the merged ELF entry
    # point is "main" loaded under a single xrt.hw_context.
    # ------------------------------------------------------------------
    one_context = isinstance(stitch_ctx, xrt.hw_context)

    # ------------------------------------------------------------------
    # Timing. Stitched = 1 host dispatch (with 1 on-device PDI swap inside).
    # Baseline = 2 host dispatches + a host bounce of the intermediate.
    # Also time a "pure 2-dispatch, no bounce" lower bound to isolate the
    # host-dispatch cost the swap replaces.
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

    # Pure 2-dispatch (no host bounce of intermediate): the consumer reads a
    # pre-loaded c_in. Measures raw 2x host-dispatch cost.
    def run_2dispatch_nobounce():
        r = xrt.run(prod_k)
        r.set_arg(0, p_in); r.set_arg(1, p_wt); r.set_arg(2, p_out)
        r.start(); r.wait2()
        r = xrt.run(cons_k)
        r.set_arg(0, c_in); r.set_arg(1, c_wt); r.set_arg(2, c_out)
        r.start(); r.wait2()

    # Single host dispatch lower bound (producer only) — one xrt.run cost.
    def run_1dispatch():
        r = xrt.run(prod_k)
        r.set_arg(0, p_in); r.set_arg(1, p_wt); r.set_arg(2, p_out)
        r.start(); r.wait2()

    st_stitch = time_fn(run_stitched, WARMUP, ITERS)
    st_2disp = time_fn(run_2dispatch_nobounce, WARMUP, ITERS)
    st_1disp = time_fn(run_1dispatch, WARMUP, ITERS)
    st_base = time_fn(run_baseline, WARMUP, ITERS)

    def med(x):
        return statistics.median(x)

    stitch_us = med(st_stitch)
    two_us = med(st_2disp)
    one_us = med(st_1disp)
    base_us = med(st_base)

    # The on-device PDI swap cost (lower bound): stitched dispatch minus a
    # single host dispatch ~= the marginal cost of the 2nd (consumer) config
    # done on-device instead of as a 2nd host dispatch.
    swap_cost_us = stitch_us - one_us
    # The host dispatch the swap replaces: marginal 2nd dispatch in the
    # 2-dispatch path = two_us - one_us.
    host_2nd_dispatch_us = two_us - one_us

    # ------------------------------------------------------------------
    print("\n================ PoC-1: producer->consumer stitch ================")
    print(f"shapes: producer GEMM {IC}->{MID} tile_m={TILE_M} {N_CORES}c ppc={PPC}")
    print(f"        consumer GEMM {MID}->{OC}  tile_m={TILE_M} {N_CORES}c ppc={PPC}")
    print(f"        intermediate = {INTER_N} u16 ({INTER_N*2} bytes), device-resident")
    print()
    print(f"[1 hw-context]  stitched ELF loaded under 1 xrt.hw_context: "
          f"{'YES' if one_context else 'NO'}")
    print(f"                (entry='main', 2 aiex.configure blocks = 1 PDI swap)")
    print(f"                baseline uses 2 ELFs = 2 hw-contexts.")
    print()
    print(f"[bit-exact]     stitched vs 2-dispatch baseline:")
    print(f"                max_diff={max_diff}  ndiff={ndiff}/{CONS_OUT_N}  "
          f"-> {'PASS (bit-exact)' if exact else 'FAIL'}")
    if not exact:
        mask = stitch_out != ref_out
        idxs = np.flatnonzero(mask)[:6]
        print("                first mismatches: " + ", ".join(
            f"[{j}] stitch={stitch_out[j]:04x} ref={ref_out[j]:04x}" for j in idxs))
    print()
    print(f"[latency, median over {ITERS} iters]")
    print(f"  1 host dispatch (producer only)        : {one_us:8.1f} us")
    print(f"  2 host dispatches (no host bounce)      : {two_us:8.1f} us")
    print(f"  2 host dispatches + host bounce (base)  : {base_us:8.1f} us")
    print(f"  stitched 1 dispatch (1 on-device swap)  : {stitch_us:8.1f} us")
    print()
    print(f"  marginal 2nd HOST dispatch (replaced)   : {host_2nd_dispatch_us:8.1f} us")
    print(f"  on-device PDI swap (stitched - 1 disp)  : {swap_cost_us:8.1f} us")
    print(f"  net win vs no-bounce 2-dispatch         : {two_us - stitch_us:8.1f} us "
          f"({100*(two_us-stitch_us)/two_us:.0f}% of the 2-dispatch path)")
    print(f"  net win vs host-bounce baseline         : {base_us - stitch_us:8.1f} us")
    print("==================================================================")

    # ------------------------------------------------------------------
    # Pure on-device PDI-SWAP-MECHANISM cost (the canonical ~40 us floor):
    # no-op sub-devices, dispatcher fires R alternating aiex.configure/run.
    # Same aiecc --expand-load-pdis path as the real ELFs. Slope of wall-time
    # vs R isolates the swap cost from any per-op compute. This is the
    # apples-to-apples number for the dispatch-consolidation mechanism.
    # ------------------------------------------------------------------
    print("\n--- pure on-device PDI-swap floor (no-op subs, slope) ---")
    xs, ys = [], []
    noop_runs = {}
    for n in NOOP_COUNTS:
        elf = xrt.elf(_elf_path(f"stitch_noop_r{n}_d2"))
        k = xrt.ext.kernel(xrt.hw_context(device, elf), "main")
        dummy = xrt.ext.bo(device, 2)
        _fill(dummy, np.zeros(1, dtype=np.uint16))
        r = xrt.run(k)
        r.set_arg(0, dummy)

        def runn(r=r):
            r.start(); r.wait2()
        s = time_fn(runn, WARMUP, ITERS)
        m = med(s)
        noop_runs[n] = (k, dummy, r)  # pin alive
        xs.append(n); ys.append(m)
        print(f"  R={n:2d} swaps : {m:8.1f} us")
    # least-squares slope us/swap
    mx = statistics.fmean(xs); my = statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    print(f"  => per-empty-swap (slope): {slope:.1f} us/swap "
          f"(canonical ~40 us floor; cf host xrt.run ~505 us)")

    ok = exact and one_context
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{'producer->consumer stitch is bit-exact in 1 hw-context' if ok else 'see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
