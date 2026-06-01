#!/usr/bin/env python3
"""Bytewise correctness check for multi-clone batch-fanout ELFs.

The fanout ELFs (merged_<variant>_xN.elf) hold N clones of one mc_<variant>
sub-device with a shared weight arg. For each variant in build_x1_mc._FANOUT,
this test runs the fanout ELF once with one weight + N distinct inputs and
compares each per-clone output against the x1 ELF for the same variant run
N times with the same (wt, in_i).

Missing ELFs are built on demand.

Run from the mdv6 dir:
  source env.sh && source venv/bin/activate
  flock /tmp/npu-dev.lock python3 conv/test_fanout.py --variant all
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyxrt as xrt
from build_merged import build_merged, _resolve_build_dir
from build_x1_mc import _FANOUT
from mc_configs import CONFIGS as MC_CONFIGS


def _shape_for(variant):
    for t in MC_CONFIGS:
        if t[0] == variant:
            return t
    raise KeyError(f"unknown mc variant: {variant}")


def _derived(variant):
    _, n_cores, tile_h, tile_w, ic, oc, ks, stride, ppc = _shape_for(variant)
    patch_h = (tile_h - 1) * stride + ks
    patch_w = (tile_w - 1) * stride + ks
    patch_raw = patch_h * patch_w * ic
    patch_size = patch_raw + (patch_raw % 2)
    core_input = ppc * patch_size
    weight_slot = oc * ic * ks * ks + 2 * oc
    output_tile = tile_h * tile_w * oc
    core_output = ppc * output_tile
    host_input = n_cores * core_input
    host_output = n_cores * core_output
    return dict(host_input=host_input, host_output=host_output,
                weight_slot=weight_slot)


def _build_if_missing(variant, n_clones):
    suffix = "x1" if n_clones == 1 else f"x{n_clones}"
    out = f"merged_{variant.replace('mc_', '')}_{suffix}"
    elf = os.path.join(_resolve_build_dir(), f"{out}.elf")
    if os.path.exists(elf):
        return elf
    print(f"  building {out}.elf (missing)...")
    sub_names = [variant] * n_clones
    path = build_merged(out, sub_names, share_arg_idxs={1})
    if path is None:
        raise RuntimeError(f"build failed: {out}")
    return path


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _bo_read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def _run_one(variant, n_clones, device):
    s = _derived(variant)
    print(f"[{variant} ×{n_clones}] "
          f"host_in={s['host_input']}u16 host_out={s['host_output']}u16 "
          f"wt_slot={s['weight_slot']}u16")

    fanout_elf = _build_if_missing(variant, n_clones)
    x1_elf = _build_if_missing(variant, 1)

    rng = np.random.default_rng(seed=2026)
    wt = rng.integers(0x3c00, 0x4000,
                      size=s["weight_slot"], dtype=np.uint16)
    ins = [rng.integers(0x3c00, 0x4000, size=s["host_input"], dtype=np.uint16)
           for _ in range(n_clones)]

    # --- Fanout ELF: one xrt.run, args = wt + N*(in, out) ---
    fanout = xrt.elf(fanout_elf)
    fanout_kernel = xrt.ext.kernel(xrt.hw_context(device, fanout), "main")

    f_wt = xrt.ext.bo(device, s["weight_slot"] * 2)
    f_ins = [xrt.ext.bo(device, s["host_input"] * 2) for _ in range(n_clones)]
    f_outs = [xrt.ext.bo(device, s["host_output"] * 2) for _ in range(n_clones)]
    _bo_fill(f_wt, wt)
    for bo, arr in zip(f_ins, ins):
        _bo_fill(bo, arr)

    t0 = time.perf_counter()
    r = xrt.run(fanout_kernel)
    r.set_arg(0, f_wt)
    for i in range(n_clones):
        r.set_arg(1 + 2 * i, f_ins[i])
        r.set_arg(2 + 2 * i, f_outs[i])
    r.start()
    r.wait2()
    fanout_ms = (time.perf_counter() - t0) * 1000
    fanout_results = [_bo_read(bo, s["host_output"]) for bo in f_outs]
    print(f"  fanout (1 call, {n_clones} clones): {fanout_ms:.2f} ms")

    # --- x1 ELF: N sequential runs with (wt, in_i) ---
    x1 = xrt.elf(x1_elf)
    x1_kernel = xrt.ext.kernel(xrt.hw_context(device, x1), "main")

    x_wt = xrt.ext.bo(device, s["weight_slot"] * 2)
    x_in = xrt.ext.bo(device, s["host_input"] * 2)
    x_out = xrt.ext.bo(device, s["host_output"] * 2)
    _bo_fill(x_wt, wt)

    refs = []
    total_x1_ms = 0.0
    for arr in ins:
        _bo_fill(x_in, arr)
        t0 = time.perf_counter()
        r = xrt.run(x1_kernel)
        r.set_arg(0, x_wt)
        r.set_arg(1, x_in)
        r.set_arg(2, x_out)
        r.start()
        r.wait2()
        total_x1_ms += (time.perf_counter() - t0) * 1000
        refs.append(_bo_read(x_out, s["host_output"]))
    print(f"  x1 ({n_clones} calls):              {total_x1_ms:.2f} ms "
          f"→ speedup {total_x1_ms/fanout_ms:.2f}×")

    all_match = True
    for i in range(n_clones):
        diff = int(np.sum(fanout_results[i] != refs[i]))
        if diff != 0:
            all_match = False
            mask = fanout_results[i] != refs[i]
            idxs = np.flatnonzero(mask)[:4]
            print(f"  clone{i} diff = {diff}/{s['host_output']}; first: " +
                  ", ".join(f"[{j}] fan={fanout_results[i][j]:04x} "
                            f"ref={refs[i][j]:04x}" for j in idxs))
    if all_match:
        print(f"[{variant} ×{n_clones}] PASS — all {n_clones} clones match x1 reference.")
    else:
        print(f"[{variant} ×{n_clones}] FAIL")
    return all_match


def main():
    p = argparse.ArgumentParser()
    variants = [v for v, _ in _FANOUT]
    p.add_argument("--variant", choices=variants + ["all"], default="all")
    args = p.parse_args()

    selected = _FANOUT if args.variant == "all" else \
        [(args.variant, dict(_FANOUT)[args.variant])]

    device = xrt.device(0)
    results = []
    for variant, n_clones in selected:
        try:
            ok = _run_one(variant, n_clones, device)
        except Exception as e:
            print(f"[{variant} ×{n_clones}] FAIL: {e}")
            ok = False
        results.append((variant, n_clones, ok))
        print()

    n_pass = sum(1 for _, _, ok in results if ok)
    print(f"=== {n_pass}/{len(results)} fanouts PASS ===")
    for variant, n_clones, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {variant} ×{n_clones}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
