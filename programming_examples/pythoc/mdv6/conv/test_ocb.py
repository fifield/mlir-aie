#!/usr/bin/env python3
"""Bytewise correctness check for OCB-unrolled conv ELFs.

For a given layer (in conv/build_ocb.py _LAYERS), runs the OCB-unrolled
ELF (n_ocb=N) once with concatenated weights [wt0, wt1, ..., wtN-1] and
compares its output slice-by-slice against a reference ELF (n_ocb=1) run
N times with each individual weight. Each slice must match bytewise.

Shapes are derived from _LAYERS so adding a new OCB layer doesn't need a
new test file. Missing ELFs are built on demand via build_ocb._build_layer.

Run from the mdv6 dir:
  cd /home/jfifield/npu-dev-pythoc && source env.sh
  flock /tmp/npu-dev.lock python3 mlir-aie/programming_examples/pythoc/mdv6/conv/test_ocb.py --layer re8_rn3
  flock /tmp/npu-dev.lock python3 mlir-aie/programming_examples/pythoc/mdv6/conv/test_ocb.py --layer all
"""
import argparse
import os
import sys
import time
import numpy as np

import pyxrt as xrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ocb
from build_merged import _resolve_build_dir


def _shape(label):
    """Derive all sizes from the build_ocb._LAYERS tuple for `label`."""
    cfgs = build_ocb._all_layers()
    if label not in cfgs:
        raise KeyError(f"unknown layer: {label}")
    (n_cores, tile_h, tile_w, ic, oc_block, n_ocb, ks, stride, ppc,
     _at, _ai, _ao) = cfgs[label]

    patch_h = (tile_h - 1) * stride + ks
    patch_w = (tile_w - 1) * stride + ks
    patch_raw = patch_h * patch_w * ic
    patch_size = patch_raw + (patch_raw % 2)
    core_input = ppc * patch_size
    weight_slot = oc_block * ic * ks * ks + 2 * oc_block
    output_tile = tile_h * tile_w * oc_block
    core_output = ppc * output_tile
    host_input = n_cores * core_input
    host_output = n_cores * core_output

    return dict(
        n_cores=n_cores, tile_h=tile_h, tile_w=tile_w,
        ic=ic, oc_block=oc_block, n_ocb=n_ocb, ks=ks, stride=stride, ppc=ppc,
        host_input=host_input, host_output=host_output,
        weight_slot=weight_slot,
    )


def _ensure_elf(label):
    elf = os.path.join(_resolve_build_dir(), f"ocb_{label}_x1.elf")
    if os.path.exists(elf):
        return elf
    cfgs = build_ocb._all_layers()
    print(f"  building {os.path.basename(elf)} (missing)...")
    if build_ocb._build_layer(label, cfgs[label]) is None:
        raise RuntimeError(f"build failed: {label}")
    return elf


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _bo_read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def _run_one(label, device):
    s = _shape(label)
    n_ocb = s["n_ocb"]
    if n_ocb < 2:
        print(f"[{label}] SKIP: n_ocb={n_ocb} — nothing to unroll, "
              "OCB ELF IS its own reference.")
        return True

    ocb_elf = _ensure_elf(label)
    ref_elf = _ensure_elf(f"{label}_ref")

    print(f"[{label}] tile={s['tile_h']}×{s['tile_w']} ic={s['ic']} "
          f"oc_block={s['oc_block']} n_ocb={n_ocb} ppc={s['ppc']} "
          f"stride={s['stride']}")
    print(f"  host_input={s['host_input']}u16 host_output={s['host_output']}u16 "
          f"weight_slot={s['weight_slot']}u16")

    # Mid-range bf16 values (~1.0..2.0) to exercise the kernel without NaNs.
    rng = np.random.default_rng(seed=2026)
    in_arr = rng.integers(0x3c00, 0x4000,
                          size=s["host_input"], dtype=np.uint16)
    wts = [rng.integers(0x3c00, 0x4000,
                        size=s["weight_slot"], dtype=np.uint16)
           for _ in range(n_ocb)]

    # --- OCB-unrolled ELF: one xrt.run with concatenated weights ---
    ocb = xrt.elf(ocb_elf)
    ocb_kernel = xrt.ext.kernel(xrt.hw_context(device, ocb), "main")

    big_W = np.concatenate(wts)
    big_W_bo = xrt.ext.bo(device, big_W.nbytes)
    in_bo = xrt.ext.bo(device, in_arr.nbytes)
    big_O_bo = xrt.ext.bo(device, n_ocb * s["host_output"] * 2)

    _bo_fill(in_bo, in_arr)
    _bo_fill(big_W_bo, big_W)

    # share_arg_idxs={1} → arg0=wt, arg1=in, arg2=out
    t0 = time.perf_counter()
    r = xrt.run(ocb_kernel)
    r.set_arg(0, big_W_bo)
    r.set_arg(1, in_bo)
    r.set_arg(2, big_O_bo)
    r.start()
    r.wait2()
    ocb_ms = (time.perf_counter() - t0) * 1000
    big_O = _bo_read(big_O_bo, n_ocb * s["host_output"])
    print(f"  ocb ({n_ocb} OCBs, 1 call):  {ocb_ms:.2f} ms")

    # --- Reference: n_ocb=1 ELF run n_ocb times with each individual weight ---
    ref = xrt.elf(ref_elf)
    ref_kernel = xrt.ext.kernel(xrt.hw_context(device, ref), "main")

    ref_in_bo = xrt.ext.bo(device, in_arr.nbytes)
    ref_W_bo = xrt.ext.bo(device, s["weight_slot"] * 2)
    ref_O_bo = xrt.ext.bo(device, s["host_output"] * 2)
    _bo_fill(ref_in_bo, in_arr)

    refs = []
    total_ref_ms = 0.0
    for wt in wts:
        _bo_fill(ref_W_bo, wt)
        t0 = time.perf_counter()
        r = xrt.run(ref_kernel)
        r.set_arg(0, ref_W_bo)
        r.set_arg(1, ref_in_bo)
        r.set_arg(2, ref_O_bo)
        r.start()
        r.wait2()
        total_ref_ms += (time.perf_counter() - t0) * 1000
        refs.append(_bo_read(ref_O_bo, s["host_output"]))
    print(f"  ref ({n_ocb} calls):         {total_ref_ms:.2f} ms "
          f"→ speedup {total_ref_ms/ocb_ms:.2f}×")

    all_match = True
    for i in range(n_ocb):
        slice_ = big_O[i * s["host_output"]:(i + 1) * s["host_output"]]
        diff = int(np.sum(slice_ != refs[i]))
        if diff != 0:
            all_match = False
            mask = slice_ != refs[i]
            idxs = np.flatnonzero(mask)[:4]
            print(f"  OCB{i} diff = {diff}/{s['host_output']}; "
                  f"first mismatches: " + ", ".join(
                      f"[{j}] ocb={slice_[j]:04x} ref={refs[i][j]:04x}"
                      for j in idxs))
    if all_match:
        print(f"[{label}] PASS — all {n_ocb} OCB slices match reference bytewise.")
    else:
        print(f"[{label}] FAIL")
    return all_match


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer",
                   choices=list(build_ocb._LAYERS) + ["all"],
                   default="re8_rn3",
                   help="`all` runs every production OCB layer "
                        "(skips re4_rn3 which has n_ocb=1)")
    args = p.parse_args()

    if args.layer == "all":
        layers = [k for k in build_ocb._LAYERS if k != "re4_rn3"]
    else:
        layers = [args.layer]

    device = xrt.device(0)
    results = []
    for label in layers:
        try:
            ok = _run_one(label, device)
        except Exception as e:
            print(f"[{label}] FAIL: {e}")
            ok = False
        results.append((label, ok))
        print()

    n_pass = sum(1 for _, ok in results if ok)
    print(f"=== {n_pass}/{len(results)} layers PASS ===")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
