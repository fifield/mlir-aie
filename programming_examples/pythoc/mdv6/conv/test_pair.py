#!/usr/bin/env python3
"""Bytewise correctness check for GEMM rn1 pair ELFs.

For each shape in build_pair_rn1._RN1_PAIRS, dispatches the pair ELF
(shared input BO, two independent wt+out args) and compares each output
against the single-kernel ELF run twice with the same weights. They must
be bytewise identical — same input, same weights, same kernel.

Missing ELFs (pair or single counterpart) are built on demand.

Run from the mdv6 dir:
  source env.sh && source venv/bin/activate
  flock /tmp/npu-dev.lock python3 conv/test_pair.py --shape all
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyxrt as xrt
from build_merged import build_merged, _resolve_build_dir
from build_pair_rn1 import _RN1_PAIRS, _GEMM_SCRIPT

N_CORES = 32


def _shape_for(label):
    for t in _RN1_PAIRS:
        if t[0] == label:
            return t
    raise KeyError(f"unknown shape: {label}")


def _pair_elf_name(tile_m, ic, oc, ppc):
    return f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_pair_x1"


def _single_elf_name(tile_m, ic, oc, ppc):
    return f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_x1"


def _build_pair_if_missing(tile_m, ic, oc, ppc):
    name = _pair_elf_name(tile_m, ic, oc, ppc)
    elf = os.path.join(_resolve_build_dir(), f"{name}.elf")
    if os.path.exists(elf):
        return elf
    print(f"  building {name}.elf (missing)...")
    sub_args = ["32", str(tile_m), str(ic), str(oc), str(ppc), "0"]
    sub_names = [f"gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_a",
                 f"gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_b"]
    path = build_merged(
        name, sub_names, kind="gemm",
        sub_spec_overrides={
            sub_names[0]: (_GEMM_SCRIPT, sub_args),
            sub_names[1]: (_GEMM_SCRIPT, sub_args),
        },
        chain_links=[(0, 0, 1, 0)],  # sub1.in = sub0.in
    )
    if path is None:
        raise RuntimeError(f"build failed: {name}")
    return path


def _build_single_if_missing(tile_m, ic, oc, ppc):
    name = _single_elf_name(tile_m, ic, oc, ppc)
    elf = os.path.join(_resolve_build_dir(), f"{name}.elf")
    if os.path.exists(elf):
        return elf
    print(f"  building {name}.elf (missing)...")
    sub_args = ["32", str(tile_m), str(ic), str(oc), str(ppc), "0"]
    sub_label = f"gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}"
    path = build_merged(
        name, [sub_label], kind="gemm",
        share_arg_idxs={1},  # arg0=wt, arg1=in, arg2=out
        sub_spec_overrides={sub_label: (_GEMM_SCRIPT, sub_args)},
    )
    if path is None:
        raise RuntimeError(f"build failed: {name}")
    return path


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _bo_read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def _run_one(label, device):
    _, tile_m, ic, oc, ppc = _shape_for(label)
    in_nelem = N_CORES * ppc * tile_m * ic
    out_nelem = N_CORES * ppc * tile_m * oc
    wt_nelem = ic * oc + 2 * oc
    print(f"[{label}] tile_m={tile_m} ic={ic} oc={oc} ppc={ppc}")
    print(f"  in={in_nelem}u16 out={out_nelem}u16 wt={wt_nelem}u16")

    pair_elf = _build_pair_if_missing(tile_m, ic, oc, ppc)
    single_elf = _build_single_if_missing(tile_m, ic, oc, ppc)

    rng = np.random.default_rng(seed=42)
    in_arr = rng.integers(0x3c00, 0x4000, size=in_nelem, dtype=np.uint16)
    wt_a = rng.integers(0x3c00, 0x4000, size=wt_nelem, dtype=np.uint16)
    wt_b = rng.integers(0x3c00, 0x4000, size=wt_nelem, dtype=np.uint16)

    # --- Pair ELF: one xrt.run, shared in, two wt/out ---
    pair = xrt.elf(pair_elf)
    pair_kernel = xrt.ext.kernel(xrt.hw_context(device, pair), "main")
    pair_in = xrt.ext.bo(device, in_nelem * 2)
    pair_wt0 = xrt.ext.bo(device, wt_nelem * 2)
    pair_out0 = xrt.ext.bo(device, out_nelem * 2)
    pair_wt1 = xrt.ext.bo(device, wt_nelem * 2)
    pair_out1 = xrt.ext.bo(device, out_nelem * 2)
    _bo_fill(pair_in, in_arr)
    _bo_fill(pair_wt0, wt_a)
    _bo_fill(pair_wt1, wt_b)

    t0 = time.perf_counter()
    r = xrt.run(pair_kernel)
    r.set_arg(0, pair_in)
    r.set_arg(1, pair_wt0)
    r.set_arg(2, pair_out0)
    r.set_arg(3, pair_wt1)
    r.set_arg(4, pair_out1)
    r.start()
    r.wait2()
    pair_ms = (time.perf_counter() - t0) * 1000
    out0 = _bo_read(pair_out0, out_nelem)
    out1 = _bo_read(pair_out1, out_nelem)
    print(f"  pair (1 call):    {pair_ms:.2f} ms")

    # --- Single ELF: two sequential runs with each weight ---
    single = xrt.elf(single_elf)
    single_kernel = xrt.ext.kernel(xrt.hw_context(device, single), "main")
    s_in = xrt.ext.bo(device, in_nelem * 2)
    s_wt = xrt.ext.bo(device, wt_nelem * 2)
    s_out = xrt.ext.bo(device, out_nelem * 2)
    _bo_fill(s_in, in_arr)

    refs = []
    total_single_ms = 0.0
    for wt in (wt_a, wt_b):
        _bo_fill(s_wt, wt)
        t0 = time.perf_counter()
        r = xrt.run(single_kernel)
        # share_arg_idxs={1} → arg0=wt, arg1=in, arg2=out
        r.set_arg(0, s_wt)
        r.set_arg(1, s_in)
        r.set_arg(2, s_out)
        r.start()
        r.wait2()
        total_single_ms += (time.perf_counter() - t0) * 1000
        refs.append(_bo_read(s_out, out_nelem))
    print(f"  single (2 calls): {total_single_ms:.2f} ms "
          f"→ speedup {total_single_ms/pair_ms:.2f}×")

    diff0 = int(np.sum(out0 != refs[0]))
    diff1 = int(np.sum(out1 != refs[1]))
    if diff0 == 0 and diff1 == 0:
        print(f"[{label}] PASS — both pair outputs match single-twice baseline.")
        return True
    print(f"[{label}] FAIL: out0_diff={diff0}/{out_nelem} out1_diff={diff1}/{out_nelem}")
    for tag, ocb_data, ref_data, diff in (
        ("out0", out0, refs[0], diff0),
        ("out1", out1, refs[1], diff1),
    ):
        if diff == 0:
            continue
        mask = ocb_data != ref_data
        idxs = np.flatnonzero(mask)[:4]
        print(f"  {tag} first mismatches: " + ", ".join(
            f"[{j}] pair={ocb_data[j]:04x} ref={ref_data[j]:04x}" for j in idxs))
    return False


def main():
    p = argparse.ArgumentParser()
    shapes = [t[0] for t in _RN1_PAIRS]
    p.add_argument("--shape", choices=shapes + ["all"], default="re6_rn1")
    args = p.parse_args()

    labels = shapes if args.shape == "all" else [args.shape]
    device = xrt.device(0)
    results = []
    for label in labels:
        try:
            ok = _run_one(label, device)
        except Exception as e:
            print(f"[{label}] FAIL: {e}")
            ok = False
        results.append((label, ok))
        print()

    n_pass = sum(1 for _, ok in results if ok)
    print(f"=== {n_pass}/{len(results)} shapes PASS ===")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
