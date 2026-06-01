#!/usr/bin/env python3
"""Bytewise correctness check for packed GEMM spatial-fanout ELFs.

Runs one packed ELF with args [wt, packed_in, packed_out] and compares each
packed output slice against the existing x1 ELF run once per spatial batch.
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyxrt as xrt
from build_merged import build_merged, _resolve_build_dir
from build_packed_gemm import (
    iter_layer_shapes,
    packed_gemm_elf_name,
    build_one as build_packed_one,
    _GEMM_SCRIPT,
    N_CORES,
)
from run_tiled_mc import _merged_gemm_elf_name


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
              np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _bo_read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def _single_elf_if_missing(tile_m, ic, oc, k_block, ppc):
    name = _merged_gemm_elf_name(tile_m, ic, oc, k_block, ppc)
    elf = os.path.join(_resolve_build_dir(), f"{name}.elf")
    if os.path.exists(elf):
        return elf
    print(f"  building {name}.elf (missing)...")
    kb_str = f"kb{k_block}_" if k_block > 0 else ""
    sub_label = f"gemm_t{tile_m}_ic{ic}_oc{oc}_{kb_str}p{ppc}"
    sub_args = [str(N_CORES), str(tile_m), str(ic), str(oc), str(ppc), str(k_block)]
    path = build_merged(
        name, [sub_label], kind="gemm", share_arg_idxs={1},
        sub_spec_overrides={sub_label: (_GEMM_SCRIPT, sub_args)},
    )
    if path is None:
        raise RuntimeError(f"build failed: {name}")
    return path


def _shape_from_layer(layer):
    rows = list(iter_layer_shapes(layer))
    if len(rows) != 1:
        raise KeyError(f"expected exactly one layer row for {layer!r}, got {len(rows)}")
    name, H, W, IC, OC, tile_m, k_block, ppc, n_batches = rows[0]
    return H, W, IC, OC, tile_m, k_block, ppc, n_batches


def _run_layer(layer, device):
    H, W, ic, oc, tile_m, k_block, ppc, n_batches = _shape_from_layer(layer)
    input_size = tile_m * ic
    output_size = tile_m * oc
    total_slots = N_CORES * ppc
    host_in_size = total_slots * input_size
    host_out_size = total_slots * output_size
    if k_block > 0:
        n_k_blocks = ic // k_block
        wt_nelem = n_k_blocks * (k_block * oc + 2 * oc)
    else:
        wt_nelem = ic * oc + 2 * oc

    packed_name = packed_gemm_elf_name(tile_m, ic, oc, k_block, ppc, n_batches)
    print(f"[{layer}] {packed_name}")
    print(f"  n_batches={n_batches} host_in={host_in_size}u16 host_out={host_out_size}u16 wt={wt_nelem}u16")
    packed_elf = os.path.join(_resolve_build_dir(), f"{packed_name}.elf")
    if not os.path.exists(packed_elf):
        print(f"  building {packed_name}.elf (missing)...")
        built = build_packed_one(tile_m, ic, oc, k_block, ppc, n_batches)
        if built is None:
            raise RuntimeError(f"build failed: {packed_name}")
        packed_elf = built
    single_elf = _single_elf_if_missing(tile_m, ic, oc, k_block, ppc)

    rng = np.random.default_rng(seed=20260601)
    wt = rng.integers(0x3c00, 0x4000, size=wt_nelem, dtype=np.uint16)
    ins = [rng.integers(0x3c00, 0x4000, size=host_in_size, dtype=np.uint16)
           for _ in range(n_batches)]
    packed_in = np.concatenate(ins)

    # Packed single dispatch.
    p_elf = xrt.elf(packed_elf)
    p_kernel = xrt.ext.kernel(xrt.hw_context(device, p_elf), "main")
    p_wt = xrt.ext.bo(device, wt.nbytes)
    p_in = xrt.ext.bo(device, packed_in.nbytes)
    p_out = xrt.ext.bo(device, n_batches * host_out_size * 2)
    _bo_fill(p_wt, wt)
    _bo_fill(p_in, packed_in)
    t0 = time.perf_counter()
    r = xrt.run(p_kernel)
    r.set_arg(0, p_wt)
    r.set_arg(1, p_in)
    r.set_arg(2, p_out)
    r.start(); r.wait2()
    packed_ms = (time.perf_counter() - t0) * 1000
    packed_out = _bo_read(p_out, n_batches * host_out_size)
    print(f"  packed (1 call): {packed_ms:.2f} ms")

    # x1 reference, same packed input slices one by one.
    s_elf = xrt.elf(single_elf)
    s_kernel = xrt.ext.kernel(xrt.hw_context(device, s_elf), "main")
    s_wt = xrt.ext.bo(device, wt.nbytes)
    s_in = xrt.ext.bo(device, host_in_size * 2)
    s_out = xrt.ext.bo(device, host_out_size * 2)
    _bo_fill(s_wt, wt)
    refs = []
    x1_ms = 0.0
    for arr in ins:
        _bo_fill(s_in, arr)
        t0 = time.perf_counter()
        r = xrt.run(s_kernel)
        r.set_arg(0, s_wt)
        r.set_arg(1, s_in)
        r.set_arg(2, s_out)
        r.start(); r.wait2()
        x1_ms += (time.perf_counter() - t0) * 1000
        refs.append(_bo_read(s_out, host_out_size))
    print(f"  x1 ({n_batches} calls): {x1_ms:.2f} ms → speedup {x1_ms/packed_ms:.2f}×")

    ok = True
    for i, ref in enumerate(refs):
        got = packed_out[i * host_out_size:(i + 1) * host_out_size]
        diff = int(np.sum(got != ref))
        if diff:
            ok = False
            idxs = np.flatnonzero(got != ref)[:4]
            print(f"  batch{i} diff={diff}/{host_out_size}; " + ", ".join(
                f"[{j}] packed={got[j]:04x} ref={ref[j]:04x}" for j in idxs))
    print(f"[{layer}] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    layers = [row[0] for row in iter_layer_shapes() if row[-1] > 1]
    p.add_argument("--layer", choices=layers + ["all"], default="elan_c1")
    args = p.parse_args()
    selected = layers if args.layer == "all" else [args.layer]
    device = xrt.device(0)
    results = []
    for layer in selected:
        try:
            ok = _run_layer(layer, device)
        except Exception as e:
            print(f"[{layer}] FAIL: {e}")
            ok = False
        results.append((layer, ok))
        print()
    n_pass = sum(1 for _, ok in results if ok)
    print(f"=== {n_pass}/{len(results)} packed GEMM shapes PASS ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
