#!/usr/bin/env python3
"""Minimal reproducer: XRT silent context switch failure on AMD NPU (npu2).

After loading and running ~14 different xclbin contexts, subsequent
DefaultNPURuntime.run() calls silently return all-zero output instead
of producing correct results or raising an error.

The bug is in XRT/xdna-driver context switching — the NPU fails to
properly restore a context after many switches, and the run completes
without error but with stale/zero output data.

Reproduces on: AMD Strix Halo (AIE2P/npu2), xdna-driver, XRT 2024+
Filed as: mlir-aie-mi7.2

Usage:
    source ~/npu-dev-mdv6/env.sh
    python3 test_xrt_context_switch.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

BD = os.path.join(os.path.dirname(__file__), "conv", "build")


def bf16_to_uint16(arr):
    import torch
    return torch.tensor(np.asarray(arr, dtype=np.float32),
                        dtype=torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)


def uint16_to_bf16(arr):
    import torch
    return torch.tensor(arr.astype(np.int16),
                        dtype=torch.int16).view(torch.bfloat16).float().numpy()


def load_and_run(name, nc, tile_h, tile_w, ic, oc, ks, stride):
    """Load an xclbin and run it once with dummy data."""
    xclbin = os.path.join(BD, f"{name}.xclbin")
    insts = os.path.join(BD, f"{name}.bin")
    if not os.path.exists(xclbin):
        return None

    kh = DefaultNPURuntime.load(NPUKernel(xclbin, insts))

    ph = (tile_h - 1) * stride + ks
    pw = (tile_w - 1) * stride + ks
    ps_raw = ph * pw * ic
    ps = ps_raw + (ps_raw % 2)
    ots = tile_h * tile_w * oc
    cwt = oc * ic * ks * ks
    wbs = cwt + 2 * oc

    inp = np.zeros(nc * ps, dtype=np.uint16)
    wts = np.zeros(wbs, dtype=np.uint16)
    out = iron.zeros(nc * ots, dtype=np.uint16)

    DefaultNPURuntime.run(kh, [
        iron.tensor(inp, dtype=np.uint16),
        iron.tensor(wts, dtype=np.uint16),
        out
    ])
    return kh


def test_target_xclbin(name, nc, tile_h, tile_w, ic, oc, ks, stride):
    """Load target xclbin and run with non-zero data. Returns (nz, total)."""
    xclbin = os.path.join(BD, f"{name}.xclbin")
    insts = os.path.join(BD, f"{name}.bin")
    kh = DefaultNPURuntime.load(NPUKernel(xclbin, insts))

    ph = (tile_h - 1) * stride + ks
    ps_raw = ph * ph * ic
    ps = ps_raw + (ps_raw % 2)
    ots = tile_h * tile_w * oc
    cwt = oc * ic * ks * ks

    patch = bf16_to_uint16(np.ones(ps_raw) * 0.5)
    if len(patch) < ps:
        patch = np.pad(patch, (0, ps - len(patch)))
    input_all = np.tile(patch, nc)
    wt = np.concatenate([
        bf16_to_uint16(np.ones(cwt) * 0.01),
        bf16_to_uint16(np.ones(oc)),
        bf16_to_uint16(np.zeros(oc)),
    ])

    in_buf = iron.tensor(input_all, dtype=np.uint16)
    wt_buf = iron.tensor(wt, dtype=np.uint16)
    out_buf = iron.zeros(nc * ots, dtype=np.uint16)

    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])

    out = uint16_to_bf16(out_buf.numpy().copy())
    nz = int((out[:nc * ots] != 0).sum())
    return nz, nc * ots


# Single-core (nc=1) xclbins that get loaded before the target
SC_XCLBINS = [
    # (name, nc, tile_h, tile_w, ic, oc, ks, stride)
    ("ftconv0",              1, 24, 24,   3, 32, 3, 2),
    ("ftconv1",              1, 12, 12,  32, 16, 3, 2),
    ("tf_elan_conv1",        1,  8,  8,  64, 64, 1, 1),
    ("tf_elan_conv3x3",      1, 16, 16,  32, 32, 3, 1),
    ("tf_elan_conv4",        1,  8,  8, 128, 64, 1, 1),
    ("tf_aconv3",            1,  8,  8,  64, 16, 3, 2),
    ("re4_conv1",            1, 10, 10, 128, 64, 1, 1),
    ("re4_rn_conv1x1_64_32", 1, 16, 16,  64, 32, 1, 1),
    ("re4_rn_conv3x3_32_32", 1, 16, 16,  32, 32, 3, 1),
    ("re4_conv3x3",          1, 12, 12,  64, 16, 3, 1),
    ("re4_conv4",            1,  8,  8, 256, 32, 1, 1),
    ("re6_conv1",            1,  8,  8, 192, 32, 1, 1),
    ("re6_rn_c1",            1, 10, 10,  96, 48, 1, 1),
]

# Multicore (nc=32) xclbins loaded and run between SC contexts
MC_XCLBINS = [
    ("mc_elan_c1",  32,  8,  8,  64, 64, 1, 1),
    ("mc_aconv5",   32,  4,  4,  96,  8, 3, 2),
]

# Target: mc_re6_rn3 (32-core, 3x3 conv, ic=48 oc=16)
TARGET = ("mc_re6_rn3", 32, 8, 8, 48, 16, 3, 1)


def main():
    print("=" * 60)
    print("XRT silent context switch failure reproducer")
    print("=" * 60)
    print()

    # Step 1: Verify target works in isolation
    print("Step 1: Target xclbin in isolation...")
    nz, total = test_target_xclbin(*TARGET)
    print(f"  {TARGET[0]}: nz={nz}/{total} → {'PASS' if nz == total else 'FAIL'}")
    assert nz == total, "Target must work in isolation"
    print()

    # Step 2: Load and run SC + MC xclbins (simulating full model)
    print("Step 2: Loading/running prior contexts...")
    loaded = 0
    for cfg in SC_XCLBINS:
        h = load_and_run(*cfg)
        if h is not None:
            loaded += 1
            print(f"  {cfg[0]}: loaded+run ({loaded} ctx)")

    for cfg in MC_XCLBINS:
        h = load_and_run(*cfg)
        if h is not None:
            loaded += 1
            # Run MC xclbins multiple times (as in real model)
            for _ in range(3):
                ps = ((cfg[2]-1)*cfg[7]+cfg[6]) ** 2 * cfg[4]
                ps += ps % 2
                ots = cfg[2] * cfg[3] * cfg[5]
                DefaultNPURuntime.run(h, [
                    iron.tensor(np.zeros(cfg[1]*ps, dtype=np.uint16), dtype=np.uint16),
                    iron.tensor(np.zeros(cfg[5]*cfg[4]*cfg[6]**2+2*cfg[5], dtype=np.uint16), dtype=np.uint16),
                    iron.zeros(cfg[1]*ots, dtype=np.uint16)])
            print(f"  {cfg[0]}: loaded+run×4 ({loaded} ctx)")

    # Load 2 more SC to simulate re6 path
    for name, nc, th, tw, ic, oc, ks, st in SC_XCLBINS[-2:]:
        xf = os.path.join(BD, f"{name}.xclbin")
        if os.path.exists(xf):
            kh = DefaultNPURuntime.load(NPUKernel(xf, os.path.join(BD, f"{name}.bin")))
            DefaultNPURuntime.run(kh, [
                iron.tensor(np.zeros(nc*((th-1)*st+ks)**2*ic, dtype=np.uint16), dtype=np.uint16),
                iron.tensor(np.zeros(oc*ic*ks**2+2*oc, dtype=np.uint16), dtype=np.uint16),
                iron.zeros(nc*th*tw*oc, dtype=np.uint16)])
            loaded += 1

    print(f"\n  Total contexts loaded: {loaded}")
    print()

    # Step 3: Test target after all the context switches
    print("Step 3: Target xclbin after context pressure...")
    nz, total = test_target_xclbin(*TARGET)
    status = "PASS" if nz == total else "FAIL (ZEROS)"
    print(f"  {TARGET[0]}: nz={nz}/{total} → {status}")
    print()

    if nz == total:
        print("BUG NOT REPRODUCED — context switching worked correctly.")
        print("(This may be system-dependent; the bug reproduces when the")
        print("target xclbin is called via _run_tiled_mc_inner with the")
        print("OC-block loop in the full MDV6 model pipeline.)")
    else:
        print("BUG REPRODUCED: DefaultNPURuntime.run() returned all-zero")
        print("output after context switching. The NPU failed to properly")
        print("execute the target context after ~14 context switches.")

    return nz == total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
