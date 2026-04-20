#!/usr/bin/env python3
"""Build all GEMM conv1x1 xclbins needed for the full MDV6 model.

Generates MLIR via aie2_gemm_conv1x1.py and compiles for each unique
(tile_m, ic, oc, k_block, ppc) config.

With K-blocking (mi7.4): weights are split into k_block-sized IC chunks.
Full OC is processed in one call, eliminating OC blocking at the host level.
Weight layout per chunk: [k_block/8, oc/8, 8, 8] + [bn_w(oc), bn_b(oc)].
"""
import math
import os
import subprocess
import sys
import time

L1_BYTES = 65536
STACK_BYTES = 8192
AVAIL = L1_BYTES - STACK_BYTES


def compute_tile_m_kblocked(ic, oc, k_block):
    """Max tile_m (mult of 4) that fits in L1 with K-blocked weights.

    L1 holds: input(tile_m*ic) + output(tile_m*oc) + wt_chunk(k_block*oc+2*oc)
    All depth=1 (single-buffered).
    """
    wt_chunk_bytes = (k_block * oc + 2 * oc) * 2
    remaining = AVAIL - wt_chunk_bytes
    if remaining <= 0:
        return 0
    per_pixel = (ic + oc) * 2  # input + output per pixel
    max_tm = remaining // per_pixel
    return (max_tm // 4) * 4


def compute_tile_m(ic, oc_block):
    """Max tile_m (mult of 4) that fits in L1 with depth=1 FIFOs (legacy)."""
    wt_bytes = (ic * oc_block + 2 * oc_block) * 2
    remaining = AVAIL - wt_bytes
    if remaining <= 0:
        return 0
    per_pixel = (ic + oc_block) * 2
    max_tm = remaining // per_pixel
    return (max_tm // 4) * 4


MAX_K_BLOCKS = 16  # Cap unrolled K-block loop to fit in 16KB instruction memory


def choose_k_block(ic, oc, M, n_cores=32):
    """Choose k_block that minimizes spatial calls while capping K-blocks.

    Returns (k_block, tile_m) or (0, tile_m) if no K-blocking needed.
    k_block=0 means use original non-K-blocked path.
    """
    # First try: no K-blocking (full IC weights fit in L1)
    tm_full = compute_tile_m(ic, oc)
    if tm_full >= 16:
        return 0, min(tm_full, 256)

    # Need K-blocking. Find k_block that minimizes ceil(M / (tile_m * n_cores)).
    # Constrain: n_k_blocks <= MAX_K_BLOCKS, k_block % 8 == 0, ic % k_block == 0
    import math
    best_kb, best_tm, best_calls = 0, 0, float('inf')
    for n_kb in range(2, MAX_K_BLOCKS + 1):
        kb = ic // n_kb
        if kb < 8 or kb % 8 != 0 or ic % kb != 0:
            continue
        tm = compute_tile_m_kblocked(ic, oc, kb)
        tm = min(tm, 256)
        if tm < 4:
            continue
        calls = math.ceil(M / (tm * n_cores))
        # Prefer fewer calls; tie-break by fewer K-blocks (less overhead)
        if calls < best_calls or (calls == best_calls and n_kb < ic // best_kb):
            best_kb, best_tm, best_calls = kb, tm, calls
    return best_kb, best_tm


# All 1×1 conv layers in the MDV6 model.
# (name, H, W, IC, OC)
MODEL_LAYERS_1x1 = [
    # ELAN2 (160×160)
    ("elan_c1",     160, 160,  64,  64),
    ("elan_c4",     160, 160, 128,  64),
    # RepNCSPELAN4 (80×80, 128ch)
    ("re4_c1",       80,  80, 128, 128),
    ("re4_rn1",      80,  80,  64,  32),
    ("re4_c4",       80,  80, 256, 128),
    # RepNCSPELAN6 (40×40, 192ch)
    ("re6_c1",       40,  40, 192, 192),
    ("re6_rn1",      40,  40,  96,  48),
    ("re6_rnm",      40,  40,  96,  96),
    ("re6_c4",       40,  40, 384, 192),
    # RepNCSPELAN8 (20×20, 256ch)
    ("re8_c1",       20,  20, 256, 256),
    ("re8_rn1",      20,  20, 128,  64),
    ("re8_rnm",      20,  20, 256, 256),
    ("re8_c4",       20,  20, 512, 256),
    # SPP9
    ("spp_c1",       20,  20, 256, 128),
    # Neck
    ("re12_c1",      40,  40, 448, 192),
    ("re15_c1",      80,  80, 320, 128),
    ("re15_c4",      80,  80, 256, 128),
    ("re15_rnm",     80,  80, 128, 128),
    # Heads
    ("re18_c1",      40,  40, 288, 192),
    ("re21_c1",      20,  20, 384, 256),
]


XRT_BUF_MAX = 16 * 1024 * 1024  # 16MB per XRT buffer argument
L2_BUDGET = 400 * 1024  # ~400KB usable per memtile column


def compute_ppc_kblocked(M, tile_m, ic, oc, k_block, n_cores=32):
    """Compute optimal patches_per_core for K-blocked config."""
    ideal = math.ceil(M / (n_cores * tile_m))

    # Cap by XRT host buffer limits
    in_bytes = n_cores * tile_m * ic * 2
    out_bytes = n_cores * tile_m * oc * 2
    max_xrt_in = XRT_BUF_MAX // in_bytes if in_bytes > 0 else 999
    max_xrt_out = XRT_BUF_MAX // out_bytes if out_bytes > 0 else 999

    # Cap by L2 memtile budget per column (4 cores per column)
    col_in = 4 * tile_m * ic * 2       # per-column input per patch
    col_out = 4 * tile_m * oc * 2      # per-column output per patch
    n_k_blocks = ic // k_block if k_block > 0 else 1
    # Weight chunks stream through L2; at most depth=1 chunk buffered
    wt = (k_block * oc + 2 * oc) * 2 if k_block > 0 else (ic * oc + 2 * oc) * 2
    per_ppc = col_in + col_out
    max_l2 = (L2_BUDGET - wt) // per_ppc if per_ppc > 0 else 999

    return max(1, min(ideal, max_xrt_in, max_xrt_out, max_l2, 32))


def compute_ppc(M, tile_m, ic, oc_block):
    """Compute optimal patches_per_core (legacy, non-K-blocked)."""
    ideal = math.ceil(M / (32 * tile_m))
    in_bytes = 32 * tile_m * ic * 2
    out_bytes = 32 * tile_m * oc_block * 2
    max_xrt_in = XRT_BUF_MAX // in_bytes if in_bytes > 0 else 999
    max_xrt_out = XRT_BUF_MAX // out_bytes if out_bytes > 0 else 999
    col_in = 4 * tile_m * ic * 2
    col_out = 4 * tile_m * oc_block * 2
    wt = (ic * oc_block + 2 * oc_block) * 2
    per_ppc = col_in + col_out
    max_l2 = (L2_BUDGET - wt) // per_ppc if per_ppc > 0 else 999
    return max(1, min(ideal, max_xrt_in, max_xrt_out, max_l2, 32))


def derive_configs():
    """Derive unique (name, n_cores, tile_m, ic, oc, k_block, ppc) configs."""
    seen = {}
    configs = []
    for name, H, W, IC, OC in MODEL_LAYERS_1x1:
        M = H * W
        k_block, tile_m = choose_k_block(IC, OC, M)
        if tile_m < 4:
            print(f"WARNING: {name} ({IC}→{OC}) does not fit in L1!", file=sys.stderr)
            continue
        tile_m = min(tile_m, 256)

        if k_block > 0:
            # K-blocked path
            n_k_blocks = IC // k_block
            ppc = compute_ppc_kblocked(M, tile_m, IC, OC, k_block)
            key = (tile_m, IC, OC, k_block, ppc)
            xclbin_name = f"gemm_t{tile_m}_ic{IC}_oc{OC}_kb{k_block}_p{ppc}"
            calls = math.ceil(M / (32 * tile_m * ppc))
        else:
            # Non-K-blocked (original path, oc_block=OC)
            ppc = compute_ppc(M, tile_m, IC, OC)
            key = (tile_m, IC, OC, 0, ppc)
            xclbin_name = f"gemm_t{tile_m}_ic{IC}_oc{OC}_p{ppc}"
            calls = math.ceil(M / (32 * tile_m * ppc))

        if key in seen:
            print(f"  {name}: reuse {xclbin_name}", file=sys.stderr)
            continue

        kb_str = f"kb={k_block} ({n_k_blocks} K-blocks), " if k_block > 0 else ""
        print(f"  {xclbin_name}: {IC}→{OC}, {kb_str}ppc={ppc}, "
              f"{calls} calls for {name} {H}×{W}",
              file=sys.stderr)
        seen[key] = xclbin_name
        configs.append((xclbin_name, 32, tile_m, IC, OC, k_block, ppc))
    return configs


def build_kernel(build_dir):
    """Build rep_elan_bf16.o from unified kernels/ dir and copy into build_dir."""
    kernels_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")
    )
    obj_src = os.path.join(kernels_dir, "rep_elan_bf16.o")
    obj_dst = os.path.join(build_dir, "rep_elan_bf16.o")
    if not os.path.exists(obj_src):
        print(f"Building unified kernel in {kernels_dir}...", file=sys.stderr)
        result = subprocess.run(f"make -C {kernels_dir}", shell=True)
        if result.returncode != 0:
            print("FAIL: could not build rep_elan_bf16.o", file=sys.stderr)
            return False
    if (not os.path.exists(obj_dst)
            or os.path.getmtime(obj_src) > os.path.getmtime(obj_dst)):
        import shutil
        shutil.copy2(obj_src, obj_dst)
    return True


def build_one(name, n_cores, tile_m, ic, oc, k_block, ppc, build_dir):
    """Generate MLIR and compile one GEMM conv1x1 xclbin."""
    xclbin_path = os.path.join(build_dir, f"{name}.xclbin")
    if os.path.exists(xclbin_path):
        obj_path = os.path.join(build_dir, "rep_elan_bf16.o")
        if not os.path.exists(obj_path) or os.path.getmtime(xclbin_path) > os.path.getmtime(obj_path):
            print(f"  {name}: already built, skipping")
            return True

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aie2_gemm_conv1x1.py")
    mlir_path = os.path.join(build_dir, f"{name}.mlir")

    # Generate MLIR
    kb_arg = f" {k_block}" if k_block > 0 else " 0"
    cmd = f"python3 {script} {n_cores} {tile_m} {ic} {oc} {ppc}{kb_arg}"
    print(f"  {name}: MLIR...", end=" ", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL\n    {result.stderr.strip().split(chr(10))[-1]}")
        return False
    with open(mlir_path, 'w') as f:
        f.write(result.stdout)
    print("OK", end=" ", flush=True)

    # Compile xclbin
    cmd = (
        f"cd {build_dir} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts "
        f"--no-compile-host --no-xchesscc --no-xbridge "
        f"--xclbin-name={name}.xclbin --npu-insts-name={name}.bin {name}.mlir"
    )
    print("xclbin...", end=" ", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        errs = [l for l in result.stderr.split('\n') if 'error:' in l.lower()]
        print(f"FAIL\n    {errs[-1] if errs else 'unknown error'}")
        return False
    print("OK")
    return True


def main():
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    os.makedirs(build_dir, exist_ok=True)

    print("Deriving GEMM conv1x1 configs from MDV6 model...", file=sys.stderr)
    configs = derive_configs()

    # Filter by command line args
    if len(sys.argv) > 1:
        names = set(sys.argv[1:])
        configs = [c for c in configs if c[0] in names]

    print(f"\nBuilding kernel + {len(configs)} GEMM conv1x1 xclbins...")
    t0 = time.time()

    if not build_kernel(build_dir):
        print("Kernel compilation failed!")
        return False

    ok = fail = 0
    for cfg in configs:
        if build_one(*cfg, build_dir):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} OK, {fail} FAIL in {time.time() - t0:.0f}s")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
