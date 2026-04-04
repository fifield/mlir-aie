#!/usr/bin/env python3
"""Build all GEMM conv1x1 xclbins needed for the full MDV6 model.

Generates MLIR via aie2_gemm_conv1x1.py and compiles for each unique
(tile_m, ic, oc_block) config.  The kernel .o is compiled once and shared.

Weight layout: [ic/8, oc/8, 8, 8] (pre-transposed, matching conv_bf16_vec.cc).
Uses mmul<4,8,8> — no BFP16 emulation flag needed.
"""
import math
import os
import subprocess
import sys
import time

L1_BYTES = 65536
STACK_BYTES = 8192
AVAIL = L1_BYTES - STACK_BYTES


def compute_tile_m(ic, oc_block):
    """Max tile_m (mult of 4) that fits in L1 with depth=1 FIFOs."""
    wt_bytes = (ic * oc_block + 2 * oc_block) * 2
    remaining = AVAIL - wt_bytes
    if remaining <= 0:
        return 0
    per_pixel = (ic + oc_block) * 2  # input + output per pixel
    max_tm = remaining // per_pixel
    return (max_tm // 4) * 4


def choose_oc_block(ic, oc):
    """Choose largest oc_block (mult of 16) that fits with reasonable tile_m."""
    for ob in [oc, 128, 64, 48, 32, 16]:
        if ob > oc or oc % ob != 0:
            continue
        tm = compute_tile_m(ic, ob)
        if tm >= 16:
            return ob, tm
    return None, None


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


def derive_configs():
    """Derive unique (name, n_cores, tile_m, ic, oc_block) configs."""
    seen = {}
    configs = []
    for name, H, W, IC, OC in MODEL_LAYERS_1x1:
        oc_block, tile_m = choose_oc_block(IC, OC)
        if oc_block is None:
            print(f"WARNING: {name} ({IC}→{OC}) does not fit in L1!", file=sys.stderr)
            continue
        # Cap tile_m for practical buffer sizes
        tile_m = min(tile_m, 256)
        M = H * W
        key = (tile_m, IC, oc_block)
        # Name by actual config — matches run_tiled_mc.py's actual_name
        xclbin_name = f"gemm_t{tile_m}_ic{IC}_oc{oc_block}"
        if key in seen:
            print(f"  {name}: reuse {xclbin_name}", file=sys.stderr)
            continue
        n_oc_passes = OC // oc_block
        pixels_per_call = 32 * tile_m
        n_spatial = math.ceil(M / pixels_per_call)
        total_calls = n_oc_passes * n_spatial
        wt_kb = (IC * oc_block + 2 * oc_block) * 2 / 1024
        print(f"  {xclbin_name}: {IC}→{oc_block}, "
              f"wt={wt_kb:.1f}KB, {total_calls} calls for {name} {H}×{W}",
              file=sys.stderr)
        seen[key] = xclbin_name
        configs.append((xclbin_name, 32, tile_m, IC, oc_block))
    return configs


def build_kernel(build_dir):
    """Compile gemm_conv1x1_bf16.cc with Peano (once)."""
    obj_path = os.path.join(build_dir, "gemm_conv1x1_bf16.o")
    if os.path.exists(obj_path):
        return True
    src_dir = os.path.dirname(os.path.abspath(__file__))
    makefile_dir = src_dir
    result = subprocess.run(
        f"make -C {makefile_dir} build/gemm_conv1x1_bf16.o",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Kernel compile FAIL:\n{result.stderr[-500:]}", file=sys.stderr)
        return False
    return True


def build_one(name, n_cores, tile_m, ic, oc_block, build_dir):
    """Generate MLIR and compile one GEMM conv1x1 xclbin."""
    xclbin_path = os.path.join(build_dir, f"{name}.xclbin")
    if os.path.exists(xclbin_path):
        print(f"  {name}: already built, skipping")
        return True

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aie2_gemm_conv1x1.py")
    mlir_path = os.path.join(build_dir, f"{name}.mlir")

    # Generate MLIR
    cmd = f"python3 {script} {n_cores} {tile_m} {ic} {oc_block} 1"
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
