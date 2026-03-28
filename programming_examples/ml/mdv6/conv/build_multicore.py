#!/usr/bin/env python3
"""Build all multicore xclbins needed for the full MDV6 model.
Generates MLIR and compiles for each unique (tile, ic, oc, ks, stride) config."""
import os, sys, subprocess, time

# All unique configs from test_full_model.py:
# (name, n_cores, tile_h, tile_w, ic, oc_block, kernel_size, stride)
CONFIGS = [
    # Conv0/Conv1 (stride-2 stem)
    ("mc_ftconv0",     32, 24, 24,   3, 32, 3, 2),
    ("mc_ftconv1",     32, 12, 12,  32, 16, 3, 2),
    # ELAN2 sub-layers
    ("mc_elan_c1",     32,  8,  8,  64, 64, 1, 1),  # 1x1 64→64
    ("mc_elan_c3",     32, 16, 16,  32, 32, 3, 1),  # 3x3 32→32
    ("mc_elan_c4",     32,  8,  8, 128, 64, 1, 1),  # 1x1 128→64
    # AConv layers (stride-2)
    ("mc_aconv3",      32,  8,  8,  64, 16, 3, 2),  # 64→128 (oc_block=16)
    ("mc_aconv5",      32,  4,  4,  96,  8, 3, 2),  # 96→192 (oc_block=8)
    ("mc_aconv7",      32,  4,  4, 128,  4, 3, 2),  # 128→256 (oc_block=4)
    ("mc_aconv16",     32,  4,  4,  64,  8, 3, 2),  # 64→96 (oc_block=8)
    ("mc_aconv19",     32,  4,  4,  96,  4, 3, 2),  # 96→128 (oc_block=4)
    # RepNCSPELAN4 (80x80, 128ch)
    ("mc_re4_c1",      32, 10, 10, 128, 64, 1, 1),  # conv1: 128→128
    ("mc_re4_c3",      32, 12, 12,  64, 16, 3, 1),  # conv3x3: 64→64
    ("mc_re4_c4",      32,  8,  8, 256, 32, 1, 1),  # conv4: 256→128
    ("mc_re4_rn1",     32, 16, 16,  64, 32, 1, 1),  # rn 1x1: 64→32
    ("mc_re4_rn3",     32, 16, 16,  32, 32, 3, 1),  # rn 3x3: 32→32 (trn3=16, orn3=32)
    # RepNCSPELAN6 (40x40, 192ch)
    ("mc_re6_c1",      32,  8,  8, 192, 32, 1, 1),
    ("mc_re6_c3",      32,  8,  8,  96, 16, 3, 1),
    ("mc_re6_c4",      32,  4,  4, 384, 32, 1, 1),
    ("mc_re6_rn1",     32, 10, 10,  96, 48, 1, 1),
    ("mc_re6_rn3",     32,  8,  8,  48, 16, 3, 1),
    ("mc_re6_rnm",     32,  8,  8,  96, 48, 1, 1),  # RepNCSP merge: 2*neck=96→oc=96
    # RepNCSPELAN8 (20x20, 256ch)
    ("mc_re8_c1",      32,  4,  4, 256, 32, 1, 1),
    ("mc_re8_c3",      32,  4,  4, 128, 16, 3, 1),
    ("mc_re8_c4",      32,  4,  4, 512, 16, 1, 1),
    ("mc_re8_rn1",     32,  8,  8, 128, 64, 1, 1),
    ("mc_re8_rn3",     32,  8,  8,  64, 16, 3, 1),
    ("mc_re8_rnm",     32,  4,  4, 256, 32, 1, 1),
    # SPP9
    ("mc_spp_c1",      32,  4,  4, 256, 32, 1, 1),
    # Neck: re12 (uses re6 configs mostly, plus:)
    ("mc_re12_c1",     32,  4,  4, 448, 32, 1, 1),
    # Neck: re15 (uses re4 configs mostly, plus:)
    ("mc_re15_c1",     32,  6,  6, 320, 32, 1, 1),
    ("mc_re15_c4",     32,  8,  8, 256, 32, 1, 1),
    ("mc_re15_rnm",    32,  8,  8, 128, 64, 1, 1),
    # Head P4: re18 (uses re6 configs mostly, plus:)
    ("mc_re18_c1",     32,  4,  4, 288, 32, 1, 1),
    # Head P5: re21 (uses re8 configs mostly, plus:)
    ("mc_re21_c1",     32,  4,  4, 384, 32, 1, 1),
]


def build_one(name, n_cores, tile_h, tile_w, ic, oc, ks, stride, build_dir):
    """Generate MLIR and compile one multicore xclbin."""
    mlir_path = os.path.join(build_dir, f"{name}.mlir")
    xclbin_path = os.path.join(build_dir, f"{name}.xclbin")
    insts_path = os.path.join(build_dir, f"{name}.bin")

    if os.path.exists(xclbin_path):
        print(f"  {name}: already built, skipping")
        return True

    script = os.path.join(os.path.dirname(__file__), "aie2_multicore.py")
    cmd_mlir = f"python3 {script} {n_cores} {tile_h} {tile_w} {ic} {oc} {ks} {stride} 1"
    print(f"  {name}: generating MLIR...", end=" ", flush=True)
    result = subprocess.run(cmd_mlir, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL\n    {result.stderr.strip().split(chr(10))[-1]}")
        return False
    with open(mlir_path, 'w') as f:
        f.write(result.stdout)
    print("OK", end=" ", flush=True)

    cmd_xclbin = (
        f"cd {build_dir} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts "
        f"--no-compile-host --no-xchesscc --no-xbridge "
        f"--xclbin-name={name}.xclbin --npu-insts-name={name}.bin {name}.mlir"
    )
    print("compiling...", end=" ", flush=True)
    result = subprocess.run(cmd_xclbin, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL\n    {result.stderr.strip().split(chr(10))[-1]}")
        return False
    print("OK")
    return True


def main():
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_dir, exist_ok=True)

    # Filter by command line args if provided
    configs = CONFIGS
    if len(sys.argv) > 1:
        names = set(sys.argv[1:])
        configs = [c for c in CONFIGS if c[0] in names]

    print(f"Building {len(configs)} multicore xclbins...")
    t0 = time.time()
    ok = 0
    fail = 0
    for cfg in configs:
        name = cfg[0]
        if build_one(*cfg, build_dir):
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t0
    print(f"\nDone: {ok} OK, {fail} FAIL in {elapsed:.0f}s")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
