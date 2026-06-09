#!/usr/bin/env python3
"""Full-layer wrapper smoke for 32-core memtile vector rn3-pair.

This is the integration scaffold before touching `run_rn_mc`:

1. pack a full HWC image into row-major 12x12x48 halo patches,
2. pad the 25 re6 tiles to the 32-worker memtile kernel capacity,
3. pack full 48->48 rn3 pair weights into the 12 vector slots,
4. run one resident xclbin dispatch,
5. discard padded outputs and scatter 8x8x48 tiles back to HWC,
6. compare against the CPU oracle that matches the vector kernel's bf16 rounding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aie.utils.compile import compile_mlir_module

from conv.aie2_rn3_pair_vector_memtile import rn3_pair_vector_memtile  # noqa: E402
from conv.aie2_rn3_pair_vector_ocb import (  # noqa: E402
    ARENA_SIZE,
    FINAL_OFFSET,
    IC,
    INPUT_SIZE,
    MASK_OFFSET,
    MASK_SIZE,
    MID_BLOCK,
    N_MID_BLOCKS,
    N_OC_BLOCKS,
    N_WEIGHT_SLOTS,
    OC_BLOCK,
    TILE_H,
    TILE_W,
    W1_SIZE,
    W2_SIZE,
    WEIGHT_SLOT_SIZE,
)
from conv.resident_xclbin_runner import ResidentXCLBinRunner  # noqa: E402
from conv.rn3_pair_vector_memtile_runner import _conv1_valid_masks  # noqa: E402
from conv.rn3_pair_layout import (  # noqa: E402
    pack_rn3_pair_input_patches,
    scatter_tile_hwc_to_image,
)
from conv.test_rn3_pair_vector_oneblock_hw import (  # noqa: E402
    bf16_u16_to_f32,
    conv3x3_bnsilu_cpu,
    f32_to_bf16_u16,
    pack_3x3_weights_u16,
    silu,
    unpack_packed_3x3_weights_f32,
)


def compile_module(module, workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    mlir_path = workdir / "kernel.mlir"
    with open(mlir_path, "w", encoding="utf-8") as f:
        print(module, file=f)
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(workdir / "insts.bin"),
        xclbin_path=str(workdir / "final.xclbin"),
        work_dir=str(workdir),
        verbose=False,
    )
    return workdir / "final.xclbin", workdir / "insts.bin", mlir_path


def make_random_pair_weights(rng: np.random.Generator):
    w1 = rng.normal(0, 0.035, size=(MID_BLOCK * N_MID_BLOCKS, IC, 3, 3)).astype(np.float32)
    bn1w = rng.normal(1.0, 0.02, size=(MID_BLOCK * N_MID_BLOCKS,)).astype(np.float32)
    bn1b = rng.normal(0.0, 0.01, size=(MID_BLOCK * N_MID_BLOCKS,)).astype(np.float32)
    w2 = rng.normal(0, 0.035, size=(OC_BLOCK * N_OC_BLOCKS, MID_BLOCK * N_MID_BLOCKS, 3, 3)).astype(np.float32)
    bn2w = rng.normal(1.0, 0.02, size=(OC_BLOCK * N_OC_BLOCKS,)).astype(np.float32)
    bn2b = rng.normal(0.0, 0.01, size=(OC_BLOCK * N_OC_BLOCKS,)).astype(np.float32)
    return w1, bn1w, bn1b, w2, bn2w, bn2b


def pack_vector_weight_slots(w1, bn1w, bn1b, w2, bn2w, bn2b) -> np.ndarray:
    slots = np.zeros((N_WEIGHT_SLOTS, WEIGHT_SLOT_SIZE), dtype=np.uint16)
    wi = 0
    for mb in range(N_MID_BLOCKS):
        lo = mb * MID_BLOCK
        hi = lo + MID_BLOCK
        packed = pack_3x3_weights_u16(w1[lo:hi], bn1w[lo:hi], bn1b[lo:hi])
        if packed.size != W1_SIZE:
            raise AssertionError((packed.size, W1_SIZE))
        slots[wi, :W1_SIZE] = packed
        wi += 1
    for ob in range(N_OC_BLOCKS):
        olo = ob * OC_BLOCK
        ohi = olo + OC_BLOCK
        packed = pack_3x3_weights_u16(w2[olo:ohi, :], bn2w[olo:ohi], bn2b[olo:ohi])
        if packed.size != W2_SIZE:
            raise AssertionError((packed.size, W2_SIZE))
        slots[wi, :W2_SIZE] = packed
        wi += 1
    if wi != N_WEIGHT_SLOTS:
        raise AssertionError((wi, N_WEIGHT_SLOTS))
    return slots.reshape(-1)


def expected_tiles_from_patches(patches_bf: np.ndarray, weight_slots: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
    slots = weight_slots.reshape(N_WEIGHT_SLOTS, WEIGHT_SLOT_SIZE)
    w1_slots = [slots[i, :W1_SIZE].copy() for i in range(N_MID_BLOCKS)]
    w2_slots = []
    wi = N_MID_BLOCKS
    for ob in range(N_OC_BLOCKS):
        w2_slots.append(slots[wi, :W2_SIZE].copy())
        wi += 1

    expected = []
    if masks is None:
        masks = np.ones((patches_bf.shape[0], (TILE_H + 2) * (TILE_W + 2)), dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32).reshape(patches_bf.shape[0], TILE_H + 2, TILE_W + 2)
    for patch_idx, patch in enumerate(patches_bf):
        mid_parts = []
        for mb in range(N_MID_BLOCKS):
            mid_f32 = conv3x3_bnsilu_cpu(patch, w1_slots[mb], TILE_H + 2, TILE_W + 2, IC, MID_BLOCK)
            mid_f32 = mid_f32.copy()
            mid_f32[masks[patch_idx] < 0.5, :] = 0.0
            mid_parts.append(bf16_u16_to_f32(f32_to_bf16_u16(mid_f32.reshape(-1))).reshape(TILE_H + 2, TILE_W + 2, MID_BLOCK))
        mid_full = np.concatenate(mid_parts, axis=2)
        exp_blocks = []
        for ob in range(N_OC_BLOCKS):
            w2_full, bn2w, bn2b = unpack_packed_3x3_weights_f32(w2_slots[ob], OC_BLOCK, MID_BLOCK * N_MID_BLOCKS)
            out = np.zeros((TILE_H, TILE_W, OC_BLOCK), dtype=np.float32)
            for oh in range(TILE_H):
                for ow in range(TILE_W):
                    mid_patch = mid_full[oh:oh + 3, ow:ow + 3, :]
                    for co in range(OC_BLOCK):
                        acc = np.float32(0.0)
                        for ci in range(MID_BLOCK * N_MID_BLOCKS):
                            for kh in range(3):
                                for kw in range(3):
                                    acc += np.float32(mid_patch[kh, kw, ci] * w2_full[co, ci, kh, kw])
                        out[oh, ow, co] = silu(acc * bn2w[co] + bn2b[co])
            exp_blocks.append(out)
        expected.append(np.concatenate(exp_blocks, axis=2))
    return np.stack(expected, axis=0)


def pack_input_arenas(image_hwc_f32: np.ndarray, n_cores: int):
    patches = pack_rn3_pair_input_patches(image_hwc_f32, TILE_H, TILE_W, halo=2)
    n_valid = patches.shape[0]
    if n_valid > n_cores:
        raise ValueError(f"{n_valid} patches exceed n_cores={n_cores}")
    patch_u16 = f32_to_bf16_u16(patches.reshape(-1))
    patches_bf = bf16_u16_to_f32(patch_u16).reshape(patches.shape)
    masks = _conv1_valid_masks(image_hwc_f32.shape[0], image_hwc_f32.shape[1])
    arenas = np.zeros((n_cores, ARENA_SIZE), dtype=np.uint16)
    arenas[:n_valid, :INPUT_SIZE] = patch_u16.reshape(n_valid, INPUT_SIZE)
    arenas[:n_valid, MASK_OFFSET:MASK_OFFSET + MASK_SIZE] = f32_to_bf16_u16(masks.reshape(-1)).reshape(n_valid, MASK_SIZE)
    if n_valid < n_cores:
        arenas[n_valid:, MASK_OFFSET:MASK_OFFSET + MASK_SIZE] = f32_to_bf16_u16(np.ones(MASK_SIZE, dtype=np.float32))
    return arenas.reshape(-1), patches_bf, n_valid, masks


def extract_tiles(raw_flat: np.ndarray, n_valid: int) -> np.ndarray:
    raw = raw_flat.reshape(-1, ARENA_SIZE)
    got = []
    for p in range(n_valid):
        blocks = []
        for ob in range(N_OC_BLOCKS):
            start = FINAL_OFFSET + ob * TILE_H * TILE_W * OC_BLOCK
            stop = start + TILE_H * TILE_W * OC_BLOCK
            blocks.append(bf16_u16_to_f32(raw[p, start:stop]).reshape(TILE_H, TILE_W, OC_BLOCK))
        got.append(np.concatenate(blocks, axis=2))
    return np.stack(got, axis=0)


def run_full_layer(xclbin: Path, insts: Path, image_hwc: np.ndarray, weight_slots: np.ndarray, *, n_cores: int):
    input_arenas, patches_bf, n_valid, masks = pack_input_arenas(image_hwc, n_cores)
    expected_tiles = expected_tiles_from_patches(patches_bf, weight_slots, masks)
    out_arg = np.zeros(n_cores * ARENA_SIZE, dtype=np.uint16)
    with ResidentXCLBinRunner(xclbin, insts) as runner:
        res1 = runner.run(input_arenas, weight_slots, out_arg, bo_key=f"rn3vm_full_{n_cores}", output_indices={2}, static_indices={1})
        s1 = runner.last_stats
        got1 = extract_tiles(res1[2], n_valid)
        max1 = float(np.max(np.abs(got1 - expected_tiles)))
        print(f"resident_first written={s1.n_written} bytes={s1.bytes_written} write_ms={s1.write_ms:.3f} kernel_ms={s1.kernel_ms:.3f} read_ms={s1.read_ms:.3f} total_ms={s1.total_ms:.3f} max_abs={max1:.6f}")
        res2 = runner.run(input_arenas, weight_slots, out_arg, bo_key=f"rn3vm_full_{n_cores}", output_indices={2}, static_indices={1})
        s2 = runner.last_stats
        got_tiles = extract_tiles(res2[2], n_valid)
        max2 = float(np.max(np.abs(got_tiles - expected_tiles)))
        print(f"resident_second written={s2.n_written} bytes={s2.bytes_written} write_ms={s2.write_ms:.3f} kernel_ms={s2.kernel_ms:.3f} read_ms={s2.read_ms:.3f} total_ms={s2.total_ms:.3f} max_abs={max2:.6f}")
    got_image = scatter_tile_hwc_to_image(got_tiles, image_hwc.shape[0], image_hwc.shape[1], TILE_H, TILE_W)
    expected_image = scatter_tile_hwc_to_image(expected_tiles, image_hwc.shape[0], image_hwc.shape[1], TILE_H, TILE_W)
    return got_tiles, expected_tiles, got_image, expected_image, s2.total_ms, s2.kernel_ms


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=40)
    p.add_argument("--width", type=int, default=40)
    p.add_argument("--n-cores", type=int, default=32)
    p.add_argument("--seed", type=int, default=24680)
    p.add_argument("--workdir", default="conv/build_rn3_pair_vector_memtile_full_layer")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)

    module = rn3_pair_vector_memtile(n_cores=args.n_cores)
    wd = Path(args.workdir) / f"cores{args.n_cores}"
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built rn3_pair_vector_memtile_full_layer n_cores={args.n_cores} mlir={mlir}")
    if args.build_only:
        return 0

    rng = np.random.default_rng(args.seed)
    image = rng.normal(0, 0.15, size=(args.height, args.width, IC)).astype(np.float32)
    weights = pack_vector_weight_slots(*make_random_pair_weights(rng))
    got_tiles, exp_tiles, got_image, exp_image, total_ms, kernel_ms = run_full_layer(
        xclbin, insts, image, weights, n_cores=args.n_cores
    )
    tile_max = float(np.max(np.abs(got_tiles - exp_tiles)))
    image_max = float(np.max(np.abs(got_image - exp_image)))
    print(f"tiles_shape={got_tiles.shape}")
    print(f"image_shape={got_image.shape}")
    print(f"tile_max_abs={tile_max:.6f}")
    print(f"image_max_abs={image_max:.6f}")
    print(f"kernel_ms={kernel_ms:.3f}")
    print(f"total_ms={total_ms:.3f}")
    np.testing.assert_allclose(got_tiles, exp_tiles, rtol=7e-2, atol=7e-2)
    np.testing.assert_allclose(got_image, exp_image, rtol=7e-2, atol=7e-2)
    print("PASS: full-layer memtile vector rn3pair wrapper matches CPU oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
