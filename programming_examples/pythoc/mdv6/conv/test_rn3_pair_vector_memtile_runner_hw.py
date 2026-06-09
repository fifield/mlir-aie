#!/usr/bin/env python3
"""Smoke test reusable re6 memtile vector rn3-pair runner API."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conv.rn3_pair_vector_memtile_runner import (  # noqa: E402
    _conv1_valid_masks,
    close_runner,
    last_stats,
    pack_input_arenas_from_hwc,
    pack_vector_weight_slots_from_fused,
    run_re6_rn3_pair,
)
from conv.test_rn3_pair_vector_memtile_full_layer_hw import (  # noqa: E402
    expected_tiles_from_patches,
    extract_tiles,
    make_random_pair_weights,
)
from conv.test_rn3_pair_vector_oneblock_hw import (  # noqa: E402
    bf16_u16_to_f32,
    f32_to_bf16_u16,
)
from conv.rn3_pair_layout import scatter_tile_hwc_to_image  # noqa: E402


def fused_parts_to_u16(w, bw, bb):
    return np.concatenate([
        f32_to_bf16_u16(w.reshape(-1)),
        f32_to_bf16_u16(bw),
        f32_to_bf16_u16(bb),
    ]).astype(np.uint16)


def main():
    rng = np.random.default_rng(35791)
    image = rng.normal(0, 0.15, size=(40, 40, 48)).astype(np.float32)
    w1, bn1w, bn1b, w2, bn2w, bn2b = make_random_pair_weights(rng)
    fused_w1 = fused_parts_to_u16(w1, bn1w, bn1b)
    fused_w2 = fused_parts_to_u16(w2, bn2w, bn2b)

    # Build oracle from the same packed slots the runner will use.
    arenas, n_valid = pack_input_arenas_from_hwc(image)
    patches_bf = bf16_u16_to_f32(arenas.reshape(32, -1)[:n_valid, :12 * 12 * 48]).reshape(n_valid, 12, 12, 48)
    slots = pack_vector_weight_slots_from_fused(fused_w1, fused_w2)
    exp_tiles = expected_tiles_from_patches(patches_bf, slots, _conv1_valid_masks(40, 40))
    exp_image = scatter_tile_hwc_to_image(exp_tiles, 40, 40, 8, 8)

    out = run_re6_rn3_pair(torch.from_numpy(image).to(torch.bfloat16), fused_w1, fused_w2, bo_key="runner_smoke")
    got = out.float().numpy()
    stats = last_stats()
    max_abs = float(np.max(np.abs(got - exp_image)))
    print(f"shape={tuple(out.shape)} dtype={out.dtype}")
    if stats is not None:
        print(f"written={stats.n_written} bytes={stats.bytes_written} write_ms={stats.write_ms:.3f} kernel_ms={stats.kernel_ms:.3f} read_ms={stats.read_ms:.3f} total_ms={stats.total_ms:.3f}")
    print(f"max_abs={max_abs:.6f}")
    print(f"finite={np.isfinite(got).all()}")
    np.testing.assert_allclose(got, exp_image, rtol=7e-2, atol=7e-2)
    close_runner()
    print("PASS: reusable re6 rn3pair memtile runner")


if __name__ == "__main__":
    main()
