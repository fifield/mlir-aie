#!/usr/bin/env python3
"""Exercise llama-style BO/static reuse on the existing rn3 vector kernel."""
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

from conv.aie2_rn3_pair_vector import ARENA_SIZE, FINAL_OFFSET, N_WEIGHT_SLOTS, WEIGHT_SLOT_SIZE, rn3_pair_vector
from conv.resident_xclbin_runner import ResidentXCLBinRunner
from conv.test_rn3_pair_vector_batch_hw import build_weights_and_oracle_inputs, compile_module
from conv.test_rn3_pair_vector_oneblock_hw import bf16_u16_to_f32
from conv.aie2_rn3_pair_vector import N_OC_BLOCKS, TILE_H, TILE_W, OC_BLOCK


def extract_output(raw: np.ndarray, n_patches: int) -> np.ndarray:
    arenas = raw.reshape(n_patches, ARENA_SIZE)
    got_patches = []
    for p in range(n_patches):
        blocks = []
        for ob in range(N_OC_BLOCKS):
            start = FINAL_OFFSET + ob * TILE_H * TILE_W * OC_BLOCK
            stop = start + TILE_H * TILE_W * OC_BLOCK
            blocks.append(bf16_u16_to_f32(arenas[p, start:stop]).reshape(TILE_H, TILE_W, OC_BLOCK))
        got_patches.append(np.concatenate(blocks, axis=2))
    return np.stack(got_patches, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patches", type=int, default=1)
    ap.add_argument("--workdir", default="conv/build_resident_rn3_vector")
    ap.add_argument("--stack-size", type=int, default=4096)
    args = ap.parse_args(argv)

    module = rn3_pair_vector(n_patches=args.n_patches, stack_size=args.stack_size)
    wd = Path(args.workdir) / f"p{args.n_patches}_st{args.stack_size}"
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built resident rn3 vector test mlir={mlir}")

    input_all, weight_all, expected = build_weights_and_oracle_inputs(args.n_patches)
    # Deliberately initialize output with nonzero junk; runner should not copy it to NPU.
    out0 = np.full(args.n_patches * ARENA_SIZE, 0x7BAD, dtype=np.uint16)
    out1 = np.full(args.n_patches * ARENA_SIZE, 0x1234, dtype=np.uint16)

    with ResidentXCLBinRunner(xclbin, insts) as runner:
        res0 = runner.run(
            input_all,
            weight_all,
            out0,
            bo_key="re6_rn3_vector_L0",
            output_indices={2},
            static_indices={1},
        )
        stats0 = runner.last_stats
        got0 = extract_output(res0[2], args.n_patches)
        max0 = float(np.max(np.abs(got0 - expected)))
        np.testing.assert_allclose(got0, expected, rtol=7e-2, atol=7e-2)

        # Second frame: same weights, same BO key. Only dynamic input should be written.
        res1 = runner.run(
            input_all,
            weight_all,
            out1,
            bo_key="re6_rn3_vector_L0",
            output_indices={2},
            static_indices={1},
        )
        stats1 = runner.last_stats
        got1 = extract_output(res1[2], args.n_patches)
        max1 = float(np.max(np.abs(got1 - expected)))
        np.testing.assert_allclose(got1, expected, rtol=7e-2, atol=7e-2)

    assert stats0 is not None and stats1 is not None
    print(
        "first_call "
        f"written={stats0.n_written} bytes={stats0.bytes_written} "
        f"write_ms={stats0.write_ms:.3f} kernel_ms={stats0.kernel_ms:.3f} "
        f"read_ms={stats0.read_ms:.3f} total_ms={stats0.total_ms:.3f} max_abs={max0:.6f}"
    )
    print(
        "second_call "
        f"written={stats1.n_written} bytes={stats1.bytes_written} "
        f"write_ms={stats1.write_ms:.3f} kernel_ms={stats1.kernel_ms:.3f} "
        f"read_ms={stats1.read_ms:.3f} total_ms={stats1.total_ms:.3f} max_abs={max1:.6f}"
    )
    expected_second_writes = 1
    assert stats1.n_written == expected_second_writes, stats1
    assert stats1.bytes_written == input_all.nbytes, stats1
    print(
        "PASS: resident runner reuses BOs and skips static weight write "
        f"weight_bytes={N_WEIGHT_SLOTS * WEIGHT_SLOT_SIZE * 2}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
