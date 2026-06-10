# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""C3.0 — build o_gemv_ffn (c2_merged) at PYTHOC_C2_STAGES=1..6, stash ELFs.

Compiles ONLY o_gemv_ffn per stage value (compile_and_cache always rebuilds);
stashes elf + mlir into c30_stage_stash/<S>/. The cache's baseline stage-7
files are saved to c30_stage_stash/7/ first and restored at the end.

Run from build_peano:  python3 ../tools/build_c30_stage_elves.py
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernel_builder.cache import KernelCache
from kernel_builder import aie_ir_gen
from llama32_1b_weights import LlamaConfig

FILES = ["o_gemv_ffn.elf", "o_gemv_ffn.npu.air.mlir", "o_gemv_ffn.aie.mlir"]


def stash(cache_dir, dest):
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        src = os.path.join(cache_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, f))


def main():
    cache = KernelCache("decode_kernel_cache", verbose=False)
    cache_dir = str(cache.cache_dir)
    stash_root = "c30_stage_stash"
    stash(cache_dir, os.path.join(stash_root, "7"))  # baseline

    cfg = LlamaConfig()
    for s in [1, 2, 3, 4, 5, 6]:
        os.environ["PYTHOC_C2_STAGES"] = str(s)
        t0 = time.time()
        print(f"=== building stage {s} ===", flush=True)
        cache.compile_and_cache(
            "o_gemv_ffn",
            aie_ir_gen.build_o_gemv_ffn_ir(cfg.emb_dim, cfg.hidden_dim),
            instance_name="o_gemv_ffn",
        )
        stash(cache_dir, os.path.join(stash_root, str(s)))
        print(f"=== stage {s} done in {time.time() - t0:.0f}s ===", flush=True)

    # restore baseline
    for f in FILES:
        shutil.copy2(os.path.join(stash_root, "7", f), os.path.join(cache_dir, f))
    print("baseline restored")


if __name__ == "__main__":
    main()
