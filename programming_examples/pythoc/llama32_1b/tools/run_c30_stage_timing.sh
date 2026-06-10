#!/bin/bash
# C3.0 — time o_gemv_ffn per stage count using the stashed bisect ELFs.
# Run from build_peano. Stage 7 (baseline) last so the cache ends correct.
set -u
CACHE=decode_kernel_cache
STASH=c30_stage_stash
for s in 1 2 3 4 5 6 7; do
  echo "=== STAGES=$s ==="
  cp "$STASH/$s/o_gemv_ffn.elf" "$STASH/$s/o_gemv_ffn.npu.air.mlir" "$STASH/$s/o_gemv_ffn.aie.mlir" "$CACHE/"
  timeout 600 python3 ../tools/profile_c30.py --n-tokens 5 2>&1 | grep -E "o_gemv_ffn|attributed|tokens:" | head -5
done
