#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run configurations sequentially and append results to OUT_FILE.
# Usage from a build directory: <srcdir>/sweep.sh ["M N K extra-make-args"]...
# Scale ITERS with the operation count to amortize launch overhead.

srcdir=$(dirname "$(realpath "$0")")
out=${OUT_FILE:-sweep_results.md}

# "M N K extra-make-args"
configs=(
  "2048 3072 256"
  "2048 3072 512"
  "2048 3072 1024"
  "2048 768 1024"
  "2048 1536 1024"
  "2048 6144 1024 WBUF=1"
  "512 3072 1024"
  "1024 3072 1024"
  "4096 3072 1024"
  "2048 3072 1024 ABCAST=0 ASPLIT=1"
  "2048 3072 1024 OUT=i32"
  "2048 3072 1024 C_REASM=0"
  "512 768 1024"
)
[ $# -gt 0 ] && configs=("$@")

[ -f "$out" ] || {
  echo "| M | N | K | options | GEMMs/launch | µs/GEMM (median) | TOPS (median) | TOPS (best) | % peak | check |"
  echo "|---|---|---|---|---|---|---|---|---|---|"
} > "$out"

for cfg in "${configs[@]}"; do
  read -r M N K extra <<< "$cfg"
  iters=$(python3 -c "print(max(4, min(50, int(5e-3 * 70e12 / (2 * $M * $N * $K)))))")
  echo "=== $M x $N x $K $extra (ITERS=$iters)"
  log=$(make -f "$srcdir/Makefile" run M=$M N=$N K=$K ITERS=$iters $extra 2>&1)
  med=$(sed -n 's/.*time\/GEMM: median \([0-9.]*\) us.*/\1/p' <<< "$log")
  tmed=$(sed -n 's/.*throughput: median \([0-9.]*\) TOPS, best.*/\1/p' <<< "$log")
  tbest=$(sed -n 's/.*best \([0-9.]*\) TOPS  (peak.*/\1/p' <<< "$log")
  pct=$(sed -n 's/.*-> \([0-9.]*\)% \/.*/\1/p' <<< "$log")
  check=$(grep -o -m1 "PASS\|FAIL\|TIMEOUT" <<< "$log")
  [ -z "$check" ] && check="build error: $(grep -i -m1 error <<< "$log" | cut -c1-80)"
  echo "| $M | $N | $K | ${extra:--} | $iters | ${med:--} | ${tmed:--} | ${tbest:--} | ${pct:--} | $check |" | tee -a "$out"
done
