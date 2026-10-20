#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run each case at REPS=R1 and R2, TRIALS times, and report bandwidth as
# (R2-R1) * bytes / (t(R2)-t(R1)) to exclude fixed launch overhead.
# Usage from a build directory: <srcdir>/sweep.sh ["L COLS CH extra-make-args"]...

srcdir=$(dirname "$(realpath "$0")")
R1=${R1:-4}
R2=${R2:-36}
TRIALS=${TRIALS:-3}
cases=(
  "64 1 1" "256 1 1" "512 1 1" "1024 1 1" "4096 1 1"
  "64 1 2" "256 1 2" "512 1 2" "1024 1 2" "4096 1 2"
  "64 3 1" "256 3 1" "512 3 1" "1024 3 1" "4096 3 1"
  "64 3 2" "256 3 2" "512 3 2" "1024 3 2" "4096 3 2"
)
[ $# -gt 0 ] && cases=("$@")

run() { # L COLS CH REPS extra...
  local L=$1 C=$2 H=$3 R=$4
  shift 4
  make -s -f "$srcdir/Makefile" run L=$L COLS=$C CH=$H REPS=$R "$@" 2>&1 |
    tee -a sweep.log | awk '/^RESULT/ {split($2,b,"="); split($3,m,"="); print b[2], m[2]} /^FAIL/ {print "FAIL"}'
}

echo "| L (B) | cols | S2MM/col | channels | extra | t(R=$R1) us | t(R=$R2) us | GB/s slope, median of $TRIALS | slope min-max |"
echo "|---|---|---|---|---|---|---|---|---|"
for c in "${cases[@]}"; do
  set -- $c
  L=$1 C=$2 H=$3
  shift 3
  res=()
  for t in $(seq $TRIALS); do
    res+=("$(run $L $C $H $R1 "$@")" "$(run $L $C $H $R2 "$@")")
  done
  if printf '%s\n' "${res[@]}" | grep -q -v '^[0-9]'; then
    echo "| $L | $C | $H | $((C * H)) | $* | FAIL | FAIL | - | - |"
    continue
  fi
  python3 - "$L" "$C" "$H" "$*" "$R1" "$R2" "${res[@]}" <<'PY'
import statistics as st, sys
L, C, H, extra, R1, R2 = sys.argv[1:7]
R1, R2 = int(R1), int(R2)
v = [tuple(map(float, x.split())) for x in sys.argv[7:]]
b = v[0][0]
t1 = [x[1] for x in v[0::2]]
t2 = [x[1] for x in v[1::2]]
sl = sorted(b * (R2 - R1) / (y - x) / 1e3 for x, y in zip(t1, t2))
print(f"| {L} | {C} | {H} | {int(C)*int(H)} | {extra} | {st.median(t1):.0f} | {st.median(t2):.0f} | "
      f"{st.median(sl):.1f} | {sl[0]:.1f}-{sl[-1]:.1f} |")
PY
done
