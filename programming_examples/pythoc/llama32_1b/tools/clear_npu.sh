#!/bin/bash
# Clear NPU wedges: rerun a known-good probe until it passes (max 4 tries).
cd "$(dirname "$0")/.."
for i in 1 2 3 4; do
  if timeout 120 python3 tools/test_c1_wedge.py --probe --stages 6 \
      --workdir build_peano/c1_wedge_cache >/dev/null 2>&1; then
    echo "NPU clean after $i tries"; exit 0
  fi
done
echo "NPU still wedged"; exit 1
