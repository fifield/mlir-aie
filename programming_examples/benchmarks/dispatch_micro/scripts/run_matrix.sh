#!/usr/bin/env bash
# run_matrix.sh - sweep the dispatch_micro benchmark matrix.
#
# Defaults to a small smoke slice. Override axes via env vars:
#   MECHANISMS, METRICS, DEVICES, TILES, BDS, TOPOLOGIES, AB
# Set BATCHED=1 to also run a runlist-batched variant of pure_dispatch.
#
# Build artifacts go under build/<key>/. Results stream to
# results/results.jsonl (one JSON object per line).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# ctrlpkt is in the matrix only when explicitly requested — its compile path
# needs the column-control-overlay pass we have not wired into Makefile yet.
: "${MECHANISMS:=baseline load_pdi_fw load_pdi_expanded}"
: "${METRICS:=pure_dispatch warm_reconfig cold_start}"
: "${DEVICES:=npu2_1col}"
: "${TILES:=1}"
: "${BDS:=2}"
: "${TOPOLOGIES:=linear}"
: "${AB:=0}"
: "${WARMUP:=10}"
: "${ITERS:=100}"
: "${COLD_RUNS:=30}"   # how many fresh processes for cold_start
: "${BATCHED:=0}"

mkdir -p results
RESULTS="${ROOT}/results/results.jsonl"
: > "${RESULTS}"

# Ensure bench binary is built.
if [[ ! -x "${ROOT}/bench" ]]; then
  echo "[run_matrix] Building bench..."
  make bench
fi

skip_combo() {
  local mech=$1 metric=$2
  # ctrlpkt has no cold_start (presumes a loaded context).
  if [[ "$mech" == "ctrlpkt" && "$metric" == "cold_start" ]]; then return 0; fi
  # baseline has no reconfig.
  if [[ "$mech" == "baseline" && "$metric" == "warm_reconfig" ]]; then return 0; fi
  return 1
}

for dev in $DEVICES; do
  for top in $TOPOLOGIES; do
    for t in $TILES; do
      for b in $BDS; do
        for mech in $MECHANISMS; do
          KEY="${mech}_${dev}_t${t}_b${b}_${top}"
          [[ "$AB" == "1" ]] && KEY="${KEY}_ab"
          BUILD_DIR="build/${KEY}"

          # Build (idempotent: make's deps short-circuit).
          echo "[run_matrix] BUILD ${KEY}"
          make MECH="${mech}" DEVICE="${dev}" TILES="${t}" BDS="${b}" \
               TOPOLOGY="${top}" AB="${AB}" all >/dev/null

          for metric in $METRICS; do
            if skip_combo "$mech" "$metric"; then
              echo "[run_matrix] SKIP  ${mech} × ${metric}"
              continue
            fi

            if [[ "$metric" == "cold_start" ]]; then
              for ((i = 0; i < COLD_RUNS; i++)); do
                ./bench --build-dir="${BUILD_DIR}" --mechanism="${mech}" \
                        --metric=cold_start --tiles="${t}" --bds="${b}" \
                        --json-out="${RESULTS}"
              done
              echo "[run_matrix] RUN   ${mech} cold_start × ${COLD_RUNS} processes"
            else
              ./bench --build-dir="${BUILD_DIR}" --mechanism="${mech}" \
                      --metric="${metric}" --tiles="${t}" --bds="${b}" \
                      --warmup="${WARMUP}" --iters="${ITERS}" \
                      --json-out="${RESULTS}"
              echo "[run_matrix] RUN   ${mech} ${metric}"

              if [[ "$BATCHED" == "1" && "$metric" == "pure_dispatch" ]]; then
                ./bench --build-dir="${BUILD_DIR}" --mechanism="${mech}" \
                        --metric=pure_dispatch --tiles="${t}" --bds="${b}" \
                        --warmup="${WARMUP}" --iters="${ITERS}" --batched \
                        --json-out="${RESULTS}"
                echo "[run_matrix] RUN   ${mech} pure_dispatch (batched)"
              fi
            fi
          done
        done
      done
    done
  done
done

echo "[run_matrix] DONE. Results: ${RESULTS}"
