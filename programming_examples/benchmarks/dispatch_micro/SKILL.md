---
name: dispatch-micro-bench
description: Run the dispatch_micro NPU configuration/dispatch-latency micro-benchmark suite and produce a comprehensive report. Use this skill whenever the user wants to (re)run the dispatch_micro benchmarks, regenerate the report, sweep a new axis, or compare NPU dispatch mechanisms (baseline xclbin, load_pdi firmware, load_pdi expanded, ctrlpkt, runlist batching, cold start). Trigger phrases include "run the dispatch micro benchmarks", "scale up the sweep", "regenerate the report", "benchmark NPU dispatch", "rerun on the whole array", or any ask that mentions dispatch_micro, REPORT.md in this directory, results.jsonl, or the per-mechanism latency tables.
---

# dispatch_micro benchmark + report skill

This skill runs the NPU dispatch-latency micro-benchmark suite in
`programming_examples/benchmarks/dispatch_micro/` and produces a markdown
report. It is **project-local** — it lives alongside the code it drives.

## What this benchmark actually measures (read this first)

Five candidate mechanisms across two XRT runtime families. **Only three of
them are wired into `bench.cpp` today.** ctrlpkt is built but not dispatched
(v2). See "Known limits" below.

| mech                 | family   | aiecc flags                                                                           | runtime model                          |
|----------------------|----------|---------------------------------------------------------------------------------------|----------------------------------------|
| `baseline`           | xclbin   | `--aie-generate-xclbin --aie-generate-npu-insts`                                       | `xrt::xclbin` + `xrt::kernel`          |
| `load_pdi_fw`        | full ELF | `--generate-full-elf`                                                                  | `xrt::elf` + `xrt::ext::kernel`        |
| `load_pdi_expanded`  | full ELF | `--generate-full-elf --expand-load-pdis`                                               | `xrt::elf` + `xrt::ext::kernel`        |
| `ctrlpkt` *(v2)*     | xclbin   | `--aie-generate-xclbin --aie-generate-ctrlpkt`                                         | not yet plumbed in `bench.cpp`         |

**CRITICAL — PDI cache caveat.** The runtime/driver/firmware stack caches
PDI loads at the PDI level. Every dispatch in v1 targets the same PDI, so
`load_pdi_fw`'s numbers reflect the **cache-hit** path, not actual PDI
load cost. Only `load_pdi_expanded` is doing uncached reconfig work in v1
(its expansion replaces the cacheable opcode with raw write32/blockwrite).
Real load_pdi cost requires A↔B alternation — that is v2.

Always restate this caveat near the top of any new REPORT.md you write.

## Axes the harness understands

- `mechanism`: `baseline | load_pdi_fw | load_pdi_expanded` (ctrlpkt: skip in v1)
- `device`: `npu2_1col | npu2_4col | npu2` — pick by column count
- `tiles` (= **cols**): shim columns used; 1, 2, 4, or 8
- `rows_per_col`: compute tiles per column (1..4). Whole array = `tiles=8 × rows_per_col=4` = 32 cores.
- `bds`: BDs per shim DMA per direction (2, 4, 8). **`bds=8` reproducibly crashes firmware for `load_pdi_*` at `tiles ∈ {2, 4}` — skip those cells.**
- `topology`: `linear | branch | hop` — only `linear` is exercised in v1 reports.
- `ab`: 0/1 (build emits two `@device` regions; `bench.cpp` doesn't dispatch them yet).

## Metrics

- `pure_dispatch` — pre-loaded context, hot loop of dispatches. Default warmup=10, iters=100.
- `cold_start` — one fresh process per sample; emits a phase breakdown (`load`, `register`, `kernel`, `first_dispatch`). Drive 30 fresh processes per cell for stats.
- `pure_dispatch --batched --batch-size=N` — same workload but via `xrt::runlist::execute()`; per-dispatch latency = total/N.
- `warm_reconfig` — *currently identical to `pure_dispatch` for the load_pdi mechanisms* because the generator emits `npu_load_pdi(device_ref=@main)` at the top of every runtime sequence. Don't report it as a distinct metric in v1.

## Skill workflow

### 1) Set up the environment

```bash
cd /home/jfifield/npu-dev/mlir-aie/programming_examples/benchmarks/dispatch_micro
# env.sh is one level up from mlir-aie; assume already sourced. If aiecc isn't on PATH:
# source ../../../../../env.sh
make clean-all      # only if a clean run is wanted; otherwise rely on Make deps
make bench          # builds the host driver into ./bench
```

`bench` links against XRT — set `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib` when
running it (the `run_matrix.sh` helper does this automatically).

### 2) Confirm scope with the user before launching the matrix

Ask, briefly, which axes they want to sweep this round. Defaults that
usually make sense:

- **Smoke (~5 min, ~3 builds, ~3 dispatch rows):** `make smoke` then run
  `pure_dispatch` once per built mechanism. Use this if the user just wants
  to verify the harness works.
- **Latency-vs-tiles row (~12 min, 36 builds, 36 rows):** sweep
  `mechanism × tiles ∈ {1,2,4,8} × bds ∈ {2,4,8}` at `rows_per_col=1`.
  Mirrors §1 of REPORT.md.
- **Cold start (~8 min, 300+ rows):** 30 fresh processes per `(mechanism,
  tiles, bds)` cell over a small subset (e.g. `tiles ∈ {1,4} × bds ∈ {2,8}`).
  Mirrors §2 of REPORT.md.
- **Batched (~3 min, 24 rows):** `batch_size ∈ {1,4,16,64}` at two tile
  counts × 3 mechanisms. Mirrors §3.
- **Whole array (~12 min, 36 builds, 36 rows):** `rows_per_col ∈ {1,2,4}
  × tiles ∈ {1,2,4,8}` at `bds=2, linear`. Mirrors the v1 §"Whole-array
  sweep".

If the user says "run everything" or "comprehensive report", do all five.

### 3) Build and run

Always **build first, run second**. Builds are idempotent and cached by
`KEY := $(MECH)_$(DEVICE)_t$(TILES)_r$(ROWS_PER_COL)_b$(BDS)_$(TOPOLOGY)…`.

**Build a single cell:**
```bash
make MECH=baseline DEVICE=npu2_1col TILES=1 ROWS_PER_COL=1 BDS=2 TOPOLOGY=linear
```

**Run a single dispatch row:**
```bash
KEY="baseline_npu2_1col_t1_r1_b2_linear"
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib ./bench \
  --build-dir=build/$KEY --mechanism=baseline --metric=pure_dispatch \
  --tiles=1 --rows-per-col=1 --bds=2 \
  --warmup=10 --iters=100 \
  --json-out=results/results.jsonl
```

**Driving a matrix from bash (skip known-bad cells):**
```bash
for mech in baseline load_pdi_fw load_pdi_expanded; do
  for t in 1 2 4 8; do
    for b in 2 4 8; do
      # Skip the v1 firmware-crash corner.
      if [ "$mech" != "baseline" ] && [ "$t" -ne 1 ] && [ "$t" -ne 8 ] && [ "$b" -eq 8 ]; then
        echo "SKIP $mech t=$t b=$b (known-bad)"; continue
      fi
      if [ $t -eq 1 ]; then DEV=npu2_1col;
      elif [ $t -le 4 ]; then DEV=npu2_4col;
      else DEV=npu2; fi
      KEY="${mech}_${DEV}_t${t}_r1_b${b}_linear"
      make MECH=$mech DEVICE=$DEV TILES=$t ROWS_PER_COL=1 BDS=$b TOPOLOGY=linear >/dev/null
      LD_LIBRARY_PATH=/opt/xilinx/xrt/lib ./bench \
        --build-dir=build/$KEY --mechanism=$mech --metric=pure_dispatch \
        --tiles=$t --bds=$b --warmup=10 --iters=100 \
        --json-out=results/results.jsonl
    done
  done
done
```

`scripts/run_matrix.sh` is the canonical driver; use it when the axes match
its defaults, drop to bash loops when they don't.

### 4) Generate plots

```bash
mkdir -p results/plots
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib python3 scripts/plot.py results/results.jsonl
```

Produces these PNGs under `results/plots/`:
- `pure_dispatch_vs_tiles.png`, `pure_dispatch_vs_tiles_bds{2,4,8}.png`
- `pure_dispatch_vs_bds.png`, `pure_dispatch_vs_bds_t{1,4,8}.png`
- `pure_dispatch_vs_total_tiles.png` (whole-array view: cols × rows on log₂ x)
- `pure_dispatch_batched.png` (per-dispatch latency vs batch size)
- `cold_start_breakdown.png` (stacked bars per mechanism)

### 5) Write or update REPORT.md

Append a section per axis swept. Always include:
- **A legend block** explaining `t` / `b` / `r` / `bs` symbols (the user has
  asked for legends in past iterations — include them every time).
- **The PDI-cache caveat** front-and-center if `load_pdi_fw` numbers are
  presented (they always need that context).
- **Tables** with µs (p50) cells; mark cells that crashed as ✗ with a
  footnote pointing at "Failures" section.
- **Headline findings** in 3-5 plain-language bullets.
- **Caveats** section listing what wasn't measured (e.g. "this run only
  covered topology=linear; branch/hop untested").
- **Cross-references to the PNGs** under `results/plots/`.

A useful Python one-liner to extract a clean p50 table from
`results/results.jsonl`:

```bash
python3 <<'PY'
import json, statistics
rows = [json.loads(l) for l in open("results/results.jsonl")]
for mech in ("baseline","load_pdi_fw","load_pdi_expanded"):
    for t in (1,2,4,8):
        for b in (2,4,8):
            hits = [r for r in rows
                    if r["mechanism"]==mech and r["metric"]=="pure_dispatch"
                    and r["tiles"]==t and r["bds"]==b and r["batch_size"]==1
                    and r.get("rows_per_col",1)==1 and r.get("iters",0)==100]
            v = f"{hits[0]['ns']['p50']/1000:.1f}" if hits else "—"
            print(f"{mech:<22} t={t} b={b}: {v} µs")
PY
```

For cold_start tables, aggregate across the 30 fresh processes with
`statistics.median([r["cold_phases"][p] for r in hits])`.

### 6) Sanity checks before declaring done

- `pure_dispatch` p50 is the lowest latency across mechanisms at the same
  cell — if not, suspect the harness.
- `load_pdi_expanded` p50 grows with `tiles × bds × rows_per_col` — if it
  doesn't, suspect the cache.
- `cold_start` `register` phase ≈ 40 ms across mechanisms — if it diverges
  by more than ~5%, suspect XRT/driver state.
- `--batched` reduces per-dispatch latency by 10×+ for ELF mechanisms.
- The runs append-only to `results/results.jsonl`. Wipe it (`: > results/results.jsonl`)
  if you want a clean reproduction; otherwise the same cell may appear
  multiple times and the plotter will quietly keep the last.

## Known limits

- **ctrlpkt is not in `bench.cpp` yet.** The build path needs the
  column-control-overlay pass before `aiecc --aie-generate-ctrlpkt` will
  compile placed IRON. Skip the cell.
- **Firmware crash:** `load_pdi_* × tiles ∈ {2, 4} × bds=8` reliably aborts
  the process. `baseline` at the same cell is fine. File-worthy bug.
- **PDI cache:** see the critical caveat at the top. Any number labelled
  `load_pdi_fw` in v1 is a cache-hit measurement.
- **AB mode** (two `@device` regions + `ab_orch` orchestrator) builds but
  is not dispatched by `bench.cpp` — v2.
- **5-BO arg cap (`tools/aiecc/aiecc.cpp:3558`):** generator packs all
  tiles into a shared in/out tensor. Don't try to add per-tile BOs without
  bypassing the cap.
- **Per-core program memory is tiny** (~752 bytes `.text` per tile, ~24 KB
  total at whole array, ~4.6% of the per-core program-memory budget on
  AIE2P). Dispatch-cost growth comes from txn-stream length and memtile
  register programming, not core code size.

## Quick reference: file roles

- `generate.py` — placed-IRON MLIR emitter; one source for all mechanisms.
- `Makefile` — per-mechanism aiecc invocations; `KEY` encodes the cell.
- `bench.cpp` + `bench_runner.h` + `json_writer.h` — host driver, two
  runtime-family paths, per-iter `ns_samples` to JSONL.
- `verify_artifact.py` — txn opcode histogram + mechanism invariants;
  auto-run by Makefile.
- `scripts/run_matrix.sh` — sweep driver with env-var widening.
- `scripts/plot.py` — matplotlib chart generator.
- `REPORT.md` — the report this skill produces / updates.
- `results/results.jsonl` — append-only one-JSON-per-line dataset.
- `results/plots/` — generated PNGs.

## When *not* to use this skill

- Single-file edits to `generate.py` / `bench.cpp` etc. — just edit directly.
- Researching how mechanisms differ in MLIR/aiecc — read the v1 plan
  `~/.claude/plans/i-need-micro-benchmarks-of-reactive-quill.md` first.
- v2 work (ctrlpkt host-side, AB dispatch, warm_reconfig isolation) — that
  needs `bench.cpp` extensions, not just a sweep. This skill is for
  *running and reporting on what's already built*, not for extending it.
