# dispatch_micro: NPU configuration / dispatch micro-benchmarks

Apples-to-apples latency numbers for the various ways an NPU workload can be
(re)configured and dispatched on Strix (`npu2`). Five mechanisms cross two
runtime families and are timed against a common workload (passthrough kernel,
single shim DMA in/out per tile).

| Mechanism            | Family    | aiecc flags                                                                 | Runtime model                          |
|----------------------|-----------|-----------------------------------------------------------------------------|----------------------------------------|
| `baseline`           | xclbin    | `--aie-generate-xclbin --aie-generate-npu-insts`                            | `xrt::xclbin` + `xrt::kernel`          |
| `load_pdi_fw`        | full ELF  | `--generate-full-elf`                                                       | `xrt::elf` + `xrt::ext::kernel`        |
| `load_pdi_expanded`  | full ELF  | `--generate-full-elf --expand-load-pdis`                                    | `xrt::elf` + `xrt::ext::kernel`        |
| `ctrlpkt` *(v2)*     | xclbin    | `--aie-generate-xclbin --aie-generate-ctrlpkt --ctrlpkt-name=...`           | (compiler-generated ctrlpkt artifacts) |
| (`pure_dispatch`)    | n/a       | same as `baseline` — measured by the `pure_dispatch` metric, not a mechanism | n/a                                    |

`ctrlpkt` is not yet wired into `bench.cpp` or the smoke Makefile target.
The current placed-IRON design also needs the column-control-overlay pass
(see `test/npu-xrt/ctrl_packet_reconfig_elf/run.lit`) before `aiecc
--aie-generate-ctrlpkt` will succeed. Tracked as v2.

The compute kernel is `aie_kernels/generic/passThrough.cc` (reused via VPATH,
not copied). All BD/DMA work happens on a single shared in/out tensor pair;
per-tile slicing is done via shim-DMA offsets in the runtime sequence
because aiecc caps the kernel signature at 5 BO arg slots
(`tools/aiecc/aiecc.cpp:3558`). The benchmark therefore scales to 8 tiles
without running out of kernel arg slots.

## Quick start

```bash
source ../../../env.sh
cd programming_examples/benchmarks/dispatch_micro

# Build the smoke slice (one config per mechanism) and the bench binary:
make smoke

# Or pick one combination and build it:
make MECH=load_pdi_expanded DEVICE=npu2_1col TILES=1 BDS=4 TOPOLOGY=linear

# Run a single benchmark:
./bench --build-dir=build/load_pdi_expanded_npu2_1col_t1_b4_linear \
        --mechanism=load_pdi_expanded --metric=pure_dispatch \
        --tiles=1 --bds=4 --warmup=10 --iters=100

# Run the full default matrix and plot:
./scripts/run_matrix.sh
python scripts/plot.py results/results.jsonl
```

## Sweep axes

Defaults (`run_matrix.sh`) target the smoke slice. Override via env:

| Var          | Default          | Notes                                          |
|--------------|------------------|------------------------------------------------|
| `MECHANISMS` | the four above   | space-separated                                |
| `METRICS`    | all three        | `pure_dispatch warm_reconfig cold_start`       |
| `DEVICES`    | `npu2_1col`      | `npu2_1col npu2_4col npu2`                     |
| `TILES`      | `1`              | per-device max: 1 / 4 / 8                      |
| `BDS`        | `2`              | BDs per shim DMA task                          |
| `TOPOLOGIES` | `linear`         | `linear branch hop`                            |
| `AB`         | `0`              | set to `1` to emit a two-config A↔B build      |
| `BATCHED`    | `0`              | set to `1` for an extra runlist variant        |
| `COLD_RUNS`  | `30`             | fresh processes per `cold_start` measurement   |

A full sweep is `mechanisms × devices × tiles × bds × topologies × ab` builds.
Keep the default small; widen one axis at a time.

## Metrics

`cold_start` — one fresh-process sample of, separately:
- artifact load (`xrt::xclbin` ctor / `xrt::elf` ctor)
- registration / context (`device.register_xclbin` + `xrt::hw_context` for
  xclbin family; `xrt::hw_context(device, elf)` for ELF family)
- kernel handle (`xrt::kernel` / `xrt::ext::kernel`)
- first dispatch

Note: across `COLD_RUNS` iterations, page cache + firmware-side caches may
warm. We do not flush them. The number reflects "first launch on a system
that has run this artifact before," not cold-from-truly-cold.

`warm_reconfig` — pre-built context; brackets only the reconfig dispatch
(submit + wait). Verification of correctness happens **outside** the loop.
For A↔B builds, each iteration is one direction; emit one row per direction.

`pure_dispatch` — pre-loaded context, fixed configuration, hot loop. With
`--batched`, the loop runs `xrt::runlist::execute()` over `--batch-size`
runs and reports `time / batch_size`.

All three metrics emit `ns_samples` (raw per-iteration array) plus min /
p50 / p90 / p99 / max / avg in the JSON.

## Output

`results/results.jsonl` — one JSON object per run, one line each:

```json
{"mechanism":"load_pdi_expanded","metric":"warm_reconfig","build_dir":"build/...",
 "tiles":1,"bds":4,"warmup":10,"iters":100,"batched":false,"batch_size":1,
 "ns":{"min":12345,"p50":13100,"p90":14800,"p99":19800,"max":21200,"avg":13550},
 "ns_samples":[...]}
```

`plot.py` produces (where data is present):
- `pure_dispatch_vs_tiles.png`, `pure_dispatch_vs_bds.png`
- `warm_reconfig_vs_tiles.png`, `warm_reconfig_vs_bds.png`
- `cold_start_breakdown.png` (stacked bars per mechanism)

## Sanity checks

After a successful matrix run, the data should show:
- `pure_dispatch` is the lowest latency across mechanisms.
- `load_pdi_expanded` warm_reconfig scales roughly linearly with `tiles × bds`.
- `load_pdi_fw` warm_reconfig is ~constant in txn size (firmware-side dominates).
- `cold_start` is orders of magnitude larger than any `warm_reconfig`.
- `--batched` reduces `pure_dispatch` median at high `iters`.

## Caveats

- `ctrlpkt` dispatch is stubbed in `bench.cpp`. The build path works
  end-to-end and `verify_artifact.py` checks the artifacts; the host driver
  for it is v2.
- `AB=1` is supported only for `load_pdi_fw` / `load_pdi_expanded`. Baseline
  and ctrlpkt skip it.
- Only `npu2` (Strix) is wired in the Makefile's kernel build rule; extend
  the `passThrough.cc.o` rule for `npu1` if you need it.
- The unified-buffer layout (single in/out shared across tiles) keeps us
  under aiecc's 5-BO arg-slot cap, but means the per-tile data placement
  is contiguous in host memory rather than per-tile-isolated. If you need
  per-tile-independent buffers, you'll need to bypass the cap (e.g., via
  a column-overlay xclbin that exposes more arg slots).
