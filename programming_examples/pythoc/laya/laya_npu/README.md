# laya_npu — canonical Laya NPU implementation

`laya_npu/` is the production version of Laya on the NPU, built entirely from local source and raw model inputs, and served by the [local API](../laya_local/README.md).

The build covers 634 core ELFs, 868 packed weight/data blobs, and 35 full-ELF graphs: 16 exact shapes, 16 fast shapes, two B1/L64 pair variants, and the exact B2→B1/L64 mixed route. It also stages a runtime package (`laya_npu_runtime`) with a `Client` for them. `tests/expected/` holds the identity of every binary; a change that alters generated code updates it and says why in the commit.

Exact is the default and supports all 16 shapes. Fast is explicit and eligible only at B1/B2/B4 L64. Those three graphs use M30's m2n2 dense leaf (`src/dense/mmul_m2n2.cc`, core build `dense_m2n2`), which gives output bitwise identical to the earlier fast leaf. Their gate/up epilogue is the branchless GELU (`M8B_GELU_FAST` in `src/dense/wrapper_m8b_d1.cc`, M35), bitwise identical to `gelu_observable` on every BF16 input; the other dense builds keep `gelu_observable`. The B1/L64 fast pair and the other 13 fast shapes keep the earlier leaf; the 13 are rebuilt only to reproduce their historical numerical failures. The three fast graphs and the fast B1/L64 pair also run backbone attention on M35's native-float leaf (`src/special/leaves/m35_native.py`, special device `l64_native`: bf16 products, f32 accumulation, the NLF exp2 and inv in place of emulated IEEE FP32), which passes the fast screen at all three shapes and saves 28/45/81 ms per B1/B2/B4 call; fast B8/L64 keeps `l64_fast` and its recorded output.

Fast B1/L64 runs its 28-layer backbone on one configuration, `device_cb` (M36). The dense engine (B1-split, core build `dense_m2n2_cb`) sits on columns 0-3 and native attention (`l64_native_cb`) on columns 4-7, so the 57 alternating device_02/device_05 regions become one. Output is bit-identical and the call drops from 100.3 to 57.3 ms device time.
- Design: `src/graphs/families/device_cb.py` (IRON Python, with `engine_cb.py` and `attention_cb.py`). It also generates the two halves' core-build templates.
- Weight layout: `src/packing/engine_cb.py`, produced at pack time from the compact fast weights.
- Fast B2/B4 L64 keep device_02/device_05.

Fast B1/B2/B4 L64 and the fast B1/L64 pair run decision attention on `device_05f` (M37) in place of device_08: device_05 with the `l64_fast` cores, driven by the same-B backbone sequence, which saves 6/12/24/12 ms per call. device_08 and `l64_fast` differ by about 1 bf16 ulp on rare inputs, so five qualification fixtures were re-recorded. The fast screen passes at all three shapes with M35's margins.

Exact B1/L64 and both B1/L64 pairs run the engine on `device_s8` (M37) in place of device_02. On device_02, at B = 1 both 64-row halves compute batch row 0. device_s8 is the same B1 split on all 8 columns: each half gets different weight tiles, so every core runs half of device_02's tiles. Output is bit-identical. Device time per call (paired with the previous stage) is:

| graph | before | after |
|---|---:|---:|
| exact/b1_l64 | 171.9 ms | 138.4 ms |
| pair/exact/b1_l64 | 279.2 ms | 212.1 ms |
| pair/fast/b1_l64 | 160.2 ms | 140.8 ms |

- Design: `src/graphs/families/device_s8.py` and `engine_s8.py`. It also generates the core-build template.
- Core builds: `dense_native_s8` (exact) and `dense_fast_s8` (the fast pair's leaf). Both compile `dispatch.cc` with the RoPE rule `3 * j < 2 * ntile`.
- Weight layout: `src/packing/engine_cb.py` with `cols=8`, produced at pack time from the compact weights.
- The exact B2→B1 mixed route keeps device_02. Its B2 and B1 dispatches share each engine region, so splitting the B1 request out would add a reload per region, about as much as it saves.
- An exact engine + attention device like device_cb does not pay. Exact compute is compute-bound, so 4 columns double it. After the 8-column split it would save at most 1.6 ms, at B1/L64.

## Prerequisites

`laya_npu` carries no compiler. It builds against the SDK the enclosing repository builds: Peano/llvm-aie, MLIR-AIE, llvmlite, PythoC, the Python environment, and XRT. See [toolchain/README.md](toolchain/README.md). Activate it first. From the directory that contains `laya_npu/`:

```bash
source ../../../../env.sh
python laya_npu/fetch_model.py      # raw checkpoint; too large for git, verified against manifests/model.json
```

## Build

```bash
python laya_npu/rebuild.py --output /tmp/laya_npu-build                           # all cores, blobs and graphs (~30 min)
python laya_npu/stage_runtime.py --build /tmp/laya_npu-build --output /tmp/laya_npu-stage
python laya_npu/stage_research.py --build /tmp/laya_npu-build --stage /tmp/laya_npu-stage   # optional: rejected fast shapes
```

The graph device templates are generated, not stored. Each design family in `src/graphs/families/` is IRON Python: residual, LayerNorm, cast, decision GEMM, dense engine, attention, `device_cb` and `device_s8`. `src/graphs/composition.json` declares which devices, core builds and runtime sequences each template holds. `rebuild.py` runs `src/graphs/emit.py`, which lowers the devices and sequences with `aie-opt` into `BUILD/templates/` (about 35 s on 32 cores); run `python laya_npu/src/graphs/emit.py OUT_DIR` to emit them alone.

`rebuild.py --targets exact/b1_l64 ...` builds a subset of graphs; every core is still built. The per-family builders (`build.py` for residual, `build_scalar.py`, `build_decision.py`, `build_dense.py`, `build_attention.py`, `build_special.py`, `build_packing.py`) can also be run alone with `--target NAME --output NEW_DIRECTORY`. Builders refuse to write into an existing directory.

To use the staged runtime, put the stage directory on `PYTHONPATH` with the SDK environment still active. Then use `Client('exact')`, or select `Client('fast')` explicitly. The Client owns the device lock (`/tmp/npu-dev.lock`) for its lifetime, so use it as a context manager.

## Tests

```bash
cd laya_npu
python -m pytest                                   # unit tests: plans, policy, runtime ownership, scheduling (no SDK build, no device)
LAYA_NPU_BUILD=/tmp/laya_npu-build python -m pytest -m build # rebuilt binaries match tests/expected/; staging; dense patch; ELF normalization
LAYA_NPU_STAGE=/tmp/laya_npu-stage python -m pytest -m npu   # NPU end-to-end
```

Without `LAYA_NPU_BUILD`, the `build` tests run `rebuild.py` themselves (about 30 minutes). Without `LAYA_NPU_STAGE`, the `npu` tests stage from the build fixture.

- **`build`** compares every rebuilt core, its pre-transform native dense cores, and its shared objects against [tests/expected/cores.json](tests/expected/cores.json). It also checks every compressed blob against `packing.json`, and every graph's normalized ELF, control streams included, against `graphs.json`. Graph ELFs differ only in the PDI revoke_id wrapper fields, which `src/packaging/identity.py` normalizes and nothing else. `test_core_layout.py` checks that every prebuilt core's buffer symbols sit at the addresses its graph template gives them. Run it from a clean checkout (`git worktree add`) to show that the build needs nothing outside git except the fetched checkpoint.
- **`npu`** covers the following:
  - All 19 eligible shapes, twice each, plus their changed/return lifecycle cases, against the expected output, raw-logit and formatted-answer hashes in `inputs/qualification/fixtures.json`.
  - Pair and mixed services must equal serial execution.
  - The 13 rejected fast shapes must reproduce their recorded outputs.
  - The public Client's lock ownership and routing.
  - The session stops if the amdxdna driver logs anything new.

The core and graph hashes are those of this branch's mlir-aie (rebuilt 2026-11-08, when the device templates became generated); the NPU outputs in `fixtures.json` are unchanged. If a change is meant to alter generated code, update `tests/expected/` and say why in the commit.

## Benchmark

```bash
python laya_npu/benchmark.py --stage /tmp/laya_npu-stage --output /tmp/laya_npu-perf
python laya_npu/summarize_benchmark.py --report /tmp/laya_npu-perf/benchmark.json --markdown /tmp/laya_npu-perf/summary.md --csv /tmp/laya_npu-perf/summary.csv
```

The benchmark validates outputs outside the timed region and measures two opposite-order blocks of resident calls after cold setup and warmup, for 19 regular shapes and six service configurations. Results: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Layout

- `src/`: kernel sources (PythoC Python, C++, MLIR templates) per family, packing recipes, the graph designs, compositions and plans (`src/graphs/`), packaging identity, and the runtime.
- `inputs/model/`: checkpoint config and tokenizer. `model.safetensors` is fetched, not tracked.
- `inputs/qualification/fixtures.json`: NPU test inputs and expected outputs.
- `tests/`: unit, `build` and `npu` tests, and `tests/expected/` golden output identities.
- `toolchain/`: SDK prerequisites and compiler provenance.
- `docs/`: [performance](docs/PERFORMANCE.md).

`staged/` (a staged runtime, used by default by `laya_local`) is outside git.
