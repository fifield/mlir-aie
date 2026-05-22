# llama32_1b PythoC port -- metrics harness

Two harnesses live here. Both shell out to `llama32_1b_inference.py`; neither
imports the orchestration directly.

## `test_phase_snapshot.py` -- per-phase JSON snapshot

What it does: runs `--run-only --verify --profile --synthetic-weights
--n-tokens 10` against the cached kernels in `build_peano/`, parses stdout
into a JSON snapshot, and diffs against the most recent prior snapshot.

What it captures:

- per-kernel compile time (parses `Compiled <name>: <s>s -> ...`)
- per-token decode latency (median / p95 / max / mean)
- per-layer K/V correlation min + max
- logits correlation, NPU top-1, CPU top-1, match boolean
- decode token id sequence
- branch + short rev, PEANO commit hash, UTC timestamp

Snapshots land in `tests/snapshots/<branch>_<rev>_<ts>.json`.

Gates (`compare_snapshots`, default thresholds):

- per-kernel compile time > +15% slower => FAIL
- decode p95 latency > +15% slower => FAIL
- token-sequence drift => reported, NOT a failure (synthetic weights
  amplify bf16 drift; use the HF gate for answer-level correctness)

Run via:

```bash
source /home/jfifield/npu-dev-pythoc/env.sh
make snapshot                          # capture + diff vs latest
pytest tests/test_phase_snapshot.py -v -s
python tests/test_phase_snapshot.py compare A.json B.json
```

## `test_hf_answer_gate.py` -- gold-standard answer gate

What it does: with **real** Meta-LLaMA-3.2-1B-Instruct weights from HF,
runs the inference end-to-end on "What is the capital of France?" and
asserts that the first 5 decode tokens detokenize to a string containing
"Paris" (case-insensitive). If HF weights aren't cached locally the test
auto-skips with a clear message.

This is the gate to run at each PythoC-phase exit when HF auth is set up;
synthetic-weight snapshots can pass while real-weight behavior regresses.

```bash
source /home/jfifield/npu-dev-pythoc/env.sh
make hf-gate
pytest tests/test_hf_answer_gate.py -v -s
```
