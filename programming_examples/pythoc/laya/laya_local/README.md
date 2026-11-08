# Local app API

Run from the Laya example directory with the existing toolchain environment.
No extra serving dependencies are required.

The backend is [`laya_npu`](../laya_npu/README.md), the canonical NPU implementation, built from source. Build and
stage it once; the Client loads the staged runtime from `laya_npu/staged/current`, or
from the directory named by `LAYA_NPU_STAGE`:

```sh
python laya_npu/rebuild.py --output /tmp/laya_npu-build
python laya_npu/stage_runtime.py --build /tmp/laya_npu-build --output laya_npu/staged/current
```

The default is **`exact`**, supporting all 16 combinations of B1/B2/B4/B8 and
L64/L128/L256/L512. Eligible exact three-question L64 requests automatically use the
mixed B2+B1 launch when cache capacity permits.

Approximate `backend="fast"` is never the default and supports only **B1/L64,
B2/L64 and B4/L64**; the other 13 fast shapes failed fixed quality gates. Its dense
leaf is M30's m2n2 kernel, bitwise identical in output to the earlier fast leaf. An
unsupported shape produces an error before execution, with no silent fallback,
padding or forced bucket change. All chunks and states of a batch are checked
before the first NPU launch. Routing names the backend, arithmetic and approximate
status; `/models` lists supported shapes and rejection reasons.

The runtime retains up to **two contexts and 2 GiB of device arenas** by default.
These bounds exclude model/CPU memory. Complete-response memoization
(`memo_entries > 0`) and independent request pairing (`phase_major=True`) are off by
default. Pairing is limited to two supplied independent states with one question
each, naturally B1/L64; other inference runs serially with a visible scheduling
reason. It does not wait across HTTP requests. Measurements are in
[PERFORMANCE.md](../laya_npu/docs/PERFORMANCE.md).

```python
from laya_local import Client

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "payments and refunds", "support": "technical problems"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["low", "medium", "high"],
    },
    "cancel": {"type": "noul", "instructions": "Does the customer want to cancel?"},
}

with Client() as client:
    result = client.predict("I was billed twice. Please refund me.", questions)
    print(result["answers"]["department"]["choice"])
    results = client.predict_batch(["Please cancel.", "My app crashes."], questions)
```

`predict` returns `model`, `answers`, `usage`, `routing`, and `elapsed_seconds`.
Answers retain the existing implementation's probabilities, confidence, action
probability, and typed value (`choice`, `score`, or `noul`). A score is the expected
zero-based rubric index; noul is the probability of yes. `predict_batch` takes a
nonempty list of states and a shared question mapping and returns a list in state
order. States may be strings, JSON objects, or lists. Question IDs retain order.

Keep one Client open for the lifetime of the app. It owns `/tmp/npu-dev.lock`,
serializes calls, and releases the runtime and lock on exit. Do not run another
NPU application alongside it. Multiple states run sequentially unless optional eligible pairing is enabled; each state's
questions split into batches of 8/4/2/1 as needed.

## HTTP

```sh
source <npu-dev>/env.sh
python -m laya_local.server --port 8000
```

Open **http://127.0.0.1:8000/** for the Support desk demo. Paste a customer
message or choose a sample to see department probabilities, urgency, and the
probability of cancellation. The expandable panel shows the actual API request
and response. The page and API share the same origin; no frontend build or
external assets are needed.

```sh
curl http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"state":"Please cancel my subscription.","questions":{"cancel":{"type":"noul","instructions":"Does the customer want to cancel?"}}}'
```

| Endpoint | Body / response |
| --- | --- |
| `GET /` or `/demo` | Support desk demo |
| `POST /predict` | `{ "state": ..., "questions": {...} }` → one result |
| `POST /predict/batch` | `{ "states": [...], "questions": {...} }` → list of results |
| `GET /models` | Selected backend, qualified shapes, rejection reasons and limits |
| `GET /health` | Server/runtime initialized; not an active device probe |

The server listens on `0.0.0.0` (all IPv4 interfaces) and handles requests sequentially.
From another machine, open `http://<server-ip>:8000/`. Invalid
requests return JSON errors with HTTP 400; runtime failures return 500 with details
in the server log. Bodies are limited to 1 MiB. Browser apps should proxy these
endpoints through their local dev server (no cross-origin CORS support).

This surface serves the pinned English checkpoint only. Routing metadata identifies
the selected backend; there is no language detection or checkpoint selection. Exact
mode supports a 512-token sequence budget; qualified fast shapes stop at L64. The
192-token head budget remains unchanged. Prompt/options consume
part of the sequence budget; the remaining state text is truncated by the pinned
tokenizer path. An option set that loses candidate markers is rejected. Token usage
counts encoded tokens across question rows, including repeated state text.

The API preserves the accepted implementation's numerical behavior and exceptions;
it does not claim parity with newer upstream releases. It intentionally omits raw
tensors, capture/debug controls, training, and long-context inference.

## Repeatable benchmark

Stop the local server first so the benchmark can acquire `/tmp/npu-dev.lock`.
With the existing environment activated, run from the Laya example directory:

```sh
python -m laya_local.benchmark --output /tmp/laya-benchmark.json
# Shorter trial, or select specific shapes:
python -m laya_local.benchmark --lengths 64 128 --batches 1 2 --repeats 3 \
  --output /tmp/laya-benchmark-short.json
```

The default requested grid covers all **16 combinations** of physical lengths 64/128/256/512 and
native row batches 1/2/4/8, with one warm-up and five measured calls per shape.
Use `--warmup` and `--repeats` to change those counts. Select the backend with
`--backend exact` or `--backend fast`. Fast requires selecting its qualified
subset (`--lengths 64 --batches 1 2 4`), or explicitly passing
`--skip-unavailable`. Without that flag, any unavailable shape fails preflight
before inference. With it, omissions are listed and `complete_requested_matrix`
is false; no skipped shape is represented as measured. B is the number of
independent question rows in one native call, not the number of sequential states
in `Client.predict_batch`.

Inputs are the deterministic, nearly full real-text rows of the laya_npu qualification
fixtures (`laya_npu/inputs/qualification/fixtures.json`). The benchmark verifies each actual token length is within
eight tokens of the requested bucket. Reports include those lengths, full input
rows and hashes, checkpoint/composition identity, raw timings, median, nearest-rank
p95, and row/token throughput. Five samples give only a coarse p95 (the maximum).
Formatted answers and valid-token output bytes must repeat exactly across calls;
this checks repeatability, not model accuracy. Runtime incidents/fallbacks fail
the run rather than mixing execution paths into a single timing result.

Reports also record bundle hashes, runtime source hashes, and per-call launch
statistics so implementations can be distinguished.

Timing covers the native `Inference.rows()` call, including tokenization, NPU
execution, and CPU heads. It excludes fixture generation, output hashing, HTTP,
and owner initialization. Owner setup and each shape's warm-up are recorded
separately. Shapes run in length-major, batch-minor order on one persistent selected-backend
owner. Keep that order, repeat counts, and machine conditions the same when
comparing runs. The full grid may take several minutes.

The JSON report is saved after every completed shape and marked `complete` only
after the selected available grid succeeds; `complete_requested_matrix` separately
records whether any requested shapes were omitted. Failures preserve completed cases and
the error. Existing reports are never overwritten; omit `--output` for a unique
timestamped filename. A busy NPU lock fails immediately without touching the device.

## Snake demo

Start the same server and open **http://127.0.0.1:8000/snake** (or
`http://<server-ip>:8000/snake` remotely). Play, pause, advance one move, reset
with a chosen seed, or cap the presentation speed. Every move waits for a fresh
NPU result. The page shows original direction probabilities, the executed move,
shield corrections, food estimates, inference latency, and completed moves/s.
The first move includes runtime warm-up; browser median/p95 exclude it.
Pausing lets the outstanding decision finish. Reset is available after it finishes.
Changing shield mode starts a new game.
One Snake game is shared by this server; revisions reject stale moves from other tabs.

The English checkpoint receives compact planner features, without Snake training.
A Hamiltonian cycle planner supplies safe directions and the direction with the
most safe progress toward food. With the shield enabled, the highest-probability
admissible model direction is executed. Turning it off executes the raw first
choice, while still supplying planner features. Food reachability describes the
current empty cells, rather than future space after the tail moves. The model's
estimates are not calibrated death probabilities.

Four real questions (move, safe route, food reachable, food adjacent) fit one
native B4 call. Three questions use B2 + B1. laya_npu retains both contexts within the
default cache bounds. This differs from the upstream MLX demo's B3 multilingual workload.
The normal 24 × 16 board uses L64 for the sampled compact prompts.

For a terminal view or repeatable headless measurement:

```sh
source <npu-dev>/env.sh
python -m laya_local.snake
python -m laya_local.snake --headless --steps 300 --fps 0 --seed 102 \
  --output /tmp/laya-snake-run.json
# Raw model execution, with the same supplied planner features:
python -m laya_local.snake --unassisted --headless --steps 300 --fps 0
```

`--fps 0` removes pacing. `--width`, `--height`, and `--initial-length` change the
board; at least one dimension must be even. Ctrl-C ends the terminal run and
prints the completed-move summary. Reports contain every input, original model
prediction, before/after board, executed direction, correction, and timing.
Existing reports are not overwritten. CLI statistics exclude a separate warm-up
prediction and model loading. Inference timing covers `Client.predict`, including
tokenization, NPU execution, CPU heads, and formatting. Overall rate also includes
planning, game updates, recording overhead, and any requested pacing.
