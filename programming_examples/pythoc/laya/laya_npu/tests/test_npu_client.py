"""The public staged Client on the NPU: device ownership, pairing and mixed routing.

Client takes /tmp/npu-dev.lock itself, so these tests must not hold device_lock.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.npu
FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "inputs/qualification/fixtures.json"
    ).read_text()
)


def payload(r):
    return {k: r[k] for k in ("model", "answers", "usage")}


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_client(backend, journal, mode):
    Client = backend.api.Client
    row = next(c for c in FIXTURES["cases"] if c["target"] == mode + "/b1_l64")["rows"][
        0
    ]
    questions = {row["id"]: row["question"]}
    states = [row["state"], row["state"].replace("blue", "red")]
    journal()
    with Client(mode, phase_major=True, cache_entries=1) as client:
        serial = []
        for state in states:
            serial.append(payload(client.predict(state, questions)))
            journal()
        paired = client.predict_batch(states, questions)
        journal()
        assert list(map(payload, paired)) == serial
        assert all(r["routing"]["paired"] for r in paired)
        with pytest.raises(RuntimeError, match="NPU is in use"):
            Client(mode)
        if mode == "exact":
            mixed = FIXTURES["mixed"]
            result = client.predict(mixed["state"], mixed["questions"])
            journal()
            assert result["routing"]["scheduling"] == "phase_major_b2_b1_l64"
    with pytest.raises(RuntimeError, match="closed"):
        client.predict(states[0], questions)
