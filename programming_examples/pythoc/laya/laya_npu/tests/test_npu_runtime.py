"""End-to-end NPU runs of the staged runtime against fixed expected outputs.

Every eligible shape, its changed/return lifecycle cases, independent-pair and
mixed B2->B1 services, and the rejected fast shapes, which must reproduce their
historical (numerically failing) outputs exactly and stay ineligible.
Fixtures: inputs/qualification/fixtures.json.
"""

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.npu
FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "inputs/qualification/fixtures.json"
    ).read_text()
)


def hashes(result):
    h = hashlib.sha256()
    for row, n in enumerate(result["prepared"]["valid_lengths"]):
        h.update(result["output"][row, :n].tobytes())
    return {
        "output": h.hexdigest(),
        "raw": {
            k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in result["raw"].items()
        },
        "formatted": hashlib.sha256(
            json.dumps(result["result"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


class Models:
    """At most one Inference (and its device contexts) is open at a time."""

    def __init__(self, backend):
        self.backend, self.current, self.key = backend, None, None

    def get(self, mode, key=None):
        key = key or mode
        if self.key != key:
            self.close()
            self.current = self.backend.infer.Inference(
                use_lock=False, arithmetic=mode, cache_entries=1
            )
            self.key = key
        return self.current

    def fresh(self, mode):
        self.close()
        return self.get(mode, key=object())

    def close(self):
        if self.current is not None:
            self.current.close()
        self.current = self.key = None


@pytest.fixture(scope="module")
def models(backend, device_lock, journal):
    journal()
    m = Models(backend)
    yield m
    m.close()


def assert_clean(model):
    assert not model.incidents and not model.fallback


def test_eligibility(models):
    assert len(models.get("exact").availability()["supported_shapes"]) == 16
    assert len(models.get("fast").availability()["supported_shapes"]) == 3


@pytest.mark.parametrize(
    "case",
    sorted(FIXTURES["cases"], key=lambda c: c["target"]),
    ids=lambda c: c["target"],
)
def test_shape(models, journal, case):
    model = models.get(case["target"].split("/")[0])
    for _ in range(2):
        result = model.rows(case["rows"], bucket=case["length"])
        journal()
        assert hashes(result) == case["expected"]
        assert_clean(model)
    for extra in [
        c for c in FIXTURES["eligible_lifecycle"] if c["target"] == case["target"]
    ]:
        result = model.rows(extra["rows"], bucket=extra["length"])
        journal()
        assert hashes(result) == extra["expected"], extra["name"]
        assert_clean(model)


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_pair_matches_serial(models, journal, mode):
    case = next(c for c in FIXTURES["cases"] if c["target"] == mode + "/b1_l64")
    row = case["rows"][0]
    questions = {row["id"]: row["question"]}
    states = [row["state"], row["state"].replace("blue", "red")]
    model = models.fresh(mode)
    serial = []
    for state in states:
        serial.append(model.answer(state, questions)["result"])
        journal()
    model.cache.close()
    model.phase_major = True
    paired = model.answer_batch(states, questions)
    journal()
    assert model.last_schedule["events"][0]["paired"]
    assert [x["result"] for x in paired] == serial


def test_mixed_matches_serial(models, journal):
    fixture = FIXTURES["mixed"]
    model = models.fresh("exact")
    mixed = model.answer(fixture["state"], fixture["questions"])["result"]
    journal()
    assert model.last_response["scheduling"] == "phase_major_b2_b1_l64"
    model.cache.close()
    model.bundle["mixed_requests"] = {}
    serial = model.answer(fixture["state"], fixture["questions"])["result"]
    journal()
    assert mixed == serial


@pytest.fixture(scope="module")
def research_model(models, stage_dir):
    research = json.loads((stage_dir / "research_plans.json").read_text())
    model = models.fresh("fast")
    assert not set(research["plans"]) & set(
        model.plans
    ), "a research shape became eligible"
    model.plans = research["plans"]
    yield model
    models.close()


@pytest.mark.parametrize("case", FIXTURES["research"], ids=lambda c: c["name"])
def test_rejected_fast_shape_reproduces(research_model, journal, case):
    result = research_model.rows(case["rows"], bucket=case["length"])
    journal()
    assert hashes(result) == case["expected"]
    assert_clean(research_model)
