"""Eligibility policy and the NPU fixtures agree: every shape is either eligible or a
rejected research case, and every eligible shape and service has expected outputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "src/runtime/policy.json").read_text())
FIXTURES = json.loads((ROOT / "inputs/qualification/fixtures.json").read_text())
SHAPES = {f"b{b}_l{l}" for l in (64, 128, 256, 512) for b in (1, 2, 4, 8)}


def test_exact_covers_every_shape():
    assert set(POLICY["variants"]["exact"]["eligible_shapes"]) == SHAPES
    assert not POLICY["variants"]["exact"]["rejected_shapes"]


def test_fast_is_eligible_only_where_it_passed():
    fast = POLICY["variants"]["fast"]
    eligible, rejected = set(fast["eligible_shapes"]), set(fast["rejected_shapes"])
    assert eligible == {"b1_l64", "b2_l64", "b4_l64"}
    assert not eligible & rejected and eligible | rejected == SHAPES


def test_fixtures_cover_eligible_shapes_and_rejected_research():
    targets = {c["target"] for c in FIXTURES["cases"]}
    eligible = {
        m + "/" + s for m, v in POLICY["variants"].items() for s in v["eligible_shapes"]
    }
    assert targets == eligible
    research = {c["target"] for c in FIXTURES["research"]}
    assert research == {
        "fast/" + s for s in POLICY["variants"]["fast"]["rejected_shapes"]
    }
    assert {c["target"] for c in FIXTURES["eligible_lifecycle"]} <= eligible


def test_staged_targets():
    from stage_runtime import required_targets

    recipes = json.loads((ROOT / "src/graphs/recipes.json").read_text())
    required = required_targets(POLICY)
    assert len(required) == 22 and required <= set(recipes)
