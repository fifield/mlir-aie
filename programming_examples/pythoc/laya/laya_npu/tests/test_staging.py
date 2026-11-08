"""Staging a build yields the eligible runtime plus research-only rejected fast shapes."""

import json

import pytest

pytestmark = pytest.mark.build


def test_staged_bundle(stage_dir):
    bundle = json.loads((stage_dir / "laya_npu_runtime/data/bundle.json").read_text())
    assert len(bundle["variants"]["exact"]["plans"]) == 16
    assert len(bundle["variants"]["fast"]["plans"]) == 3
    assert set(bundle["phase_pairs"]) == {"exact", "fast"}
    assert set(bundle["mixed_requests"]["exact"]) == {"b2_b1_l64"}
    research = json.loads((stage_dir / "research_plans.json").read_text())["plans"]
    assert set(research) == set(bundle["variants"]["fast"]["rejected_shapes"])
    assert len(research) == 13 and not set(research) & set(
        bundle["variants"]["fast"]["plans"]
    )
