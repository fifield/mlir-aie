"""Bounded two-request eligibility; no cross-call queue or speculative padding."""


def pair_eligibility(prepared, enabled, record):
    if not enabled:
        return False, "disabled"
    if len(prepared) != 2:
        return False, "requires_two_independent_requests"
    if any(len(p["valid_lengths"]) != 1 for p in prepared):
        return False, "requires_one_question_per_request"
    if any(p["bucket"] != 64 for p in prepared):
        return False, "natural_bucket_not_l64"
    if record is None:
        return False, "validated_pair_plan_unavailable"
    return True, "phase_major_b1_l64"
