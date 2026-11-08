"""Three-question fusion is independent of optional B1+B1 request pairing."""


def eligibility(prepared, arithmetic, record, cache_limit):
    if arithmetic != "exact":
        return False, "explicit_fast_unchanged"
    if len(prepared) != 2 or [len(p["valid_lengths"]) for p in prepared] != [2, 1]:
        return False, "requires_greedy_b2_b1"
    if any(
        p["bucket"] != 64 or not all(0 < n <= 64 for n in p["valid_lengths"])
        for p in prepared
    ):
        return False, "natural_bucket_not_l64"
    if record is None:
        return False, "mixed_record_unavailable"
    if record["arena_bytes"] > cache_limit:
        return False, "mixed_exceeds_cache_bound"
    return True, "phase_major_b2_b1_l64"
