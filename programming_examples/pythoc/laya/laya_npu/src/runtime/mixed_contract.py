"""Exact local mixed contract; all ordinary plans use unchanged native validation."""

from .common import ROOT, read, digest, require, validate_plan as validate_native
from .pins import MIXED_PLAN_SHA256


def validate_mixed_structure(plan):
    """Reject malformed allocation/dispatch/group layouts before touching XRT."""
    require(
        plan["batch"] == 3 and plan["length"] in (64, 128, 256, 512),
        "unsupported plan shape",
    )
    require(
        0 < len(plan["arenas"]) <= 32 and all(n > 0 for n in plan["arenas"]),
        "invalid arenas",
    )
    for a in plan["allocations"]:
        require(
            0 <= a["arena"] < len(plan["arenas"])
            and a["offset"] >= 0
            and a["nbytes"] > 0,
            "invalid allocation",
        )
        require(
            a["offset"] + a["nbytes"] <= plan["arenas"][a["arena"]],
            "allocation outside arena",
        )
    spans = {}
    for a in plan["allocations"]:
        spans.setdefault(a["arena"], []).append(
            (a["offset"], a["offset"] + a["nbytes"])
        )
    for ranges in spans.values():
        ordered = sorted(ranges)
        require(
            all(x[1] <= y[0] for x, y in zip(ordered, ordered[1:])),
            "overlapping allocations",
        )

    def binding(b):
        slot, offset, nbytes = b
        require(0 <= slot < len(plan["allocations"]), "invalid allocation slot")
        require(
            offset >= 0
            and nbytes > 0
            and offset + nbytes <= plan["allocations"][slot]["nbytes"],
            "binding outside allocation",
        )

    for d in plan["dispatches"]:
        for b in d["bindings"]:
            binding(b)
        require(
            all(0 <= o < len(d["bindings"]) for o in d["outputs"]),
            "invalid output operand",
        )
    boundaries = set()
    for name in ("commands", "groups"):
        cursor = 0
        for part in plan[name]:
            require(
                part["start"] == cursor
                and cursor < part["end"] <= len(plan["dispatches"]),
                f"invalid {name} partition",
            )
            if name == "commands":
                boundaries.update((part["start"], part["end"]))
            else:
                require(
                    part["start"] in boundaries and part["end"] in boundaries,
                    "launch splits a command",
                )
                require(0 < part["deadline"] <= 1.5, "invalid launch deadline")
            cursor = part["end"]
        require(cursor == len(plan["dispatches"]), f"incomplete {name}")
    require(
        len({g["instance"] for g in plan["groups"]}) == len(plan["groups"]),
        "duplicate kernel instance",
    )
    for name in ("embedding", "mask", "type"):
        require(len(plan["inputs"][name]) == plan["batch"], "input batch mismatch")
        for b in plan["inputs"][name]:
            binding(b)
    require(len(plan["outputs"]) == plan["batch"], "output batch mismatch")
    for b in plan["outputs"]:
        binding(b)


def validate_plan(plan):
    if "schedule_kind" not in plan:
        return validate_native(plan)
    require(
        plan.get("schedule_kind") == "phase_major_b2_b1_l64",
        "unsupported schedule contract",
    )
    path = ROOT / "graphs/mixed/exact/b2_b1_l64/plan.json"
    require(
        digest(path) == MIXED_PLAN_SHA256 and plan == read(path),
        "mixed plan identity changed",
    )
    require(
        plan["batch"] == 3
        and plan["length"] == 64
        and plan["mixed_requests"] == [2, 1]
        and plan["mixed_execution_order"] == [2, 1],
        "mixed request shape",
    )
    require(
        len(plan["arenas"]) == 26
        and len(plan["dispatches"]) == 442
        and len(plan["commands"]) == 83
        and len(plan["groups"]) == 1
        and plan["groups"][0]["instance"] == "mixed_b2_b1",
        "mixed executable contract",
    )
    validate_mixed_structure(plan)


def validate_record(record):
    if record is None:
        return
    path = ROOT / "graphs/mixed/exact/b2_b1_l64/plan.json"
    require(
        record["path"] == "graphs/mixed/exact/b2_b1_l64/plan.json"
        and record["sha256"] == MIXED_PLAN_SHA256
        and digest(path) == MIXED_PLAN_SHA256,
        "mixed registry identity changed",
    )
    plan = read(path)
    validate_plan(plan)
    require(
        record["arena_bytes"] == sum(plan["arenas"]), "mixed registry footprint changed"
    )
    require(
        record["entry"] == "main:mixed_b2_b1"
        and record["arithmetic"] == "native_exact"
        and record["schedule_kind"] == "phase_major_b2_b1_l64",
        "mixed registry execution changed",
    )
