"""Local artifact paths and content identities."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BUILD = ROOT / "build"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_plan(plan):
    """Reject malformed allocation/dispatch/group layouts before touching XRT."""
    require(
        plan["batch"] in (1, 2, 4, 8) and plan["length"] in (64, 128, 256, 512),
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
