#!/usr/bin/env python3
"""Build minimal fused rn3-pair prototype ELFs."""
import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "kernels"))

from build_merged import build_merged, _resolve_build_dir  # noqa: E402
from rn3_pair_pythoc import build_obj  # noqa: E402

_SCRIPT = str(_HERE / "aie2_rn3_pair.py")

# Correctness-smoke shapes. Keep tiny first because the scalar fused kernel is
# intentionally unoptimized and recomputes conv1 values.
_LAYERS = {
    "tiny": (8, 8, 4, 4, 4),
    "re4_tile": (8, 8, 32, 32, 32),
    "re6_tile": (8, 8, 48, 48, 48),
    "re6_oc4": (8, 8, 48, 48, 4),
    "re6_oc8": (8, 8, 48, 48, 8),
}


def _name(label):
    return f"rn3pair_{label}_x1"


def build_one(label="tiny"):
    if label not in _LAYERS:
        raise KeyError(label)
    build_obj(_HERE.parent / "kernels" / "build")
    out_name = _name(label)
    elf = os.path.join(_resolve_build_dir(), f"{out_name}.elf")
    if os.path.exists(elf):
        print(f"  {out_name}: already built, skipping")
        return elf
    args = [str(x) for x in _LAYERS[label]]
    sub = f"rn3pair_{label}"
    return build_merged(
        out_name,
        [sub],
        kind="mc",
        sub_spec_overrides={sub: (_SCRIPT, args)},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=list(_LAYERS), default="tiny")
    args = p.parse_args()
    return 0 if build_one(args.layer) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
