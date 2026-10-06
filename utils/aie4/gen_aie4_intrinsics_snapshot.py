#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate the AIE4 intrinsic snapshot from llvm-aie TableGen sources.

Run llvm-tblgen --dump-json and write signatures and normalized properties
sorted by intrinsic name. Regeneration is manual; review the diff and restore
the source-revision header as described in utils/aie4/README.md.

Default input paths use the enclosing superproject's llvm-aie/ source tree
and install/mlir/bin/llvm-tblgen. Override them with --llvm-aie-include and
--llvm-tblgen. The output defaults to aie4_intrinsics_snapshot.yaml beside
this script. PyYAML is required.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

# Layout: <superproject>/mlir-aie/utils/aie4/<script>.
SUPERPROJECT_ROOT = Path(__file__).resolve().parents[3]

THIS_DIR = Path(__file__).resolve().parent

# Reject unexpected source files so TableGen layout changes require review.
AIE4_TD_BASENAMES = {"IntrinsicsAIE4.td", "IntrinsicsAIE4Mac.td"}


def derive_llvm_name(record_name: str, llvm_name: str = "") -> str:
    """Derive the LLVM name, e.g. int_aie4_vadd32_512 -> llvm.aie4.vadd32.512."""
    assert record_name.startswith("int_"), record_name
    # An explicit LLVMName overrides record-name derivation.
    assert llvm_name == "", (
        f"{record_name}: LLVMName is non-empty ({llvm_name!r}); the naive "
        "derive_llvm_name derivation is no longer valid for this record"
    )
    return "llvm." + record_name[len("int_") :].replace("_", ".")


def _extract_properties(val: dict, tblgen_json: dict) -> list[str]:
    """Normalize IntrProperties into stable, sorted strings.

    Anonymous TableGen names change when unrelated records are added. Encode
    these properties using their class and sorted fields, e.g. ImmArg<ArgNo=1>.
    """
    props = []
    for p in val.get("IntrProperties", []):
        def_name = p["def"]
        resolved = tblgen_json.get(def_name, {})
        if resolved.get("!anonymous"):
            superclasses = [
                s for s in resolved.get("!superclasses", []) if s != "IntrinsicProperty"
            ]
            assert len(superclasses) == 1, (
                f"{def_name}: expected exactly one non-IntrinsicProperty "
                f"superclass for an anonymous IntrinsicProperty def, got "
                f"{superclasses}"
            )
            cls_name = superclasses[0]
            # IsDefault belongs to IntrinsicProperty, not the attribute payload.
            fields = {
                k: v
                for k, v in resolved.items()
                if not k.startswith("!") and k != "IsDefault"
            }
            field_str = ",".join(f"{k}={v}" for k, v in sorted(fields.items()))
            props.append(f"{cls_name}<{field_str}>")
        else:
            props.append(p["printable"])
    result = sorted(props)
    for s in result:
        # Named properties must not leak unresolved anonymous references either.
        assert (
            "anonymous_" not in s
        ), f"unresolved/unstable anonymous property leaked into output: {s!r}"
    return result


def dump_tblgen_json(llvm_tblgen: Path, include_dir: Path, umbrella_td: Path) -> dict:
    with tempfile_named() as out_path:
        cmd = [
            str(llvm_tblgen),
            "-I",
            str(include_dir),
            "--dump-json",
            str(umbrella_td),
            "-o",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, cwd=SUPERPROJECT_ROOT)
        with open(out_path) as f:
            return json.load(f)


class tempfile_named:
    """Keep a closed temporary file available by path until llvm-tblgen finishes."""

    def __enter__(self):
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json", prefix="aie4-intrinsics-")
        import os

        os.close(fd)
        self.path = Path(path)
        return self.path

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def extract_aie4_intrinsics(tblgen_json: dict) -> list[dict]:
    """Extract and normalize int_aie4_* Intrinsic records from TableGen JSON."""
    records = []
    for name, val in tblgen_json.items():
        if not isinstance(val, dict):
            continue  # e.g. the "!instanceof" index key
        if not name.startswith("int_aie4_"):
            continue
        superclasses = val.get("!superclasses", [])
        if "Intrinsic" not in superclasses:
            continue

        locs = val.get("!locs", [])
        if len(locs) != 1:
            raise ValueError(f"{name}: expected exactly one !locs entry, got {locs}")
        src_file, _, src_line = locs[0].partition(":")
        src_file = Path(src_file).name
        if src_file not in AIE4_TD_BASENAMES:
            raise ValueError(
                f"{name}: unexpected source file {src_file!r}; expected one of "
                f"{sorted(AIE4_TD_BASENAMES)}. A toolchain bump may have moved "
                "AIE4 intrinsics to a new file -- update AIE4_TD_BASENAMES."
            )

        ret_types = [t["printable"] for t in val.get("RetTypes", [])]
        param_types = [t["printable"] for t in val.get("ParamTypes", [])]
        properties = _extract_properties(val, tblgen_json)
        disable_default_attributes = int(val.get("DisableDefaultAttributes", 0))

        records.append(
            {
                "name": derive_llvm_name(name, val.get("LLVMName", "")),
                "record_name": name,
                "source_file": src_file,
                "source_line": int(src_line),
                "ret_types": ret_types,
                "param_types": param_types,
                "properties": properties,
                "disable_default_attributes": disable_default_attributes,
            }
        )
    return records


def to_yaml_str(records: list[dict]) -> str:
    # Stable-sort by LLVM name so the file is diff-friendly across regens.
    records_sorted = sorted(records, key=lambda r: r["name"])
    doc = {
        "schema_version": 1,
        "description": (
            "Signature snapshot of AIE4 LLVM IR intrinsics (int_aie4_* records), "
            "generated from the pinned llvm-aie submodule. Used for drift "
            "detection against a weekly-floating toolchain pin -- see "
            "check_aie4_intrinsics.py. Do not hand-edit; regenerate "
            "with gen_aie4_intrinsics_snapshot.py."
        ),
        "count": len(records_sorted),
        "intrinsics": records_sorted,
    }
    return yaml.safe_dump(
        doc,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--llvm-tblgen",
        type=Path,
        default=SUPERPROJECT_ROOT / "install/mlir/bin/llvm-tblgen",
    )
    ap.add_argument(
        "--llvm-aie-include",
        type=Path,
        default=SUPERPROJECT_ROOT / "llvm-aie/llvm/include",
    )
    ap.add_argument(
        "--umbrella-td",
        type=Path,
        default=None,
        help="Defaults to <llvm-aie-include>/llvm/IR/Intrinsics.td",
    )
    ap.add_argument(
        "--output", type=Path, default=THIS_DIR / "aie4_intrinsics_snapshot.yaml"
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Print YAML to stdout instead of writing --output",
    )
    args = ap.parse_args()

    umbrella_td = args.umbrella_td or (args.llvm_aie_include / "llvm/IR/Intrinsics.td")

    if not args.llvm_tblgen.exists():
        sys.exit(f"error: llvm-tblgen not found at {args.llvm_tblgen}")
    if not umbrella_td.exists():
        sys.exit(f"error: umbrella Intrinsics.td not found at {umbrella_td}")

    tblgen_json = dump_tblgen_json(args.llvm_tblgen, args.llvm_aie_include, umbrella_td)
    records = extract_aie4_intrinsics(tblgen_json)
    if not records:
        sys.exit(
            "error: found zero int_aie4_* intrinsics -- something is wrong "
            "with the tblgen invocation or the filter"
        )

    yaml_str = to_yaml_str(records)
    if args.print_only:
        print(yaml_str, end="")
    else:
        args.output.write_text(
            "# Copyright (C) 2026 Advanced Micro Devices, Inc.\n"
            "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n\n" + yaml_str
        )
        print(f"wrote {len(records)} intrinsics to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
