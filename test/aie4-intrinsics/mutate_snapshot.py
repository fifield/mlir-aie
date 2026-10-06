#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Apply one mutation to a snapshot copy for the drift check's negative tests.

Requires PyYAML to read and write the snapshot.
"""

import argparse
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("output", type=Path)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_remove = sub.add_parser("remove", help="Delete one intrinsic entry entirely.")
    p_remove.add_argument("name")

    p_rename = sub.add_parser("rename", help="Rename one intrinsic entry.")
    p_rename.add_argument("name")
    p_rename.add_argument("new_name")

    p_param = sub.add_parser("mutate-param", help="Change one param_types entry.")
    p_param.add_argument("name")
    p_param.add_argument("index", type=int)
    p_param.add_argument("new_type")

    p_prop = sub.add_parser("mutate-property", help="Replace one properties entry.")
    p_prop.add_argument("name")
    p_prop.add_argument("old_property")
    p_prop.add_argument("new_property")

    args = ap.parse_args()

    doc = yaml.safe_load(args.snapshot.read_text())
    by_name = {r["name"]: r for r in doc["intrinsics"]}
    if args.name not in by_name:
        sys.exit(f"error: {args.name!r} not found in {args.snapshot}")
    record = by_name[args.name]

    if args.mode == "remove":
        doc["intrinsics"] = [r for r in doc["intrinsics"] if r["name"] != args.name]
        doc["count"] = len(doc["intrinsics"])
    elif args.mode == "rename":
        if args.new_name in by_name:
            sys.exit(f"error: {args.new_name!r} already exists in {args.snapshot}")
        record["name"] = args.new_name
    elif args.mode == "mutate-param":
        if args.index >= len(record["param_types"]):
            sys.exit(
                f"error: {args.name!r} has only {len(record['param_types'])} "
                f"param_types, index {args.index} out of range"
            )
        old = record["param_types"][args.index]
        record["param_types"][args.index] = args.new_type
        print(
            f"mutated {args.name} param_types[{args.index}]: {old} -> {args.new_type}",
            file=sys.stderr,
        )
    elif args.mode == "mutate-property":
        if args.old_property not in record["properties"]:
            sys.exit(
                f"error: {args.name!r} does not have property "
                f"{args.old_property!r} (has {record['properties']})"
            )
        record["properties"] = [
            args.new_property if p == args.old_property else p
            for p in record["properties"]
        ]
        print(
            f"mutated {args.name} properties: {args.old_property} -> "
            f"{args.new_property}",
            file=sys.stderr,
        )

    args.output.write_text(
        yaml.safe_dump(
            doc,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=100,
        )
    )


if __name__ == "__main__":
    main()
