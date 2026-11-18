#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""
Generate a register database JSON from an aie-codegen register header
(aie-regdb/globalparams/xaie*gbl_params.h).

The output has the same layout as aie_registers_aie2.json and is read by the
C++ register database (AIERegisterDatabase.cpp) and python/utils/regdb.py:

  {"architecture": ..., "source": ..., "modules": {
      "<module>": {"registers": [
          {"name", "offset", "width", "reset",
           "bit_fields": [{"name", "bits", "bit_range", "reset"}]}]}}}

The header carries offsets, widths, field positions and reset values. It has
no access types or descriptions, so those keys are omitted.

Each register block in the header looks like:

  #define <P>_<REG>                 0x000A0110
  #define <P>_<REG>_WIDTH           32
  #define <P>_<REG>_MASK            0x8000FF00
  #define <P>_<REG>_<FIELD>_LSB     8      /* <FIELD> */
  #define <P>_<REG>_<FIELD>_WIDTH   8      /* <FIELD> */
  #define <P>_<REG>_<FIELD>_MASK    0xFF00 /* <FIELD> */
  #define <P>_<REG>_<FIELD>_DEFVAL  0x0    /* <FIELD> */
  #define value_<P>_<REG>(Field_A,...)  (...)

Some registers carry a name suffix after the _WIDTH/_MASK/_LSB keyword
(<REG>_PRIVILEGED with <REG>_WIDTH_PRIVILEGED); the register is named <REG>.
The value_ macro, when present, spells the field names in their documented
case; otherwise the header spelling is used.

Invoked at build time by the MLIRAIERegDBResources CMake target
(lib/Dialect/AIE/Util/CMakeLists.txt).
"""

import argparse
import json
import re
import sys

# Per-arch header prefix and (prefix -> module) map; module names match
# aie_registers_aie2.json and AIETargetModel.cpp.
ARCH_CONFIGS = {
    "aie4": {
        "prefix": "XAIE4GBL_",
        "modules": [
            ("CORE_INTERNAL_MODULE_", "core"),
            ("CORE_MODULE_", "core"),
            ("MEMORY_MODULE_", "memory"),
            ("MEM_TILE_MODULE_", "memory_tile"),
            ("NOC_MODULE_", "shim"),
            ("PL_MODULE_", "shim"),
            ("UC_MODULE_", "uc"),
            ("NPI_", "npi"),
        ],
    },
}

DEFINE_RE = re.compile(
    r"^#define\s+(\w+)\s+(0x[0-9A-Fa-f]+|\d+)\s*(?:/\*\s*(\w+)\s*\*/)?\s*$"
)
VALUE_RE = re.compile(r"^#define\s+value_(\w+)\(([^)]*)\)")


def register_base(name, width_name, mask_name):
    """Return the <P>_<REG> stem shared by a register's address, _WIDTH and
    _MASK defines, or None if the three lines are not a register block.

    The stem is `name` up to an optional suffix such as _PRIVILEGED. A few
    header entries drop the underscore before the suffix (_WIDTHPRIVILEGED),
    so the suffix itself is not compared."""
    pos = width_name.rfind("_WIDTH")
    if pos <= 0:
        return None
    base = width_name[:pos]
    if not name.startswith(base) or not mask_name.startswith(base + "_MASK"):
        return None
    return base


def parse_header(path, prefix):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]

    parsed = []
    for line in lines:
        m = DEFINE_RE.match(line)
        if m:
            parsed.append(("def", m.group(1), m.group(2), m.group(3)))
            continue
        m = VALUE_RE.match(line)
        if m:
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            parsed.append(("value", m.group(1), params, None))
            continue
        parsed.append(("other", None, None, None))

    registers = []
    i = 0
    while i < len(parsed):
        kind, name, val, comment = parsed[i]
        base = None
        if (
            kind == "def"
            and comment is None
            and name.startswith(prefix)
            and i + 2 < len(parsed)
            and parsed[i + 1][0] == "def"
            and parsed[i + 2][0] == "def"
        ):
            base = register_base(name, parsed[i + 1][1], parsed[i + 2][1])
        if not base:
            # Every <prefix> define belongs to a register block; fail on a layout
            # change rather than silently drop registers.
            if kind in ("def", "value") and name.startswith(prefix):
                raise ValueError(f"{path}: unexpected define {name}")
            i += 1
            continue
        reg = {
            "name": base[len(prefix) :],
            "offset": int(val, 16) if val.startswith("0x") else int(val),
            "width": int(parsed[i + 1][2], 0),
            "fields": {},
            "field_order": [],
            "case": {},
        }
        j = i + 3
        while j < len(parsed) and parsed[j][0] == "def" and parsed[j][3]:
            _, fname, fval, fcomment = parsed[j]
            # Match on the keyword after <REG>_<FIELD>; the trailing suffix is
            # not checked (see register_base).
            stem = base + "_" + fcomment
            rest = fname[len(stem) :] if fname.startswith(stem) else ""
            for key in ("_LSB", "_WIDTH", "_MASK", "_DEFVAL"):
                if rest.startswith(key):
                    field = reg["fields"].setdefault(fcomment, {})
                    if fcomment not in reg["field_order"]:
                        reg["field_order"].append(fcomment)
                    field[key[1:]] = int(fval, 0)
                    break
            else:
                raise ValueError(f"{path}: unexpected field define {fname}")
            j += 1
        if j < len(parsed) and parsed[j][0] == "value":
            for p in parsed[j][2]:
                reg["case"][p.upper()] = p
            j += 1
        registers.append(reg)
        i = j
    return registers


def module_for(name, modules):
    for pfx, module in modules:
        if name.startswith(pfx):
            return module, name[len(pfx) :]
    return None, None


def build_database(header, arch):
    config = ARCH_CONFIGS[arch]
    registers = parse_header(header, config["prefix"])
    modules = {}
    seen = {}
    for reg in registers:
        module, short = module_for(reg["name"], config["modules"])
        if module is None:
            raise ValueError(f"{header}: no module for register {reg['name']}")
        bit_fields = []
        reset = 0
        for fname in reg["field_order"]:
            f = reg["fields"][fname]
            if "LSB" not in f or "WIDTH" not in f:
                raise ValueError(f"{header}: incomplete field {reg['name']}.{fname}")
            lsb, width = f["LSB"], f["WIDTH"]
            msb = lsb + width - 1
            defval = f.get("DEFVAL", 0)
            reset |= defval << lsb
            bit_fields.append(
                {
                    "name": reg["case"].get(fname, fname),
                    "bits": f"{msb}:{lsb}" if width > 1 else f"{lsb}",
                    "bit_range": [lsb, msb],
                    "reset": hex(defval),
                }
            )
        entry = {
            "name": short,
            "offset": f"0x{reg['offset']:08X}",
            "width": reg["width"],
            "reset": f"0x{reset:08X}",
            "bit_fields": bit_fields,
        }
        # Keep one copy of verbatim-repeated blocks; shim NoC/PL registers that
        # share a name at distinct offsets are all kept.
        key = (module, short, reg["offset"])
        if key in seen:
            if seen[key] != entry:
                raise ValueError(f"{header}: conflicting definitions of {short}")
            continue
        seen[key] = entry
        modules.setdefault(module, {"registers": []})["registers"].append(entry)

    for module in modules.values():
        regs = module["registers"]
        # Drop the RESERVED address alias so lookups find the real register.
        named = {r["offset"] for r in regs if r["name"] != "RESERVED"}
        regs[:] = [
            r for r in regs if r["name"] != "RESERVED" or r["offset"] not in named
        ]
        regs.sort(key=lambda r: (int(r["offset"], 16), r["name"]))
    return {
        "architecture": arch,
        "source": "aie-codegen aie-regdb/globalparams register header",
        "modules": dict(sorted(modules.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("-i", "--input", required=True, help="xaie*gbl_params.h")
    parser.add_argument("-a", "--arch", required=True, choices=sorted(ARCH_CONFIGS))
    parser.add_argument("-o", "--output", required=True, help="output JSON file")
    args = parser.parse_args()

    try:
        db = build_database(args.input, args.arch)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    with open(args.output, "w") as f:
        json.dump(db, f, indent=1)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
