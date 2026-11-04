# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the compiled configurations needed by the cross-ELF dispatch cases."""

import re
import sys
from pathlib import Path

ids = {}
for name in ("a", "b", "a_copy"):
    assembly = Path(f"project_{name}/full_elf_cert.mlir.asm")
    loads = re.findall(r"^\s*LOAD_PDI\s+(\d+),", assembly.read_text(), re.MULTILINE)
    if len(loads) != 1:
        sys.exit(f"{assembly}: expected one LOAD_PDI, found {len(loads)}")
    ids[name] = int(loads[0])
    if not 0 < ids[name] < 0xFFFFFFFF:
        sys.exit(f"{assembly}: invalid or reserved PDI id {ids[name]}")
    print(f"{name}.elf: PDI id {ids[name]}")

if ids["a"] != ids["a_copy"]:
    sys.exit("a and a_copy must share a PDI id to exercise a skipped load")
if ids["a"] == ids["b"]:
    sys.exit("a and b must have different PDI ids to exercise reconfiguration")
