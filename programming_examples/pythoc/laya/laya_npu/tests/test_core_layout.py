"""Every prebuilt core agrees with its graph template on buffer addresses.

A core ELF reaches its buffers through absolute symbol addresses fixed when the core
was built; the graph template fixes where DMA puts them. If the two disagree, the
graph still compiles and runs, but the core reads the wrong memory.
"""

import json
import re
import struct
from pathlib import Path

import pytest

pytestmark = pytest.mark.build
ROOT = Path(__file__).resolve().parents[1]
RECIPES = json.loads((ROOT / "src/graphs/recipes.json").read_text())
TILE_MEMORY = 0x70000  # a core's own data memory in its address map


def symbols(raw):
    shoff = struct.unpack_from("<I", raw, 32)[0]
    shent, shnum = struct.unpack_from("<HH", raw, 46)
    secs = [struct.unpack_from("<10I", raw, shoff + i * shent) for i in range(shnum)]
    found = {}
    for table in secs:
        if table[1] != 2:
            continue
        strtab = secs[table[6]]
        names = raw[strtab[4] : strtab[4] + strtab[5]]
        for offset in range(table[4], table[4] + table[5], table[9]):
            name, value, _, _, _, index = struct.unpack_from("<IIIBBH", raw, offset)
            if index:
                found[names[name : names.index(b"\0", name)].decode()] = value
    return found


def core_paths(build_dir, build_record, recipe):
    def path(target, name):
        items = build_record["core_builds"][target]["cores"]
        if isinstance(items, list):
            items = {str(i): item for i, item in enumerate(items)}
        return build_dir / "cores" / target / items[name]["path"]

    swap = recipe.get("swap_cores")
    paths = {}
    for core in recipe["cores"]:
        target, name = core.split("/", 1)
        if swap and target == swap[0]:
            target = swap[1]
        paths[core] = path(target, name)
    return paths


@pytest.mark.parametrize("target", sorted(RECIPES))
def test_core_buffers_match_template(build_dir, build_record, target):
    recipe = RECIPES[target]
    paths = core_paths(build_dir, build_record, recipe)
    text = (build_dir / build_record["templates"][recipe["devices"]]).read_text()
    checked, wrong = 0, []
    for device in text.split("aie.device(")[1:]:
        buffers = {}
        for tile, address, name in re.findall(
            r'aie\.buffer\(%(\w+)\) \{address = (\d+) : i32[^}]*sym_name = "(\w+)"',
            device,
        ):
            buffers.setdefault(tile, {})[name] = int(address)
        for tile, core in re.findall(
            r'aie\.core\(%(\w+)\).*?elf_file = "@CORE_([^@]+)@"', device, re.S
        ):
            elf = symbols(paths[core].read_bytes())
            for name, address in buffers.get(tile, {}).items():
                if name in elf:
                    checked += 1
                    if elf[name] != TILE_MEMORY + address:
                        wrong.append((core, name, address, elf[name] - TILE_MEMORY))
    assert checked
    assert not wrong, wrong[:5]
