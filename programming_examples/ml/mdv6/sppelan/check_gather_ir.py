"""Fail-closed footprint/completion gate for compiled full gather-probe MLIR.

Usage: python3 sppelan/check_gather_ir.py --build-dir BUILD
Checks the fixed four-column proof, not arbitrary MLIR or physical routing.
No AIE installation or hardware is needed to inspect an existing compiled map.
"""
import argparse
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def body_after(text, start):
    opening = text.index('{', start)
    depth = 1
    for end in range(opening + 1, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[opening + 1:end], end + 1
    raise ValueError('unterminated operation body')


def check(text):
    # Strip comments so fixture notes cannot satisfy required operations.
    text = re.sub(r'//[^\n]*', '', text)
    require(not re.search(r'\baie\.core\b', text), 'gather must use zero compute cores')
    tiles = dict((name, (int(col), int(row))) for name, col, row in re.findall(
        r'(%\w+) = aie\.tile\((\d+),\s*(\d+)\)', text))
    mems = {name: col for name, (col, row) in tiles.items() if row == 1}
    require(len(mems) == 4 and set(mems.values()) == set(range(4)),
            'expected exactly four memtiles in columns 0..3')
    require(all(row in (0, 1) for col, row in tiles.values()),
            'unexpected compute-tile allocation')
    buffers = re.findall(r'aie\.buffer\((%\w+)\)\s*\{([^}]*)\}\s*:\s*memref<(\d+)xui16>', text)
    require(len(buffers) == 8 and len(re.findall(r'\baie\.buffer\(', text)) == 8,
            'expected exactly eight recognized uint16 buffers')
    peaks = []
    for mem, col in mems.items():
        allocated = []
        names = set()
        for tile, attrs, elements in buffers:
            if tile != mem:
                continue
            name = re.search(r'sym_name = "([^"]+)"', attrs)
            address = re.search(r'address = (\d+) : i32', attrs)
            require(name is not None and address is not None, 'missing compiled buffer address/name')
            name = name.group(1)
            expected = {f'source_{col}': 51200, f'stripe_{col}': 8192}
            require(name in expected and name not in names and int(elements) == expected[name],
                    f'incorrect buffer contract in column {col}')
            names.add(name)
            begin = int(address.group(1)); end = begin + 2 * int(elements)
            require(end <= 512 * 1024, f'L2 overflow in column {col}')
            allocated.append((begin, end))
        require(names == {f'source_{col}', f'stripe_{col}'}, f'missing buffers in column {col}')
        allocated.sort()
        require(allocated[0][1] <= allocated[1][0], f'overlapping buffers in column {col}')
        peaks.append(allocated[-1][1])
        locks = re.findall(r'aie\.lock\(' + re.escape(mem) + r',\s*(\d+)\)', text)
        require(len(locks) == 7 and set(map(int, locks)) == {0, 1, 2, 3, 4, 16, 17},
                f'incorrect lock allocation in column {col}')
        programs = list(re.finditer(r'aie\.memtile_dma\(' + re.escape(mem) + r'\)', text))
        require(len(programs) == 1, f'expected one memtile DMA program in column {col}')
        program, _ = body_after(text, programs[0].end())
        ids = re.findall(r'(?<!next_)\bbd_id = (\d+) : i32', program)
        require(len(ids) == 7 and len(set(ids)) == 7 and all(int(i) < 48 for i in ids),
                f'incorrect static descriptor allocation in column {col}')
        require(len(re.findall(r'\baie\.dma_bd\(', program)) == 7,
                f'incorrect descriptor count in column {col}')
        channels = re.findall(r'aie\.dma_start\((S2MM|MM2S),\s*(\d+)', program)
        require(sorted(channels) == sorted([('S2MM', str(i)) for i in range(5)]
                                          + [('MM2S', str(col)), ('MM2S', '4')]),
                f'incorrect DMA channel allocation in column {col}')
    starts = list(re.finditer(r'\baie\.runtime_sequence\(', text))
    require(len(starts) == 1, 'expected one runtime sequence')
    runtime, _ = body_after(text, starts[0].end())
    tasks = []
    for match in re.finditer(r'(%\w+) = aiex\.dma_configure_task\((%\w+),\s*(S2MM|MM2S),\s*0\)', runtime):
        _, end = body_after(runtime, match.end())
        token = re.match(r'\s*\{issue_token = true\}', runtime[end:]) is not None
        tasks.append((*match.groups(), token))
    require(len(tasks) == 8, 'expected four input and four output runtime tasks')
    outputs = [name for name, tile, direction, token in tasks if direction == 'S2MM' and token]
    inputs = [name for name, tile, direction, token in tasks if direction == 'MM2S']
    for direction in ('S2MM', 'MM2S'):
        require({tiles.get(tile) for name, tile, d, token in tasks if d == direction}
                == {(col, 0) for col in range(4)}, f'incorrect {direction} shim coverage')
    awaits = re.findall(r'aiex\.dma_await_task\((%\w+)\)', runtime)
    require(len(outputs) == 4 and awaits == outputs, 'all four output tasks must be awaited')
    require(re.findall(r'aiex\.dma_free_task\((%\w+)\)', runtime) == inputs,
            'expected exactly four input-task frees')
    require(runtime.index('aiex.dma_free_task') > runtime.rindex('aiex.dma_await_task'),
            'input tasks freed before all output completion')
    return dict(status='PASS', compute_cores=0, memtiles=4,
                resident_source_bytes_per_memtile=102400, stripe_bytes_per_memtile=16384,
                allocated_bytes_per_memtile=118784, maximum_allocation_end=max(peaks),
                static_bds_per_memtile=7, locks_per_memtile=7,
                s2mm_channels_per_memtile=5, mm2s_channels_per_memtile=2,
                runtime_output_awaits=4, physical_routing_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiled_mlir', type=Path, nargs='?')
    parser.add_argument('--build-dir', type=Path,
                        help='directory containing gather_probe.mlir.prj')
    args = parser.parse_args()
    if (args.compiled_mlir is None) == (args.build_dir is None):
        parser.error('provide exactly one compiled MLIR path or --build-dir')
    path = args.compiled_mlir if args.compiled_mlir is not None else (
        args.build_dir / 'gather_probe.mlir.prj' / 'input_with_addresses.mlir')
    print(json.dumps(check(path.read_text()), sort_keys=True))
