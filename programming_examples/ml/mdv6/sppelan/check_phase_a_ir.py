"""Focused one-worker phase-A compiled-map and ELF gate (no hardware).

Run with --build-dir DIR after Makefile.phase_a_shard builds both programs.
This checks allocations and completion, not a general MLIR verifier or proof
of arithmetic correctness. ELF allocation sections must not hide data/BSS.
"""
import argparse
import json
from pathlib import Path
import re
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def memref_size(shape):
    parts = shape.split('x')
    widths = {'ui8': 1, 'i8': 1, 'ui16': 2, 'i16': 2, 'bf16': 2,
              'f16': 2, 'ui32': 4, 'i32': 4, 'f32': 4, 'ui64': 8,
              'i64': 8, 'f64': 8}
    require(parts[-1] in widths and all(p.isdecimal() and int(p) > 0 for p in parts[:-1]),
            f'unknown or nonstatic memref type {shape}')
    size = widths[parts[-1]]
    for dim in parts[:-1]:
        size *= int(dim)
    return size


def check_map(text, pool_only=False):
    text = re.sub(r'//[^\n]*', '', text)
    tiles = {name: (int(c), int(r)) for name, c, r in re.findall(
        r'(%\w+) = aie\.tile\((\d+),\s*(\d+)\)', text)}
    cores = re.findall(r'aie\.core\((%\w+)\)', text)
    stacks = re.findall(r'stack_size = (\d+) : i32', text)
    require(len(cores) == len(stacks) == 1, 'expected one core and explicit stack')
    core = cores[0]; stack = int(stacks[0])
    require(core in tiles and tiles[core][1] >= 2 and 0 < stack <= 65536,
            'invalid core/stack contract')
    allocations = {core: [(0, stack, 'stack', None)]}
    records = re.findall(r'(%\w+) = aie\.buffer\((%\w+)\)\s*\{([^}]*)\}\s*:\s*memref<([^>]+)>', text)
    require(len(records) == len(re.findall(r'\baie\.buffer\(', text)) and records,
            'unrecognized buffer declaration')
    names = {}
    for value, tile, attrs, shape in records:
        address = re.search(r'address = (\d+) : i32', attrs)
        name = re.search(r'sym_name = "([^"]+)"', attrs)
        require(address is not None and name is not None and tile in tiles,
                'buffer missing compiled address/name/tile')
        require(tile == core or tiles[tile][1] == 1, 'hidden neighbor-core or shim allocation')
        require(name.group(1) not in names, 'duplicate buffer name')
        begin = int(address.group(1)); end = begin + memref_size(shape)
        allocations.setdefault(tile, []).append((begin, end, name.group(1), shape))
        names[name.group(1)] = (tile, shape)
        require(end <= (524288 if tiles[tile][1] == 1 else 65536), 'buffer exceeds tile memory')
    expected = {'shard_planes_buff_0': '12800xui16', 'shard_metadata_buff_0': '32xui16',
                'shard_input_cons_buff_0': ('3200xui16' if pool_only else '4096xui16')}
    if not pool_only:
        expected['shard_weights_cons_buff_0'] = '2112xui16'
    for name, shape in expected.items():
        require(names.get(name) == (core, shape), f'incorrect required buffer {name}')
    require(sum(shape == '12800xui16' for tile, shape in names.values()) == 1,
            'feature planes duplicated')
    require(sum(shape == '32xui16' for tile, shape in names.values()) == 1,
            'metadata duplicated')
    # Only small compiler synchronization buffers may accompany the declared
    # interface. This rejects hidden feature copies, even with a reshaped type.
    extras = [r for r in records if re.search(r'sym_name = "([^"]+)"', r[2]).group(1) not in expected]
    require(all(r[1] == core and r[3] in ('3xi32', '4xi32') for r in extras)
            and len(extras) == 1, 'unexpected hidden scratch/allocation')
    for tile, buffers in allocations.items():
        buffers.sort()
        require(all(a[1] <= b[0] for a, b in zip(buffers, buffers[1:])),
                f'overlapping buffers or stack on {tile}')
    signature = re.findall(r'aie\.runtime_sequence\(([^\n]*)\)\s*\{', text)
    require(len(signature) == 1, 'expected one runtime signature')
    abi = re.findall(r'memref<([^>]+)>', signature[0])
    expected_abi = ['3200xui16', '12800xui16', '32xui16'] if pool_only else [
        '102400xui16', '2112xui16', '12800xui16', '32xui16']
    require(abi == expected_abi, 'incorrect host ABI')
    runtime = text[text.index('aie.runtime_sequence'):]
    tasks = re.findall(r'(%\w+) = aiex\.dma_configure_task_for @(\w+) \{.*?\n\s*\}(?: (\{issue_token = true[^}]*\}))?', runtime, re.S)
    outputs = [value for value, name, token in tasks if name in ('shard_planes_shim_alloc', 'shard_metadata_shim_alloc') and token]
    inputs = [value for value, name, token in tasks if name in ('shard_input_shim_alloc', 'shard_weights_shim_alloc') and not token]
    require(len(tasks) == len(expected_abi) and len(outputs) == 2 and len(inputs) == len(expected_abi) - 2,
            'incorrect input/output DMA tasks')
    require(re.findall(r'aiex\.dma_await_task\((%\w+)\)', runtime) == outputs,
            'both feature and metadata outputs must be awaited')
    require(re.findall(r'aiex\.dma_free_task\((%\w+)\)', runtime) == inputs,
            'incorrect input task frees')
    require(runtime.index('aiex.dma_free_task') > runtime.rindex('aiex.dma_await_task'),
            'input freed before both output completions')
    return dict(core=core, stack_bytes=stack, host_abi=abi,
                maximum_l1_end=max(b[1] for b in allocations[core]),
                allocations={tile: [dict(begin=b, end=e, name=n, type=t) for b, e, n, t in rows]
                             for tile, rows in allocations.items()}, output_awaits=2)


def check_elf(data):
    require(len(data) >= 52 and data[:6] == b'\x7fELF\x01\x01', 'expected little-endian ELF32')
    offset = struct.unpack_from('<I', data, 32)[0]
    stride, count = struct.unpack_from('<HH', data, 46)
    require(stride >= 40 and count > 0 and offset + stride * count <= len(data), 'invalid ELF section table')
    text = initialized = bss = 0
    for i in range(count):
        _, kind, flags, address, file_offset, size, *_ = struct.unpack_from('<10I', data, offset + stride * i)
        if not flags & 2:
            continue
        if flags & 4:
            text += size
        elif kind == 8:
            bss += size
        else:
            initialized += size
    require(text > 0, 'ELF missing executable text')
    require(initialized == bss == 0, 'ELF contains hidden data/BSS outside compiled buffer map')
    return dict(text_bytes=text, data_bytes=initialized, bss_bytes=bss)


def check_build(root):
    reports = {}
    for name, pool in [('phase_a_shard', False), ('phase_a_pool', True)]:
        project = root / f'{name}.mlir.prj'
        report = check_map((project / 'input_with_addresses.mlir').read_text(), pool)
        elfs = list(project.glob('main_core_*.elf'))
        require(len(elfs) == 1, f'{name}: expected exactly one core ELF')
        report['elf'] = check_elf(elfs[0].read_bytes())
        reports[name] = report
    return dict(status='PASS', programs=reports, physical_routing_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True, type=Path)
    print(json.dumps(check_build(parser.parse_args().build_dir), sort_keys=True))
