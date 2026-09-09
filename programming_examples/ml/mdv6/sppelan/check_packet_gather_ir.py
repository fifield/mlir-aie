"""Actual-map gate for resident four-column packet aggregation then gather.

Checks explicit resources, DMA ownership, stream order and logical endpoints.
It does not independently decode the physical router or measure stack usage.
"""
import argparse
import itertools
import json
from pathlib import Path
import re

try:
    from .check_gather_ir import body_after
    from .check_phase_a_ir import check_elf, memref_size, require
    from .check_packet_aggregate_ir import dma_program
except ImportError:
    from check_gather_ir import body_after
    from check_phase_a_ir import check_elf, memref_size, require
    from check_packet_aggregate_ir import dma_program


SOURCE_DIMS = [(25, 128), (4, 12800), (4, 3200), (128, 1)]
SCATTER_DIMS = [(4, 8), (4, 128), (16, 512), (8, 1)]


def addresses(offset, dimensions):
    return [offset + sum(i * stride for i, (_, stride) in zip(index, dimensions))
            for index in itertools.product(*(range(n) for n, _ in dimensions))]


def check_order():
    source = addresses(0, SOURCE_DIMS)
    expected = [worker*12800 + level*3200 + stripe*128 + pixel*8 + lane
                for stripe in range(25) for worker in range(4) for level in range(4)
                for pixel in range(16) for lane in range(8)]
    require(source == expected and sorted(source) == list(range(51200)), 'incorrect native source stream order')
    coverage = []
    for col in range(4):
        scatter = addresses(col*32, SCATTER_DIMS)
        expected = [pixel*512 + level*128 + col*32 + worker*8 + lane
                    for worker in range(4) for level in range(4) for pixel in range(16) for lane in range(8)]
        require(scatter == expected, 'incorrect pool-first destination stream order')
        coverage.extend(scatter)
    require(sorted(coverage) == list(range(8192)), 'scatter must partition one entire concat stripe')


def descriptor(block, buffers):
    found = re.findall(r'aie\.dma_bd\((%\w+) : memref<([^>]+)>, (\d+), (\d+)(?:, (\[[^\]]+\]))?\)', block)
    require(len(found) == 1 and len(re.findall(r'\baie\.dma_bd\(', block)) == 1, 'unknown descriptor syntax')
    value, shape, offset, length, dims_text = found[0]
    dims = [(int(n), int(s)) for n, s in re.findall(r'<size = (\d+), stride = (\d+)>', dims_text)]
    offset, length = int(offset), int(length)
    require(value in buffers and buffers[value][1] == shape, 'descriptor buffer/type mismatch')
    if dims:
        require(len(addresses(offset, dims)) == length, 'descriptor dimension/length mismatch')
        last = offset + sum((n-1)*stride for n, stride in dims)
    else:
        last = offset + length - 1
    require(length > 0 and (last+1)*memref_size(shape.split('x')[-1]) <= memref_size(shape), 'descriptor exceeds actual allocation')
    locks = re.findall(r'aie\.use_lock\((%\w+), (\w+), (\d+)\)', block)
    require(len(locks) == 2 and [(a, n) for _, a, n in locks] == [('AcquireGreaterEqual', '1'), ('Release', '1')],
            'every permanent BD needs exactly one acquire/release credit')
    require(block.index('AcquireGreaterEqual') < block.index('aie.dma_bd') < block.index('Release'), 'credit order does not bracket DMA')
    packets = re.findall(r'packet = #aie.packet_info<pkt_type = (\d+), pkt_id = (\d+)>', block)
    require(not packets or len(packets) == 1 and packets[0][0] == '0', 'invalid packet header')
    return value, offset, length, dims, locks[0][0], locks[1][0], None if not packets else int(packets[0][1])


def check_map(text):
    text = re.sub(r'//[^\n]*', '', text)
    tiles = {v: (int(c), int(r)) for v, c, r in re.findall(r'(%\w+) = aie\.tile\((\d+), (\d+)\)', text)}
    require(len(tiles) == 24 and set(tiles.values()) == set(itertools.product(range(4), range(6))), 'expected four complete columns')
    tile_at = {coord: value for value, coord in tiles.items()}
    cores = list(re.finditer(r'aie\.core\((%\w+)\)', text))
    require(len(cores) == 16 and {tiles[m[1]] for m in cores} == set(itertools.product(range(4), range(2, 6))), 'expected sixteen workers')
    allocations = {tile: [] for tile in tiles}
    for match in cores:
        _, end = body_after(text, match.end())
        attrs = re.match(r'\s*\{([^}]*)\}', text[end:])
        stack = re.search(r'stack_size = (\d+) : i32', attrs[1] if attrs else '')
        require(stack and int(stack[1]) == 4096, 'expected explicit4096-byte stacks')
        allocations[match[1]].append((0, 4096, 'stack', None))
    records = re.findall(r'(%\w+) = aie\.buffer\((%\w+)\)\s*\{([^}]*)\}\s*:\s*memref<([^>]+)>', text)
    require(len(records) == 48 and len(re.findall(r'\baie\.buffer\(', text)) == 48, 'unexpected buffer count/syntax')
    buffers = {}
    for value, tile, attrs, shape in records:
        name = re.search(r'sym_name = "([^\"]+)"', attrs); address = re.search(r'address = (\d+) : i32', attrs)
        require(tile in tiles and tiles[tile][1] > 0 and name and address and value == '%' + name[1] and value not in buffers, 'unknown or aliased buffer')
        begin = int(address[1]); end = begin + memref_size(shape)
        require(end <= (524288 if tiles[tile][1] == 1 else 65536), 'actual allocation exceeds tile')
        allocations[tile].append((begin, end, name[1], shape)); buffers[value] = (tile, shape)
    for rows in allocations.values():
        rows.sort(); require(all(a[1] <= b[0] for a, b in zip(rows, rows[1:])), 'buffer/stack overlap')
    return _check_ownership(text, tiles, tile_at, cores, allocations, buffers)


def _check_ownership(text, tiles, tile_at, cores, allocations, buffers):
    expected_buffers = {}; expected_locks = {}; expected_packets = []
    def expect(block, name, offset, length, acquire, release, packet=None, dims=()):
        require(descriptor(block, buffers) == ('%' + name, offset, length, list(dims), '%' + acquire, '%' + release, packet),
                'incorrect descriptor geometry or ownership chain')
    for col in range(4):
        mt = tile_at[col, 1]
        for name, shape in [(f'packet_source_{col}', '51200xui16'), (f'packet_aggregate_{col}', '51200xui16'),
                            (f'grant_token_{col}', '32xui16'), (f'stripe_{col}', '8192xui16')]:
            expected_buffers['%' + name] = (mt, shape)
        expected_locks[mt] = [(f'stage_empty_{col}', 0, 1), (f'stage_ready_{col}', 1, 0), (f'output_ready_{col}', 10, 0)] + [
            (f'{name}_{col}_{i}', base+i, int(name == 'stripe_turn' and i == 0))
            for name, base, count in [('grant_ready', 2, 4), ('receive_ready', 6, 4), ('payload_ready', 11, 3), ('stripe_turn', 14, 5)]
            for i in range(count)]
        chains = dma_program(text, mt, 'memtile_dma')
        expected_channels = {('S2MM', i): 1 for i in range(6)}
        expected_channels.update({('S2MM', 4): 4, ('MM2S', 5): 8, ('MM2S', col): 1, ('MM2S', 4): 1})
        require({key: len(value) for key, value in chains.items()} == expected_channels, 'incorrect nineteen-BD/channel budget')
        expect(chains['S2MM', 5][0], f'packet_source_{col}', 0, 51200, f'stage_empty_{col}', f'stage_ready_{col}')
        expect(chains['MM2S', col][0], f'packet_aggregate_{col}', 0, 51200, f'output_ready_{col}', f'stage_empty_{col}', dims=SOURCE_DIMS)
        expect(chains['MM2S', 4][0], f'stripe_{col}', 0, 8192, f'stripe_turn_{col}_4', f'stripe_turn_{col}_0')
        for source in range(4):
            expect(chains['S2MM', source][0], f'stripe_{col}', source*32, 2048,
                   f'stripe_turn_{col}_{source}', f'stripe_turn_{col}_{source+1}', dims=SCATTER_DIMS)
        for index, worker in enumerate(reversed(range(4))):
            expect(chains['MM2S', 5][index], f'packet_source_{col}', worker*12800, 12800,
                   f'stage_ready_{col}' if worker == 3 else f'payload_ready_{col}_{worker}',
                   f'grant_ready_{col}_0' if worker == 0 else f'payload_ready_{col}_{worker-1}', 1 << worker)
        for worker in range(4):
            suffix = f'{col}_{worker}'; ct = tile_at[col, worker+2]
            expect(chains['MM2S', 5][4+worker], f'grant_token_{col}', 0, 32, f'grant_ready_{suffix}', f'receive_ready_{suffix}', 1 << worker)
            expect(chains['S2MM', 4][worker], f'packet_aggregate_{col}', worker*12800, 12800, f'receive_ready_{suffix}',
                   f'grant_ready_{col}_{worker+1}' if worker < 3 else f'output_ready_{col}')
            expected_buffers[f'%worker_planes_{suffix}'] = (ct, '12800xui16')
            expected_buffers[f'%worker_grant_{suffix}'] = (ct, '32xui16')
            expected_locks[ct] = [(f'{name}_{suffix}', i, init) for i, (name, init) in enumerate([
                ('feature_empty', 1), ('feature_ready', 0), ('grant_empty', 1), ('worker_grant_ready', 0), ('send_ready', 0)])]
            local = dma_program(text, ct, 'mem')
            require({key: len(value) for key, value in local.items()} == {('S2MM', 1): 2, ('MM2S', 0): 1}, 'incorrect worker DMA schedule')
            expect(local['S2MM', 1][0], f'worker_planes_{suffix}', 0, 12800, f'feature_empty_{suffix}', f'feature_ready_{suffix}')
            expect(local['S2MM', 1][1], f'worker_grant_{suffix}', 0, 32, f'grant_empty_{suffix}', f'worker_grant_ready_{suffix}')
            expect(local['MM2S', 0][0], f'worker_planes_{suffix}', 0, 12800, f'send_ready_{suffix}', f'feature_empty_{suffix}', 16+worker)
            match = next(m for m in cores if m[1] == ct); core_body, _ = body_after(text, match.end())
            require(re.findall(r'aie\.use_lock\((%\w+), (\w+), (\d+)\)', core_body) == [
                (f'%feature_ready_{suffix}', 'AcquireGreaterEqual', '1'), (f'%worker_grant_ready_{suffix}', 'AcquireGreaterEqual', '1'),
                (f'%send_ready_{suffix}', 'Release', '1'), (f'%grant_empty_{suffix}', 'Release', '1')], 'incorrect worker grant credits')
            require(core_body.index('AcquireGreaterEqual') < core_body.index('func.call @packet_gather_tag') <
                    core_body.index(f'aie.use_lock(%worker_grant_ready_{suffix}'), 'tag must follow input and precede output grant')
            expected_packets.extend([(1 << worker, mt, 5, ct, 1), (16+worker, ct, 0, mt, 4)])
    require(buffers == expected_buffers, 'named allocation missing or borrowing another tile')
    locks = re.findall(r'(%\w+) = aie\.lock\((%\w+), (\d+)\)\s*\{init = (\d+) : i32, sym_name = "([^\"]+)"\}', text)
    require(len(locks) == 156 and len(re.findall(r'\baie\.lock\(', text)) == 156, 'incorrect lock count')
    for tile, expected in expected_locks.items():
        actual = [(name, int(index), int(init)) for value, owner, index, init, name in locks if owner == tile and value == '%' + name]
        require(sorted(actual) == sorted(expected), 'incorrect initial locks or physical IDs')
    expected_flows = []
    for col in range(4):
        expected_flows.extend([(tile_at[col, 0], 0, tile_at[col, 1], 5), (tile_at[col, 1], 4, tile_at[col, 0], 0)])
        expected_flows.extend((tile_at[col, 1], col, tile_at[dest, 1], col) for dest in range(4))
    flows = [(s, int(sc), d, int(dc)) for s, sc, d, dc in re.findall(r'aie\.flow\((%\w+), DMA : (\d+), (%\w+), DMA : (\d+)\)', text)]
    require(sorted(flows) == sorted(expected_flows) and len(re.findall(r'\baie\.flow\(', text)) == 24, 'incorrect circuit endpoint set')
    packets = re.findall(r'aie\.packet_flow\((\d+)\)\s*\{\s*aie\.packet_source<(%\w+), DMA : (\d+)>\s*'
                         r'aie\.packet_dest<(%\w+), DMA : (\d+)>\s*\}\s*\{keep_pkt_header = false\}', text)
    require(sorted((int(i), s, int(sc), d, int(dc)) for i, s, sc, d, dc in packets) == sorted(expected_packets), 'incorrect packet endpoint/header set')
    require(len(re.findall(r'aie\.packet_source<%\w+, DMA', text)) == 32, 'extra data packet route')
    controls = re.findall(r'aie\.packet_flow\(15\)\s*\{\s*aie\.packet_source<(%\w+), TileControl : 0>\s*'
                          r'aie\.packet_dest<(%\w+), South : 0>\s*\}\s*\{keep_pkt_header = true, priority_route = true\}', text)
    require(sorted(controls) == sorted((tile_at[c, 0], tile_at[c, 0]) for c in range(4)) and
            len(re.findall(r'\baie\.packet_flow\(', text)) == 36, 'incorrect compiler control packet routes or extra packet flow')
    check_order()
    _check_runtime(text, tiles)
    return dict(status='PASS', compute_cores=16, memtiles=4, static_bds_per_memtile=19, locks_per_memtile=19,
                static_bds_per_worker=3, locks_per_worker=5, s2mm_channels_per_memtile=6, mm2s_channels_per_memtile=3,
                buffer_bytes_per_memtile=221248, output_awaits=4, logical_circuit_routes=24, logical_packet_routes=32,
                physical_routing_verified=False,
                allocations={tile: [dict(begin=b, end=e, name=n, type=t) for b, e, n, t in rows] for tile, rows in allocations.items()})


def _check_runtime(text, tiles):
    require(len(re.findall(r'\baie\.runtime_sequence\(', text)) == 1, 'expected exactly one runtime sequence')
    match = re.search(r'aie\.runtime_sequence\(([^\n]*)\)', text)
    require(match and re.findall(r'memref<([^>]+)>', match[1]) == ['204800xui16', '819200xui16'], 'incorrect frame ABI')
    arguments = re.findall(r'(%\w+): memref<[^>]+>', match[1])
    runtime, _ = body_after(text, match.end())
    tasks = list(re.finditer(r'(%\w+) = aiex\.dma_configure_task\((%\w+), (MM2S|S2MM), 0\)', runtime))
    require(len(tasks) == 8 and len(re.findall(r'\baiex\.dma_configure_task\(', runtime)) == 8, 'expected eight shim tasks')
    inputs, outputs = [], []
    for index, task in enumerate(tasks):
        value, tile, direction = task.groups(); col = index % 4; output = index >= 4
        require(tiles.get(tile) == (col, 0) and direction == ('S2MM' if output else 'MM2S'), 'incorrect shim task column/direction')
        body, end = body_after(runtime, task.end())
        token = re.match(r'\s*\{issue_token = true\}', runtime[end:]) is not None
        require(token == output, 'incorrect shim task completion token')
        size, length, arg = (819200, 204800, arguments[1]) if output else (204800, 51200, arguments[0])
        pattern = rf'aie\.dma_bd\({re.escape(arg)} : memref<{size}xui16>, {col*length}, {length}\)'
        require(len(re.findall(r'\baie\.dma_bd\(', body)) == 1 and re.search(pattern, body), 'incorrect shim BD offset/length/type/stride')
        (outputs if output else inputs).append(value)
    starts = re.findall(r'aiex\.dma_start_task\((%\w+)\)', runtime)
    require(starts == outputs + inputs, 'start every output then every input exactly once')
    require(re.findall(r'aiex\.dma_await_task\((%\w+)\)', runtime) == outputs and
            re.findall(r'aiex\.dma_free_task\((%\w+)\)', runtime) == inputs, 'all four output awaits and input frees required')
    require(runtime.rindex('aiex.dma_start_task') < runtime.index('aiex.dma_await_task') and
            runtime.rindex('aiex.dma_await_task') < runtime.index('aiex.dma_free_task'), 'incorrect start/await/free ordering')


def check_build(root):
    project = root / 'packet_gather.mlir.prj'
    report = check_map((project / 'input_with_addresses.mlir').read_text())
    elfs = sorted(project.glob('main_core_*.elf'))
    require({p.name for p in elfs} == {f'main_core_{c}_{r}.elf' for c in range(4) for r in range(2, 6)}, 'expected all sixteen distinct worker ELFs')
    report['elfs'] = {p.name: check_elf(p.read_bytes()) for p in elfs}
    require(all(elf['text_bytes'] <= 131072 for elf in report['elfs'].values()), 'ELF text exceeds program memory')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True, type=Path)
    print(json.dumps(check_build(parser.parse_args().build_dir), sort_keys=True))
