"""Focused actual-resource/ownership/scatter gate for the packet stripe-join proof.

This is not a general MLIR verifier or an independent physical-route decoder.
Run after building with --build-dir DIR; no hardware or AIE Python is needed.
"""
import argparse
import json
from pathlib import Path
import re

try:
    from .check_gather_ir import body_after
    from .check_phase_a_ir import check_elf, memref_size, require
except ImportError:
    from check_gather_ir import body_after
    from check_phase_a_ir import check_elf, memref_size, require


try:
    from .check_packet_aggregate_ir import dma_program, descriptor as contiguous_descriptor
except ImportError:
    from check_packet_aggregate_ir import dma_program, descriptor as contiguous_descriptor


def scatter_addresses(worker, offset, dimensions):
    require(worker in range(4) and offset == 16 * worker and dimensions == [(16, 64), (16, 1)],
            'incorrect worker scatter geometry')
    addresses = [offset + pixel * 64 + lane for pixel in range(16) for lane in range(16)]
    require(len(set(addresses)) == 256 and min(addresses) >= 0 and max(addresses) < 1024,
            'scatter overlap or out of bounds')
    return addresses


def descriptor(block):
    # Validate real strided extents, then flatten syntax only for the common
    # payload/lock parser. Buffer size, offset and transfer length stay actual.
    pattern = r'(aie\.dma_bd\(%packet_aggregate : memref<1024xui16>, (\d+), 256), (\[[^\]]+\])\)'
    match = re.search(pattern, block)
    if match:
        dimensions = [(int(n), int(stride)) for n, stride in re.findall(
            r'<size = (\d+), stride = (\d+)>', match[3])]
        scatter_addresses(int(match[2]) // 16, int(match[2]), dimensions)
        block = block[:match.start()] + match[1] + ')' + block[match.end():]
    return contiguous_descriptor(block)


def check_map(text):
    text = re.sub(r'//[^\n]*', '', text)
    tiles = {v: (int(c), int(r)) for v, c, r in re.findall(
        r'(%\w+) = aie\.tile\((\d+),\s*(\d+)\)', text)}
    require(set(tiles.values()) == {(0, r) for r in range(6)} and len(tiles) == 6,
            'expected one shim, one memtile, four own-column cores')
    core_matches = list(re.finditer(r'aie\.core\((%\w+)\)', text))
    require(len(core_matches) == 4 and {tiles[m[1]] for m in core_matches} == {(0, r) for r in range(2, 6)},
            'expected four cores at rows 2..5')
    allocations = {tile: [] for tile in tiles}
    for match in core_matches:
        _, end = body_after(text, match.end())
        attrs = re.match(r'\s*\{([^}]*)\}', text[end:])
        stack = re.search(r'stack_size = (\d+) : i32', attrs[1] if attrs else '')
        require(stack is not None and int(stack[1]) == 4096, 'expected explicit 4096-byte worker stack')
        allocations[match[1]].append((0, int(stack[1]), 'stack', None))
    records = re.findall(r'(%\w+) = aie\.buffer\((%\w+)\)\s*\{([^}]*)\}\s*:\s*memref<([^>]+)>', text)
    require(len(records) == len(re.findall(r'\baie\.buffer\(', text)), 'unknown buffer declaration')
    buffers = {}
    for value, tile, attrs, shape in records:
        address = re.search(r'address = (\d+) : i32', attrs)
        name = re.search(r'sym_name = "([^"]+)"', attrs)
        require(tile in tiles and tiles[tile][1] > 0 and address and name, 'unknown buffer tile/address/name')
        begin = int(address[1]); end = begin + memref_size(shape)
        require(end <= (524288 if tiles[tile][1] == 1 else 65536), 'tile allocation overflow')
        allocations[tile].append((begin, end, name[1], shape))
        buffers[value] = (tile, shape)
    require(len(buffers) == 11, 'expected three memtile and two buffers per worker')
    for tile, rows in allocations.items():
        rows.sort()
        require(all(a[1] <= b[0] for a, b in zip(rows, rows[1:])), 'stack/buffer overlap')
        expected = [] if tiles[tile][1] == 0 else (
            ['32xui16', '1024xui16', '1024xui16'] if tiles[tile][1] == 1 else ['256xui16', '32xui16'])
        require(sorted(shape for _, _, _, shape in rows if shape) == sorted(expected), 'hidden or missing tile buffers')
    memtile = next(tile for tile, coord in tiles.items() if coord == (0, 1))
    for name, shape in [('packet_source', '1024xui16'), ('packet_aggregate', '1024xui16'), ('grant_token', '32xui16')]:
        require(buffers.get('%' + name) == (memtile, shape), 'incorrect named memtile allocation')
    chains = dma_program(text, memtile, 'memtile_dma')
    require(set(chains) == {('S2MM', 5), ('S2MM', 4), ('MM2S', 5), ('MM2S', 1)},
            'memtile channels conflict with reserved future gather budget')
    require({key: len(value) for key, value in chains.items()} == {
        ('S2MM', 5): 1, ('S2MM', 4): 4, ('MM2S', 5): 8, ('MM2S', 1): 1}, 'unexpected memtile descriptor schedule')
    for blocks in chains.values():
        for block in blocks:
            value, _, _, _ = descriptor(block)
            require(value in buffers and buffers[value][0] == memtile, 'memtile DMA borrows nonlocal buffer')
    def expect(block, name, offset, length, acquire=None, release=None, packet=None):
        locks = ([] if acquire is None else [('%' + acquire, 'AcquireGreaterEqual')]) + (
            [] if release is None else [('%' + release, 'Release')])
        require(descriptor(block) == ('%' + name, offset, length, locks), 'descriptor payload/ownership mismatch')
        packets = re.findall(r'packet = #aie.packet_info<pkt_type = (\d+), pkt_id = (\d+)>', block)
        require(packets == ([] if packet is None else [('0', str(packet))]), 'incorrect descriptor packet header')
        if acquire:
            require(block.index('AcquireGreaterEqual') < block.index('aie.dma_bd'), 'acquire must precede DMA')
        if release:
            require(block.index('Release') > block.index('aie.dma_bd'), 'release must follow DMA completion')
    expect(chains['S2MM', 5][0], 'packet_source', 0, 1024, 'stage_empty', 'stage_ready')
    expect(chains['MM2S', 1][0], 'packet_aggregate', 0, 1024, 'output_ready', 'stage_empty')
    for index, worker in enumerate(reversed(range(4))):
        expect(chains['MM2S', 5][index], 'packet_source', worker * 256, 256,
               'stage_ready' if worker == 3 else f'payload_ready_{worker}',
               'grant_ready_0' if worker == 0 else f'payload_ready_{worker-1}', 1 << worker)
    for worker in range(4):
        expect(chains['MM2S', 5][4 + worker], 'grant_token', 0, 32,
               f'grant_ready_{worker}', f'receive_ready_{worker}', 1 << worker)
        expect(chains['S2MM', 4][worker], 'packet_aggregate', worker * 16, 256,
               f'receive_ready_{worker}', f'grant_ready_{worker+1}' if worker < 3 else 'output_ready')
    expected_locks = {memtile: [('stage_empty', 0, 1), ('stage_ready', 1, 0), ('output_ready', 10, 0)] +
                      [(f'grant_ready_{i}', 2+i, 0) for i in range(4)] +
                      [(f'receive_ready_{i}', 6+i, 0) for i in range(4)] +
                      [(f'payload_ready_{i}', 11+i, 0) for i in range(3)]}
    for worker in range(4):
        tile = next(t for t, coord in tiles.items() if coord == (0, worker+2))
        require(buffers.get(f'%worker_planes_{worker}') == (tile, '256xui16') and
                buffers.get(f'%worker_grant_{worker}') == (tile, '32xui16'), 'worker buffer borrows neighbor memory')
        expected_locks[tile] = [(f'{name}_{worker}', i, init) for i, (name, init) in enumerate([
            ('feature_empty', 1), ('feature_ready', 0), ('grant_empty', 1), ('worker_grant_ready', 0), ('send_ready', 0)])]
        local = dma_program(text, tile, 'mem')
        require(set(local) == {('S2MM', 1), ('MM2S', 0)} and len(local['S2MM', 1]) == 2 and len(local['MM2S', 0]) == 1,
                'incorrect worker payload/grant/output DMA schedule')
        expect(local['S2MM', 1][0], f'worker_planes_{worker}', 0, 256, f'feature_empty_{worker}', f'feature_ready_{worker}')
        expect(local['S2MM', 1][1], f'worker_grant_{worker}', 0, 32, f'grant_empty_{worker}', f'worker_grant_ready_{worker}')
        expect(local['MM2S', 0][0], f'worker_planes_{worker}', 0, 256, f'send_ready_{worker}', f'feature_empty_{worker}', 16+worker)
        match = next(m for m in core_matches if m[1] == tile)
        core_body, _ = body_after(text, match.end())
        require(re.findall(r'aie\.use_lock\((%\w+),\s*(\w+),\s*1\)', core_body) == [
            (f'%feature_ready_{worker}', 'AcquireGreaterEqual'), (f'%worker_grant_ready_{worker}', 'AcquireGreaterEqual'),
            (f'%send_ready_{worker}', 'Release'), (f'%grant_empty_{worker}', 'Release')], 'incorrect core grant ownership')
        require(core_body.index('AcquireGreaterEqual') < core_body.index('func.call @packet_stripe_join_tag') <
                core_body.index(f'aie.use_lock(%worker_grant_ready_{worker}'), 'tag must run after input acquisition and before sending')
        for packet, source, source_channel, destination, dest_channel in [
                (1 << worker, memtile, 5, tile, 1), (16+worker, tile, 0, memtile, 4)]:
            pattern = (rf'aie\.packet_flow\({packet}\)\s*\{{\s*'
                       rf'aie\.packet_source<{re.escape(source)}, DMA : {source_channel}>\s*'
                       rf'aie\.packet_dest<{re.escape(destination)}, DMA : {dest_channel}>\s*'
                       r'\}\s*\{keep_pkt_header = false\}')
            require(len(re.findall(pattern, text)) == 1, 'missing payload-only packet route')
    locks = re.findall(r'(%\w+) = aie\.lock\((%\w+),\s*(\d+)\)\s*\{init = (\d+) : i32, sym_name = "([^\"]+)"\}', text)
    require(len(locks) == 34, 'unexpected lock count')
    for tile, expected in expected_locks.items():
        actual = [(name, int(i), int(init)) for value, t, i, init, name in locks if t == tile and value == '%' + name]
        require(sorted(actual) == sorted(expected), 'incorrect initial ownership credits/lock IDs')
    runtime_match = re.search(r'aie\.runtime_sequence\(([^\n]*)\)', text)
    require(runtime_match and re.findall(r'memref<([^>]+)>', runtime_match[1]) == ['25600xui16'] * 2, 'incorrect 25-stripe host ABI')
    runtime, _ = body_after(text, runtime_match.end())
    tasks = list(re.finditer(r'(%\w+) = aiex\.dma_configure_task\((%\w+),\s*(S2MM|MM2S),\s*0\)', runtime))
    require(len(tasks) == 2 and [m[3] for m in tasks] == ['MM2S', 'S2MM'] and
            all(tiles.get(m[2]) == (0, 0) for m in tasks), 'incorrect shim tasks')
    _, output_end = body_after(runtime, tasks[1].end())
    require(re.match(r'\s*\{issue_token = true\}', runtime[output_end:]), 'output task needs completion token')
    require(re.findall(r'aiex\.dma_await_task\((%\w+)\)', runtime) == [tasks[1][1]] and
            re.findall(r'aiex\.dma_free_task\((%\w+)\)', runtime) == [tasks[0][1]] and
            runtime.index('aiex.dma_free_task') > runtime.index('aiex.dma_await_task'), 'output must complete before input freed')
    coverage = []
    for worker, block in enumerate(chains['S2MM', 4]):
        dimensions = [(int(n), int(stride)) for n, stride in re.findall(
            r'<size = (\d+), stride = (\d+)>', block)]
        coverage.extend(scatter_addresses(worker, 16 * worker, dimensions))
    require(sorted(coverage) == list(range(1024)), 'scatter must partition whole stripe exactly')
    for task in tasks:
        body, _ = body_after(runtime, task.end())
        require(len(re.findall(r'aie\.dma_bd\(', body)) == 1 and
                re.search(r'aie\.dma_bd\(%\w+ : memref<25600xui16>, 0, 25600\)', body),
                'runtime must transfer one complete 25-stripe frame')
    return dict(status='PASS', compute_cores=4, memtiles=1,
                stripes_per_frame=25, host_elements=25600, stripe_elements=1024,
                worker_payload_elements=256, scatter_coverage_elements=1024,
                allocations={tile: [dict(begin=b, end=e, name=n, type=t) for b, e, n, t in rows]
                             for tile, rows in allocations.items()},
                static_memtile_bds=14, static_bds_per_worker=3, memtile_locks=14,
                locks_per_worker=5, runtime_output_awaits=1,
                physical_routing_verified=False)


def check_build(root):
    project = root / 'packet_stripe_join.mlir.prj'
    report = check_map((project / 'input_with_addresses.mlir').read_text())
    elfs = sorted(project.glob('main_core_*.elf'))
    require(len(elfs) == 4, 'expected four worker ELFs')
    report['elfs'] = {path.name: check_elf(path.read_bytes()) for path in elfs}
    require(all(elf['text_bytes'] <= 131072 for elf in report['elfs'].values()), 'ELF text exceeds program memory')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True, type=Path)
    print(json.dumps(check_build(parser.parse_args().build_dir), sort_keys=True))
