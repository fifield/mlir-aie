"""CPU-only mutations of the packet proof's compiled ownership contract."""
import unittest

from sppelan.check_packet_aggregate_ir import check_map


def fixture():
    lines = [f'%t{r} = aie.tile(0, {r})' for r in range(6)]
    def buf(name, tile, address, size):
        lines.append(f'%{name} = aie.buffer(%t{tile}) {{address = {address} : i32, sym_name = "{name}"}} : memref<{size}xui16>')
    def lock(name, tile, index, init=0):
        lines.append(f'%{name} = aie.lock(%t{tile}, {index}) {{init = {init} : i32, sym_name = "{name}"}}')
    def use(name, action):
        return f'aie.use_lock(%{name}, {action}, 1)\n'
    def program(tile, kind, chains):
        lines.append(f'aie.{kind}(%t{tile}) {{')
        counter = 0
        for direction, channel, entries in chains:
            first = counter
            lines.append(f'aie.dma_start({direction}, {channel}, ^b{first}, ^end)')
            for index, (name, size, offset, length, acquire, release, packet) in enumerate(entries):
                lines.append(f'^b{counter}:')
                if acquire:
                    lines.append(use(acquire, 'AcquireGreaterEqual'))
                attrs = f'bd_id = {counter} : i32'
                if packet is not None:
                    attrs += f', packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>'
                lines.append(f'aie.dma_bd(%{name} : memref<{size}xui16>, {offset}, {length}) {{{attrs}}}')
                if release:
                    lines.append(use(release, 'Release'))
                counter += 1
                lines.append(f'aie.next_bd ^b{first if index == len(entries)-1 else counter}')
        lines.append('^end: aie.end\n}')
    buf('packet_source', 1, 0, 51200); buf('packet_aggregate', 1, 102400, 51200); buf('grant_token', 1, 204800, 32)
    lock('stage_empty', 1, 0, 1); lock('stage_ready', 1, 1); lock('output_ready', 1, 10)
    for i in range(3):
        lock(f'payload_ready_{i}', 1, 11+i)
    for i in range(4):
        lock(f'grant_ready_{i}', 1, 2+i); lock(f'receive_ready_{i}', 1, 6+i)
        buf(f'worker_planes_{i}', i+2, 4096, 12800); buf(f'worker_grant_{i}', i+2, 29696, 32)
        for index, (name, init) in enumerate([('feature_empty', 1), ('feature_ready', 0), ('grant_empty', 1), ('worker_grant_ready', 0), ('send_ready', 0)]):
            lock(f'{name}_{i}', i+2, index, init)
        lines.append(f'aie.core(%t{i+2}) {{' + use(f'feature_ready_{i}', 'AcquireGreaterEqual') +
                     'func.call @packet_aggregate_tag()\n' + use(f'worker_grant_ready_{i}', 'AcquireGreaterEqual') +
                     use(f'send_ready_{i}', 'Release') + use(f'grant_empty_{i}', 'Release') + '} {stack_size = 4096 : i32}')
        program(i+2, 'mem', [('S2MM', 1, [
            (f'worker_planes_{i}', 12800, 0, 12800, f'feature_empty_{i}', f'feature_ready_{i}', None),
            (f'worker_grant_{i}', 32, 0, 32, f'grant_empty_{i}', f'worker_grant_ready_{i}', None)]),
            ('MM2S', 0, [(f'worker_planes_{i}', 12800, 0, 12800, f'send_ready_{i}', f'feature_empty_{i}', 16+i)])])
        for packet, src, sc, dst, dc in [(1 << i, 1, 5, i+2, 1), (16+i, i+2, 0, 1, 4)]:
            lines.append(f'aie.packet_flow({packet}) {{ aie.packet_source<%t{src}, DMA : {sc}> aie.packet_dest<%t{dst}, DMA : {dc}> }} {{keep_pkt_header = false}}')
    sends = [('packet_source', 51200, i*12800, 12800, 'stage_ready' if i == 3 else f'payload_ready_{i}',
              'grant_ready_0' if i == 0 else f'payload_ready_{i-1}', 1 << i) for i in reversed(range(4))]
    sends += [('grant_token', 32, 0, 32, f'grant_ready_{i}', f'receive_ready_{i}', 1 << i) for i in range(4)]
    receives = [('packet_aggregate', 51200, i*12800, 12800, f'receive_ready_{i}', f'grant_ready_{i+1}' if i < 3 else 'output_ready', None) for i in range(4)]
    program(1, 'memtile_dma', [('S2MM', 5, [('packet_source', 51200, 0, 51200, 'stage_empty', 'stage_ready', None)]),
                             ('MM2S', 5, sends), ('S2MM', 4, receives),
                             ('MM2S', 1, [('packet_aggregate', 51200, 0, 51200, 'output_ready', 'stage_empty', None)])])
    lines.append('aie.runtime_sequence(%i: memref<51200xui16>, %o: memref<51200xui16>) {\n'
                 '%in = aiex.dma_configure_task(%t0, MM2S, 0) { aie.end }\n'
                 '%out = aiex.dma_configure_task(%t0, S2MM, 0) { aie.end } {issue_token = true}\n'
                 'aiex.dma_await_task(%out)\naiex.dma_free_task(%in)\n}')
    return '\n'.join(lines)


class PacketAggregateIRTests(unittest.TestCase):
    def test_contract(self):
        self.assertEqual(check_map(fixture())['static_memtile_bds'], 14)

    def test_allocation_and_channel_mutations(self):
        for old, new in [('address = 4096', 'address = 4094'), ('address = 29696', 'address = 65530'),
                         ('51200xui16', '51200xi7'), ('stack_size = 4096', 'stack_size = 0'),
                         ('aie.dma_start(S2MM, 5', 'aie.dma_start(S2MM, 3')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_packet_and_payload_mutations(self):
        for old, new in [('keep_pkt_header = false', 'keep_pkt_header = true'), ('pkt_id = 16>', 'pkt_id = 17>'),
                         ('38400, 12800', '0, 12800'), ('25600, 12800', '25600, 12802')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_lock_and_completion_mutations(self):
        for old, new in [('init = 1 : i32', 'init = 0 : i32'),
                         ('%grant_ready_1, Release', '%grant_ready_0, Release'),
                         ('%stage_empty, Release', '%stage_ready, Release'),
                         ('aiex.dma_await_task(%out)', ''), ('issue_token = true', 'issue_token = false')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_neighbor_borrow(self):
        text = fixture().replace('aie.buffer(%t2)', 'aie.buffer(%SWAP)').replace(
            'aie.buffer(%t3)', 'aie.buffer(%t2)').replace('aie.buffer(%SWAP)', 'aie.buffer(%t3)')
        with self.assertRaisesRegex(ValueError, 'neighbor'):
            check_map(text)


if __name__ == '__main__':
    unittest.main()
