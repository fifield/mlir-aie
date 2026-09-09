"""Synthetic-map mutations and ordered native-source/gather address tests."""
import re
import unittest

from sppelan.test_packet_aggregate_ir import fixture as packet_fixture
from sppelan.check_packet_gather_ir import check_map, check_order, addresses, SOURCE_DIMS, SCATTER_DIMS
from sppelan.check_gather_ir import body_after


def fixture():
    columns = []
    for col in range(4):
        text = packet_fixture().split('aie.runtime_sequence')[0]
        names = set(re.findall(r'sym_name = "([^\"]+)"', text))
        for name in sorted(names, key=len, reverse=True):
            match = re.match(r'(.+)_(\d+)$', name)
            new = f'{match[1]}_{col}_{match[2]}' if match else f'{name}_{col}'
            text = re.sub(r'(?<![\w.])' + re.escape(name) + r'(?!\w)', new, text)
        text = text.replace('aie.tile(0,', f'aie.tile({col},').replace('@packet_aggregate_tag', '@packet_gather_tag')
        text = re.sub(r'%t(\d+)', lambda m: f'%c{col}t{m[1]}', text)
        text = text.replace('address = 204800', 'address = 221184')
        text += f'%stripe_{col} = aie.buffer(%c{col}t1) {{address = 204800 : i32, sym_name = "stripe_{col}"}} : memref<8192xui16>\n'
        for turn in range(5):
            text += f'%stripe_turn_{col}_{turn} = aie.lock(%c{col}t1, {14+turn}) {{init = {int(turn == 0)} : i32, sym_name = "stripe_turn_{col}_{turn}"}}\n'
        start = text.index(f'aie.memtile_dma(%c{col}t1)'); body, end = body_after(text, start)
        body = body.replace('aie.dma_start(MM2S, 1,', f'aie.dma_start(MM2S, {col},')
        body = body.replace(f'%packet_aggregate_{col} : memref<51200xui16>, 0, 51200)',
                            f'%packet_aggregate_{col} : memref<51200xui16>, 0, 51200, [<size = 25, stride = 128>, <size = 4, stride = 12800>, <size = 4, stride = 3200>, <size = 128, stride = 1>])')
        extra = ''
        for i in range(5):
            direction, channel = ('S2MM', i) if i < 4 else ('MM2S', 4)
            dims = ', [<size = 4, stride = 8>, <size = 4, stride = 128>, <size = 16, stride = 512>, <size = 8, stride = 1>]' if i < 4 else ''
            extra += (f'aie.dma_start({direction}, {channel}, ^g{i}, ^end)\n^g{i}:\n'
                      f'aie.use_lock(%stripe_turn_{col}_{i}, AcquireGreaterEqual, 1)\n'
                      f'aie.dma_bd(%stripe_{col} : memref<8192xui16>, {32*i if i<4 else 0}, {2048 if i<4 else 8192}{dims}) {{bd_id = {14+i} : i32}}\n'
                      f'aie.use_lock(%stripe_turn_{col}_{(i+1)%5}, Release, 1)\naie.next_bd ^g{i}\n')
        body = body.replace('^end: aie.end', extra + '^end: aie.end')
        opening = text.index('{', start)
        text = text[:opening+1] + body + text[end-1:]
        columns.append(text)
    lines = columns
    for col in range(4):
        lines += [f'aie.flow(%c{col}t0, DMA : 0, %c{col}t1, DMA : 5)', f'aie.flow(%c{col}t1, DMA : 4, %c{col}t0, DMA : 0)']
        lines += [f'aie.flow(%c{col}t1, DMA : {col}, %c{dest}t1, DMA : {col})' for dest in range(4)]
        lines += [f'aie.packet_flow(15) {{ aie.packet_source<%c{col}t0, TileControl : 0> aie.packet_dest<%c{col}t0, South : 0> }} {{keep_pkt_header = true, priority_route = true}}']
    lines += ['aie.runtime_sequence(%I: memref<204800xui16>, %O: memref<819200xui16>) {']
    for output in (False, True):
        for col in range(4):
            task, direction, arg, size, length = (f'o{col}', 'S2MM', 'O', 819200, 204800) if output else (f'i{col}', 'MM2S', 'I', 204800, 51200)
            token = ' {issue_token = true}' if output else ''
            lines += [f'%{task} = aiex.dma_configure_task(%c{col}t0, {direction}, 0) {{\naie.dma_bd(%{arg} : memref<{size}xui16>, {col*length}, {length})\naie.end\n}}{token}']
    lines += [f'aiex.dma_start_task(%{kind}{col})' for kind in ('o', 'i') for col in range(4)]
    lines += [f'aiex.dma_await_task(%o{col})' for col in range(4)]
    lines += [f'aiex.dma_free_task(%i{col})' for col in range(4)]
    return '\n'.join(lines + ['}'])


class PacketGatherIRTests(unittest.TestCase):
    def test_actual_contract_fixture(self):
        self.assertEqual(check_map(fixture())['static_bds_per_memtile'], 19)

    def test_ordered_addresses(self):
        check_order()
        source = addresses(0, SOURCE_DIMS)
        self.assertEqual(source[:128], list(range(128)))
        self.assertEqual(source[128:256], list(range(3200, 3328)))
        self.assertEqual(addresses(96, SCATTER_DIMS)[:16], [96,97,98,99,100,101,102,103,608,609,610,611,612,613,614,615])

    def test_runtime_offsets_lengths_starts_completion(self):
        for old, new in [('614400, 204800)', '0, 204800)'), ('memref<819200xui16>', 'memref<204800xui16>'),
                         ('aiex.dma_start_task(%o3)', ''), ('aiex.dma_await_task(%o3)', ''),
                         ('aiex.dma_await_task(%o0)', 'aiex.dma_free_task(%i0)\naiex.dma_await_task(%o0)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_retention_and_stripe_credit_mutations(self):
        for old, new in [('%stage_empty_0, Release', '%stage_ready_0, Release'),
                         ('%stripe_turn_0_0, Release', '%stripe_turn_0_1, Release'),
                         ('%grant_ready_0_1, Release', '%grant_ready_0_0, Release')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_dimensions_and_resources(self):
        for old, new in [('size = 25, stride = 128', 'size = 25, stride = 127'), ('size = 4, stride = 8', 'size = 4, stride = 9'),
                         ('address = 4096', 'address = 4094'), ('address = 221184', 'address = 524280'), ('8192xui16', '8192xi7')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_exact_route_sets(self):
        for extra in ['aie.flow(%c0t0, DMA : 0, %c0t1, DMA : 5)',
                      'aie.packet_flow(15) { aie.packet_source<%c0t0, TileControl : 0> aie.packet_dest<%c0t0, South : 0> } {keep_pkt_header = true, priority_route = true}']:
            with self.assertRaises(ValueError):
                check_map(fixture() + '\n' + extra)
        with self.assertRaises(ValueError):
            check_map(fixture().replace('keep_pkt_header = false', 'keep_pkt_header = true', 1))


if __name__ == '__main__':
    unittest.main()
