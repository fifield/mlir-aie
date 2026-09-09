"""Mutation tests for focused phase-A allocation/ELF inspection."""
import struct
import unittest
from sppelan.check_phase_a_ir import check_map, check_elf, memref_size


def fixture(pool=False):
    types = [('shard_planes_buff_0', '12800xui16'), ('shard_metadata_buff_0', '32xui16'),
             ('shard_input_cons_buff_0', '3200xui16' if pool else '4096xui16')]
    if not pool:
        types.append(('shard_weights_cons_buff_0', '2112xui16'))
    types.append(('_anonymous0', '3xi32'))
    lines = ['%c = aie.tile(0, 2)']; address = 8192
    for name, ty in types:
        lines.append(f'%{name} = aie.buffer(%c) {{address = {address} : i32, sym_name = "{name}"}} : memref<{ty}>')
        address += memref_size(ty)
    lines.append('aie.core(%c) {} {stack_size = 8192 : i32}')
    abi = ['3200xui16'] if pool else ['102400xui16', '2112xui16']
    lines.append('aie.runtime_sequence(' + ', '.join(f'%arg{i}: memref<{t}>' for i,t in enumerate(abi + ['12800xui16','32xui16'])) + ') {')
    names = ['shard_input'] + ([] if pool else ['shard_weights']) + ['shard_planes', 'shard_metadata']
    for i, name in enumerate(names):
        token = ' {issue_token = true}' if name in ('shard_planes', 'shard_metadata') else ''
        lines.append(f'%t{i} = aiex.dma_configure_task_for @{name}_shim_alloc {{\n aie.end\n}}{token}')
    lines += [f'aiex.dma_await_task(%t{i})' for i in range(len(names)-2,len(names))]
    lines += [f'aiex.dma_free_task(%t{i})' for i in range(len(names)-2)]
    return '\n'.join(lines + ['}'])


class PhaseAIRTests(unittest.TestCase):
    def test_both_maps(self):
        for pool in (False, True):
            self.assertEqual(check_map(fixture(pool), pool)['output_awaits'], 2)

    def test_rejects_map_corruption(self):
        for old, new in [('address = 8192', 'address = 8190'),
                         ('address = 8192', 'address = 65536'),
                         ('12800xui16', '12800xbf16'),
                         ('3xi32', '100xui16'),
                         ('memref<2112xui16>', 'memref<2112xi16>'),
                         ('102400xui16', '102401xui16'),
                         ('aiex.dma_await_task(%t3)', ''),
                         ('{issue_token = true}', '')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_rejects_neighbor_borrow(self):
        text = fixture().replace('aie.buffer(%c)', 'aie.buffer(%other)', 1)
        with self.assertRaisesRegex(ValueError, 'neighbor'):
            check_map('%other = aie.tile(1, 2)\n' + text)

    def test_primitive_types_fail_closed(self):
        self.assertEqual(memref_size('4x8xbf16'), 64)
        for value in ('4xindex', '?xui16', '4xi7', '0xi32'):
            with self.assertRaises(ValueError):
                memref_size(value)

    def test_elf_sections(self):
        data = bytearray(132); data[:6] = b'\x7fELF\x01\x01'
        struct.pack_into('<I', data, 32, 52); struct.pack_into('<HH', data, 46, 40, 2)
        struct.pack_into('<10I', data, 92, 0, 1, 6, 0, 0, 128, 0, 0, 4, 0)
        self.assertEqual(check_elf(data)['text_bytes'], 128)
        struct.pack_into('<10I', data, 52, 0, 8, 2, 0, 0, 4, 0, 0, 4, 0)
        with self.assertRaisesRegex(ValueError, 'hidden data'):
            check_elf(data)


if __name__ == '__main__':
    unittest.main()
