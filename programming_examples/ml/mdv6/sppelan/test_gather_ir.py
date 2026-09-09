"""CPU parser fixtures: reject corrupted compiled gather footprint/completion."""
import unittest
from sppelan.check_gather_ir import check


def fixture():
    lines = []
    for col in range(4):
        lines += [f'%s{col} = aie.tile({col}, 0)', f'%m{col} = aie.tile({col}, 1)']
        for name, address, size in [('source', 0, 51200), ('stripe', 102400, 8192)]:
            lines.append(f'%{name}{col} = aie.buffer(%m{col}) {{address = {address} : i32, sym_name = "{name}_{col}"}} : memref<{size}xui16>')
        for lock in (0, 1, 2, 3, 4, 16, 17):
            lines.append(f'%l{col}_{lock} = aie.lock(%m{col}, {lock})')
        lines.append(f'aie.memtile_dma(%m{col}) {{')
        for i, (direction, channel) in enumerate([('S2MM', c) for c in range(5)] + [('MM2S', col), ('MM2S', 4)]):
            lines += [f'aie.dma_start({direction}, {channel}, ^bb1, ^bb2)',
                      f'aie.dma_bd(%source{col}) {{bd_id = {i} : i32, next_bd_id = {i} : i32}}']
        lines.append('}')
    lines.append('aie.runtime_sequence(%I: memref<204800xui16>, %O: memref<819200xui16>) {')
    for col in range(4):
        lines.append(f'%i{col} = aiex.dma_configure_task(%s{col}, MM2S, 0) {{ aie.end }}')
        lines.append(f'%o{col} = aiex.dma_configure_task(%s{col}, S2MM, 0) {{ aie.end }} {{issue_token = true}}')
    lines += [f'aiex.dma_await_task(%o{col})' for col in range(4)]
    lines += [f'aiex.dma_free_task(%i{col})' for col in range(4)]
    return '\n'.join(lines + ['}'])


class GatherIRTests(unittest.TestCase):
    def test_expected_footprint(self):
        report = check(fixture())
        self.assertEqual(report['maximum_allocation_end'], 118784)
        self.assertEqual(report['runtime_output_awaits'], 4)
        self.assertFalse(report['physical_routing_verified'])

    def test_rejects_resource_contract_changes(self):
        for old, new in [('address = 102400', 'address = 102398'),
                         ('address = 102400', 'address = 524288'),
                         ('memref<8192xui16>', 'memref<8191xui16>'),
                         ('aie.tile(3, 1)', 'aie.tile(4, 1)'),
                         ('aie.lock(%m0, 17)', 'aie.lock(%m0, 18)'),
                         ('bd_id = 6 : i32', 'bd_id = 48 : i32'),
                         ('aie.dma_start(S2MM, 4', 'aie.dma_start(S2MM, 5')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check(fixture().replace(old, new, 1))

    def test_rejects_compute_core(self):
        with self.assertRaisesRegex(ValueError, 'zero compute'):
            check(fixture() + '\naie.core(%core) {}')

    def test_rejects_missing_or_wrong_completion(self):
        for old, new in [('aiex.dma_await_task(%o3)', ''),
                         ('aiex.dma_await_task(%o3)', 'aiex.dma_await_task(%i3)'),
                         ('{issue_token = true}', '')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check(fixture().replace(old, new, 1))

    def test_rejects_early_free(self):
        text = fixture().replace('aiex.dma_free_task(%i0)', '')
        text = text.replace('aiex.dma_await_task(%o0)', 'aiex.dma_free_task(%i0)\naiex.dma_await_task(%o0)')
        with self.assertRaisesRegex(ValueError, 'before all output'):
            check(text)


if __name__ == '__main__':
    unittest.main()
