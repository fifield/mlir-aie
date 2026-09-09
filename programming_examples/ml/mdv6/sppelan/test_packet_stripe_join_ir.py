"""CPU scatter enumeration and mutations of the actual stripe-join contract."""
import unittest

from sppelan.test_packet_aggregate_ir import fixture as packet_fixture
from sppelan.check_packet_stripe_join_ir import check_map, scatter_addresses


def fixture():
    text = packet_fixture().replace('51200', '1024').replace('12800', '256')
    text = text.replace('38400, 256', '768, 256').replace('25600, 256', '512, 256')
    # The aggregate receiver scatters worker lanes into pixel-major OC64.
    for worker in range(4):
        text = text.replace(f'%packet_aggregate : memref<1024xui16>, {worker*256}, 256)',
                            f'%packet_aggregate : memref<1024xui16>, {worker*16}, 256, [<size = 16, stride = 64>, <size = 16, stride = 1>])')
    text = text.replace('@packet_aggregate_tag', '@packet_stripe_join_tag')
    text = text.replace('memref<1024xui16>, %o: memref<1024xui16>', 'memref<25600xui16>, %o: memref<25600xui16>')
    text = text.replace('MM2S, 0) { aie.end }', 'MM2S, 0) { aie.dma_bd(%i : memref<25600xui16>, 0, 25600) aie.end }')
    text = text.replace('S2MM, 0) { aie.end }', 'S2MM, 0) { aie.dma_bd(%o : memref<25600xui16>, 0, 25600) aie.end }')
    return text


class PacketStripeJoinIRTests(unittest.TestCase):
    def test_contract(self):
        report = check_map(fixture())
        self.assertEqual(report['stripes_per_frame'], 25)
        self.assertEqual(report['static_memtile_bds'], 14)

    def test_exact_scatter_partition(self):
        addresses = [scatter_addresses(worker, worker*16, [(16, 64), (16, 1)]) for worker in range(4)]
        self.assertEqual(sorted(sum(addresses, [])), list(range(1024)))
        for worker in range(4):
            for pixel in range(16):
                self.assertEqual(addresses[worker][pixel*16:(pixel+1)*16],
                                 list(range(pixel*64+worker*16, pixel*64+(worker+1)*16)))

    def test_scatter_mutations(self):
        for old, new in [('stride = 64', 'stride = 63'), ('stride = 1>', 'stride = 2>'),
                         ('size = 16, stride = 64', 'size = 15, stride = 64'),
                         ('16, 256, [', '0, 256, ['), ('48, 256, [', '64, 256, [')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_runtime_and_ownership_mutations(self):
        for old, new in [('memref<25600xui16>', 'memref<1024xui16>'), ('0, 25600)', '0, 1024)'),
                         ('%stage_empty, Release', '%stage_ready, Release'), ('768, 256', '0, 256'),
                         ('keep_pkt_header = false', 'keep_pkt_header = true'), ('aiex.dma_await_task(%out)', '')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))

    def test_actual_allocation_mutations(self):
        for old, new in [('address = 4096', 'address = 4094'), ('address = 29696', 'address = 65530'),
                         ('256xui16', '256xbf16'), ('aie.dma_start(S2MM, 5', 'aie.dma_start(S2MM, 3')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_map(fixture().replace(old, new, 1))


if __name__ == '__main__':
    unittest.main()
