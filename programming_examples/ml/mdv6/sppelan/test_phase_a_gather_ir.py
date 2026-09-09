"""Ordered iteration, actual binary, and shared BD-bank ownership mutations."""
import unittest
import tempfile

from sppelan.check_phase_a_gather_ir import (addresses, check_iteration_order, check_binary,
                                            check_bd_bank_union, check_queue_word,
                                            check_allocations, check_projection_loop,
                                            check_terminating_chain, runtime_operations, check_build)
from sppelan.test_phase_alias_ir import runtime_fixture


def allocation_fixture():
    lines=[]
    for col in range(4):
        for row in range(6):
            tile=f'%tile_{col}_{row}'
            lines.append(f'{tile} = aie.tile({col}, {row})')
            if row==0: continue
            if row==1:
                shapes=[102400,51328,8448,8192,32];address=0
            else:
                lines.append(f'aie.core({tile}) {{ aie.end }} {{stack_size = 8192 : i32}}')
                shapes=[12832,4096,2112,32];address=8192
            for index,size in enumerate(shapes):
                name=f'buffer_{col}_{row}_{index}'
                lines.append(f'%{name} = aie.buffer({tile}) {{address = {address} : i32, sym_name = "{name}"}} : memref<{size}xui16>')
                address+=size*2
    return '\n'.join(lines)


def loop_fixture():
    return '''%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c25 = arith.constant 25 : index
cf.br ^head(%c0 : index)
^head(%iv: index):
%cond = arith.cmpi slt, %iv, %c25 : index
cf.cond_br %cond, ^work, ^done
^work:
aie.use_lock(%worker_input_ready_0_0, AcquireGreaterEqual, 1)
%cast = arith.index_cast %iv : index to i32
func.call @phase_a_gather_project(%worker_input_0_0, %worker_weights_0_0, %worker_packet_0_0, %cast)
aie.use_lock(%worker_input_empty_0_0, Release, 1)
%next = arith.addi %iv, %c1 : index
cf.br ^head(%next : index)
^done:
func.call @phase_a_gather_pool(%worker_packet_0_0)
'''


class PhaseAGatherIRTests(unittest.TestCase):
    def test_feature_metadata_partition_and_order(self):
        check_iteration_order()
        features = addresses(0, [(25,128),(4,12832),(4,3200),(128,1)])
        self.assertEqual(features[:128], list(range(128)))
        self.assertEqual(features[512:640], list(range(12832,12960)))
        self.assertNotIn(12800, features)
        self.assertEqual(addresses(12800,[(4,12832),(32,1)])[-1],51327)

    def test_disjoint_channel_bank_union(self):
        for col in range(4):
            static=[(0,0),(24,1),(1,2),(25,3),(26 if col%2 else 2,col)]
            runtime=[(i,5) for i in range(28,42)]+[(i,4) for i in range(4,10)]
            check_bd_bank_union(static,runtime)
            with self.assertRaisesRegex(ValueError,'collision'):
                check_bd_bank_union(static,runtime+[(0,0)])
        with self.assertRaisesRegex(ValueError,'bank'):
            check_bd_bank_union([],[(12,5)])

    def test_full_six_bit_queue_id_and_repeat(self):
        for bd in (28,30,34,38):
            check_queue_word(0x80180000|bd,bd,24,True)
            with self.assertRaises(ValueError):
                check_queue_word(0x80180000|(bd&15),bd,24,True)
            with self.assertRaises(ValueError):
                check_queue_word(0x80000000|bd,bd,24,True)

    def test_exact_runtime_binary_tie(self):
        text,binary=runtime_fixture()
        self.assertEqual(len(check_binary(text,binary)),24)
        for index in (0,16,len(binary)-1):
            bad=bytearray(binary);bad[index]^=1
            with self.assertRaises(ValueError):
                check_binary(text,bad)

    def test_actual_allocation_mutations(self):
        text=allocation_fixture()
        tiles,buffers,allocations,cores=check_allocations(text)
        self.assertEqual(len(buffers),84)
        self.assertEqual(max(x[1] for x in allocations['%tile_0_2']),46336)
        self.assertEqual(max(x[1] for x in allocations['%tile_0_1']),340800)
        for old,new in [('address = 8192 :','address = 4096 :'),
                        ('memref<12832xui16>','memref<12833xui16>'),
                        ('memref<32xui16>','memref<32xunknown>'),
                        ('stack_size = 8192','stack_size = 4096'),
                        ('address = 204800','address = 520000')]:
            with self.subTest(old=old),self.assertRaises(ValueError):
                check_allocations(text.replace(old,new,1))

    def test_projection_loop_bounds_and_ownership(self):
        text=loop_fixture();check_projection_loop(text,'0_0')
        for old,new in [('^head(%c0','^head(%c1'),('addi %iv, %c1','addi %iv, %c25'),
                        ('index_cast %iv','index_cast %c0'),('^head(%next','^done(%next'),
                        ('%worker_input_empty_0_0','%wrong_lock')]:
            with self.subTest(old=old),self.assertRaises(ValueError):
                check_projection_loop(text.replace(old,new),'0_0')

    def test_terminating_chain_rejects_cycles_and_skips(self):
        text='aie.next_bd ^b1\n^b1:\naie.next_bd ^b2\n^b2:\naie.end'
        check_terminating_chain(text,3)
        for old,new in [('next_bd ^b2','next_bd ^b1'),('next_bd ^b1','next_bd ^b2'),('aie.end','aie.next_bd ^b1')]:
            with self.assertRaises(ValueError):check_terminating_chain(text.replace(old,new),3)

    def test_direct_register_tile_coordinates(self):
        text='aie.runtime_sequence(%arg0: memref<1xui16>) { aiex.npu.write32 {address = 656940 : ui32, column = 3 : i32, row = 1 : i32, value = 28 : ui32} }'
        self.assertEqual(runtime_operations(text),[('write32',(3<<25)|(1<<20)|656940,28)])

    def test_missing_compiled_build_is_not_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'missing compiled artifact'):
                check_build(directory)


if __name__=='__main__':
    unittest.main()
