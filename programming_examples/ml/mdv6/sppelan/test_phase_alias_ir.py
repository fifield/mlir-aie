"""Focused finite-task, phase-loop, arena and concrete-binary mutations."""
import struct
import unittest
from sppelan.check_phase_alias_ir import check_map, check_lowered


def map_fixture():
    lines = ['%s = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}',
             '%c = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}',
             '%phase_arena = aie.buffer(%c) {address = 8192 : i32, sym_name = "phase_arena"} : memref<25120xui16>']
    for i, name in enumerate(('frame_empty','Aready','Asend','Bempty','Bready','Bsend')):
        lines += [f'%{name} = aie.lock(%c, {i}) {{init = {int(i==0)} : i32, sym_name = "{name}"}}']
    lines += ['aie.flow(%s, DMA : 0, %c, DMA : 1)', 'aie.flow(%c, DMA : 0, %s, DMA : 0)',
              'aie.packet_flow(27) { aie.packet_source<%c, TileControl : 0> aie.packet_dest<%s, South : 0> } {ctrl_pkt_flow = true, keep_pkt_header = true}',
              'aie.packet_flow(15) { aie.packet_source<%s, TileControl : 0> aie.packet_dest<%s, South : 0> } {keep_pkt_header = true, priority_route = true}', 'aie.core(%c) {']
    for value in (0,1,25,8192,12800,42405,23041):
        lines += [f'%c{value} = arith.constant {value} : index']
    lines += ['aie.use_lock(%Aready, AcquireGreaterEqual, 1)',
              'func.call @phase_alias_tag(%phase_arena, %c12800, %c42405)',
              'aie.use_lock(%Asend, Release, 1)', 'cf.br ^loop(%c0 : index)', '^loop(%i: index):',
              '%cmp = arith.cmpi slt, %i, %c25 : index', 'cf.cond_br %cmp, ^body, ^done', '^body:',
              'aie.use_lock(%Bready, AcquireGreaterEqual, 1)', '%cast = arith.index_cast %i : index to i32',
              '%mask = arith.addi %cast, %c23041 : i32', 'func.call @phase_alias_tag(%phase_arena, %c8192, %mask)',
              'aie.use_lock(%Bsend, Release, 1)', '%inc = arith.addi %i, %c1 : index', 'cf.br ^loop(%inc : index)', '^done:',
              'aie.use_lock(%Bempty, AcquireGreaterEqual, 1)', 'aie.use_lock(%frame_empty, Release, 1)', '} {stack_size = 8192 : i32}',
              'aie.runtime_sequence(%I: memref<217600xui16>, %O: memref<217600xui16>) {']
    for i, (direction, channel, length, acq, rel) in enumerate([
            ('S2MM',1,12800,'frame_empty','Aready'), ('MM2S',0,12800,'Asend','Bempty'),
            ('S2MM',1,8192,'Bempty','Bready'), ('MM2S',0,8192,'Bsend','Bempty')]):
        attrs = ' {issue_token = true, repeat_count = 24 : i32}' if i>=2 else ''
        lines += [f'%t{i} = aiex.dma_configure_task(%c, {direction}, {channel}) {{',
                  f'aie.use_lock(%{acq}, AcquireGreaterEqual, 1)', f'aie.dma_bd(%phase_arena : memref<25120xui16>, 0, {length})',
                  f'aie.use_lock(%{rel}, Release, 1)', 'aie.end', '}'+attrs]
    for i, direction, arg in [(4,'MM2S','I'),(5,'S2MM','O')]:
        lines += [f'%t{i} = aiex.dma_configure_task(%s, {direction}, 0) {{ aie.dma_bd(%{arg} : memref<217600xui16>, 0, 217600) aie.end }}'+(' {issue_token = true}' if i==5 else '')]
    lines += [f'aiex.dma_start_task(%t{i})' for i in (0,1,2,3,5,4)]
    lines += [f'aiex.dma_await_task(%t{i})' for i in (2,3,5)]
    lines += [f'aiex.dma_free_task(%t{i})' for i in (4,0,1)]
    return '\n'.join(lines+['}'])


def runtime_fixture():
    # Explicit fixed register contract, independent of MLIR parsing.
    ops=[]
    for i,(length,control) in enumerate([(6400,33832928),(6400,33849314),(4096,33857507),(4096,33849317)]):
        ops += [('blockwrite',0x21d000+i*32,[length,0,0,0,0,control]), ('maskwrite32',0x21d000+i*32,0x2000000,0xfffc000)]
    shim=[108800,0,0,0,0xc0000000,0x2000000,0,0x2000000]
    ops += [('blockwrite',0x1d000,shim),('address_patch',0x1d004,0,0),('blockwrite',0x1d020,shim),('address_patch',0x1d024,1,0),
            ('write32',0x21de0c,0),('write32',0x21de14,1),('maskwrite32',0x21de08,6912,7936),('write32',0x21de0c,0x80180002),
            ('maskwrite32',0x21de10,6912,7936),('write32',0x21de14,0x80180003),('maskwrite32',0x1d200,3840,7936),
            ('write32',0x1d204,0x80000001),('write32',0x1d214,0),('sync',0,2,1),('sync',1,2,0),('sync',0,0,0)]
    globals_=[]; lines=[]; words=[]
    for i,(kind,*v) in enumerate(ops):
        if kind=='blockwrite':
            addr,data=v
            globals_.append(f'memref.global "private" constant @g{i} : memref<{len(data)}xi32> = dense<[{", ".join(map(str,data))}]>')
            lines += [f'%v{i} = memref.get_global @g{i}', f'aiex.npu.blockwrite(%v{i}) {{address = {addr} : ui32}}']
            words += [1,0,addr,(4+len(data))*4]+data
        elif kind=='maskwrite32':
            addr,value,mask=v; lines += [f'aiex.npu.maskwrite32 {{address = {addr} : ui32, value = {value} : ui32, mask = {mask} : ui32}}']; words += [3,0,addr,0,value,mask,28]
        elif kind=='write32':
            addr,value=v; lines += [f'aiex.npu.write32 {{address = {addr} : ui32, value = {value} : ui32}}']; words += [0,0,addr,0,value,24]
        elif kind=='address_patch':
            addr,arg,offset=v; lines += [f'aiex.npu.address_patch {{addr = {addr} : ui32, arg_idx = {arg} : i32, arg_plus = {offset} : i32}}']; words += [129,48,0,0,0,0,addr,0,arg,0,offset,0]
        else:
            d,r,c=v; lines += [f'aiex.npu.sync {{direction = {d} : i32, row = {r} : i32, channel = {c} : i32, column = 0 : i32, column_num = 1 : i32, row_num = 1 : i32}}']; words += [128,16,d|(r<<8),(1<<8)|(1<<16)|(c<<24)]
    words=[0x06040100,0x108,len(ops),(len(words)+4)*4]+words
    return '\n'.join(globals_+['aie.runtime_sequence(%I: memref<217600xui16>, %O: memref<217600xui16>) {']+lines+['}']), struct.pack('<'+'I'*len(words),*words)


class PhaseAliasIRTests(unittest.TestCase):
    def test_map_and_binary_contract(self):
        self.assertEqual(check_map(map_fixture())['arena_end'],58432)
        self.assertEqual(check_lowered(*runtime_fixture())['binary_bytes'],756)

    def test_map_mutations(self):
        for old,new in [('address = 8192','address = 8190'),('25120xui16','25121xui16'),('repeat_count = 24','repeat_count = 25'),
                        ('aiex.dma_await_task(%t2)',''),('aiex.dma_start_task(%t0)',''),('arith.constant 25','arith.constant 26'),
                        ('%Bempty, AcquireGreaterEqual','%Bready, AcquireGreaterEqual'),('keep_pkt_header = true','keep_pkt_header = false')]:
            with self.subTest(old=old),self.assertRaises(ValueError): check_map(map_fixture().replace(old,new,1))

    def test_finite_descriptor_mutation(self):
        text=map_fixture().replace('0, 12800)','0, 12800) {next_bd_id = 0 : i32}',1)
        with self.assertRaises(ValueError): check_map(text)

    def test_lowered_mutations(self):
        text,binary=runtime_fixture()
        for old,new in [('2149056514','2147483650'),('33857507','33857508'),('mask = 268419072','mask = 268419073'),
                        ('row = 2 : i32','row = 0 : i32'),('value = 6912','value = 3840')]:
            with self.subTest(old=old),self.assertRaises(ValueError): check_lowered(text.replace(old,new,1),binary)

    def test_binary_mutation_and_truncation(self):
        text,binary=runtime_fixture()
        for index in (0,16,32,len(binary)-1):
            wrong=bytearray(binary); wrong[index]^=1
            with self.assertRaises(ValueError): check_lowered(text,wrong)
        with self.assertRaises(ValueError): check_lowered(text,binary[:-4])


if __name__=='__main__': unittest.main()
