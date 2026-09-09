"""One-core finite phase-alias actual-map and lowered-runtime gate.

DMA free operations are compiler bookkeeping, not device completion fences.
This gate checks emitted descriptors/queues; hardware rearm remains a separate
test and runtime stack high-water marks are not measured here.
"""
import argparse
import json
from pathlib import Path
import re
import struct

try:
    from .check_gather_ir import body_after
    from .check_phase_a_ir import check_elf, require
except ImportError:
    from check_gather_ir import body_after
    from check_phase_a_ir import check_elf, require


def check_map(text):
    text = re.sub(r'//[^\n]*', '', text)
    tiles = {v: (int(c), int(r)) for v, c, r in re.findall(r'(%\w+) = aie\.tile\((\d+), (\d+)\)', text)}
    require(len(tiles) == 2 and set(tiles.values()) == {(0, 0), (0, 2)}, 'expected one shim and one core only')
    ct = next(v for v, coord in tiles.items() if coord == (0, 2)); shim = next(v for v, coord in tiles.items() if coord == (0, 0))
    require(re.findall(r'aie\.core\((%\w+)\)', text) == [ct], 'expected exactly one core')
    buffers = re.findall(r'(%\w+) = aie\.buffer\((%\w+)\)\s*\{([^}]*)\}\s*:\s*memref<([^>]+)>', text)
    require(len(buffers) == len(re.findall(r'\baie\.buffer\(', text)) == 1, 'expected exactly one arena')
    value, owner, attrs, shape = buffers[0]
    require(value == '%phase_arena' and owner == ct and shape == '25120xui16' and
            re.search(r'address = 8192 : i32', attrs), 'incorrect arena address/size/owner')
    require(re.findall(r'stack_size = (\d+) : i32', text) == ['8192'], 'expected8192-byte stack')
    require(not re.search(r'\baie\.(?:mem|memtile_dma)\(', text), 'no initialization-only DMA program permitted')
    locks = re.findall(r'(%\w+) = aie\.lock\((%\w+), (\d+)\)\s*\{init = (\d+) : i32, sym_name = "([^\"]+)"\}', text)
    expected = [('frame_empty', 0, 1), ('Aready', 1, 0), ('Asend', 2, 0), ('Bempty', 3, 0), ('Bready', 4, 0), ('Bsend', 5, 0)]
    require(len(locks) == len(re.findall(r'\baie\.lock\(', text)) == 6 and
            [(name, int(i), int(init)) for v, t, i, init, name in locks if t == ct and v == '%' + name] == expected,
            'incorrect six-lock initial ownership')
    flows = re.findall(r'aie\.flow\((%\w+), DMA : (\d+), (%\w+), DMA : (\d+)\)', text)
    require(flows == [(shim, '0', ct, '1'), (ct, '0', shim, '0')] and len(re.findall(r'\baie\.flow\(', text)) == 2,
            'incorrect fixed input/output routes')
    for tile, controller, extra in [(ct, 27, 'ctrl_pkt_flow = true, keep_pkt_header = true'),
                                    (shim, 15, 'keep_pkt_header = true, priority_route = true')]:
        require(re.search(re.escape(tile) + r' = aie.tile\([^\n]+controller_id = #aie.packet_info<pkt_type = 0, pkt_id = ' + str(controller) + '>', text),
                'missing matching completion controller ID')
        require(re.search(r'aie.packet_flow\(' + str(controller) + r'\)\s*\{\s*aie.packet_source<' + re.escape(tile) +
                          r', TileControl : 0>\s*aie.packet_dest<' + re.escape(shim) + r', South : 0>\s*\}\s*\{' + re.escape(extra) + r'\}', text),
                'missing completion-token packet route')
    require(len(re.findall(r'\baie.packet_flow\(', text)) == 2, 'unexpected packet route')
    matches = list(re.finditer(r'aie\.runtime_sequence\(([^\n]*)\)', text))
    require(len(matches) == 1 and re.findall(r'memref<([^>]+)>', matches[0][1]) == ['217600xui16']*2, 'incorrect frame ABI')
    args = re.findall(r'(%\w+): memref<[^>]+>', matches[0][1]); runtime, _ = body_after(text, matches[0].end())
    tasks = list(re.finditer(r'(%\w+) = aiex\.dma_configure_task\((%\w+), (S2MM|MM2S), (\d+)\)', runtime))
    require(len(tasks) == len(re.findall(r'\baiex\.dma_configure_task\(', runtime)) == 6, 'expected four core and two shim tasks')
    ids = []; specs = [('S2MM', '1', 12800, 'frame_empty', 'Aready', 0), ('MM2S', '0', 12800, 'Asend', 'Bempty', 0),
                      ('S2MM', '1', 8192, 'Bempty', 'Bready', 24), ('MM2S', '0', 8192, 'Bsend', 'Bempty', 24)]
    for index, task in enumerate(tasks):
        task_value, tile, direction, channel = task.groups(); body, end = body_after(runtime, task.end())
        attrs = re.match(r'\s*\{([^}]*)\}', runtime[end:]); attrs = attrs[1] if attrs else ''
        repeat = re.search(r'repeat_count = (\d+) : i32', attrs); repeat = int(repeat[1]) if repeat else 0
        require(('issue_token = true' in attrs) == (index in (2, 3, 5)), 'B RX/TX and shim output must issue completion tokens')
        require(len(re.findall(r'\baie\.dma_bd\(', body)) == 1 and not re.search(r'next_bd|aie\.next_bd', body) and
                len(re.findall(r'\baie\.end\b', body)) == 1, 'each task must contain one terminating BD')
        if index < 4:
            d, ch, length, acq, rel, repetitions = specs[index]
            require((tile, direction, channel, repeat) == (ct, d, ch, repetitions), 'incorrect core task queue contract')
            require(re.search(rf'aie\.dma_bd\(%phase_arena : memref<25120xui16>, 0, {length}\)', body), 'incorrect aliased core BD')
            require(re.findall(r'aie\.use_lock\((%\w+), (\w+), (\d+)\)', body) == [
                ('%' + acq, 'AcquireGreaterEqual', '1'), ('%' + rel, 'Release', '1')], 'incorrect core task ownership')
            require(body.index('AcquireGreaterEqual') < body.index('aie.dma_bd') < body.index('Release'), 'lock order must bracket DMA')
            bd = re.findall(r'\bbd_id = (\d+) : i32', body)
            # The addressed map precedes runtime ID allocation. Physical IDs
            # are verified in the retained lowered sidecar and actual binary.
            require(not bd or bd == [str(index)], 'unexpected explicit core BD ID'); ids.append(index)
        else:
            require((tile, direction, channel, repeat) == (shim, 'MM2S' if index == 4 else 'S2MM', '0', 0), 'incorrect shim task')
            require(re.search(rf'aie\.dma_bd\({re.escape(args[index-4])} : memref<217600xui16>, 0, 217600\)', body), 'incorrect full-frame shim transfer')
    require(len(set(ids)) == 4, 'core descriptor IDs alias while active')
    values = [m[1] for m in tasks]
    require(re.findall(r'aiex\.dma_start_task\((%\w+)\)', runtime) == values[:4]+[values[5], values[4]], 'A-before-B and full start coverage required')
    require(runtime.rindex('aiex.dma_start_task') < runtime.index('aiex.dma_await_task'), 'all tasks must start before first completion wait')
    require(re.findall(r'aiex\.dma_await_task\((%\w+)\)', runtime) == [values[2], values[3], values[5]], 'both finite B tasks and shim output must complete')
    require(re.findall(r'aiex\.dma_free_task\((%\w+)\)', runtime) == [values[4]]+values[:2] and
            runtime.index('aiex.dma_free_task') > runtime.rindex('aiex.dma_await_task'), 'input/A tasks freed only after all three completions')
    check_core(text, ct)
    return dict(status='PASS', core_bd_ids=ids, arena_begin=8192, arena_end=58432, arena_bytes=50240,
                stack_bytes=8192, core_tasks=4, locks=6, phase_b_executions=25, frame_bytes=435200,
                task_free_is_device_fence=False)


def check_core(text, ct):
    match = re.search(r'aie\.core\(' + re.escape(ct) + r'\)', text); body, _ = body_after(text, match.end())
    constants = {int(value): name for name, value in re.findall(r'(%\w+) = arith.constant (\d+) : (?:index|i32)', body)}
    require(all(n in constants for n in (0, 1, 25, 8192, 12800, 42405, 23041)), 'missing bounded loop/tag constants')
    locks = re.findall(r'aie\.use_lock\((%\w+), (\w+), (\d+)\)', body)
    require(locks == [('%Aready', 'AcquireGreaterEqual', '1'), ('%Asend', 'Release', '1'),
                      ('%Bready', 'AcquireGreaterEqual', '1'), ('%Bsend', 'Release', '1'),
                      ('%Bempty', 'AcquireGreaterEqual', '1'), ('%frame_empty', 'Release', '1')], 'incorrect core phase/last-drain credits')
    # Check the actual lowered finite loop: zero initialization, <25, unit
    # increment backedge, and rearm on its false branch, not in its body.
    cmp = re.search(r'(%\w+) = arith.cmpi slt, (%\w+), ' + re.escape(constants[25]) + r' : index\s*cf.cond_br \1, \^(\w+), \^(\w+)', body)
    require(cmp is not None, 'missing explicit bounded25 loop')
    loop = re.search(r'\^(\w+)\(' + re.escape(cmp[2]) + r': index\):', body)
    require(loop and f'cf.br ^{loop[1]}({constants[0]} : index)' in body, 'B loop must start at zero')
    loop_body = re.search(r'\^' + cmp[3] + r':(.*?)\^' + cmp[4] + r':', body, re.S)
    require(loop_body is not None, 'unexpected finite loop CFG')
    inc = re.search(r'(%\w+) = arith.addi ' + re.escape(cmp[2]) + ', ' + re.escape(constants[1]) + r' : index', loop_body[1])
    require(inc and f'cf.br ^{loop[1]}({inc[1]} : index)' in loop_body[1], 'B loop must increment by one')
    require('%Bempty' not in loop_body[1] and '%frame_empty' not in loop_body[1] and
            'aie.use_lock(%Bready' in loop_body[1] and 'aie.use_lock(%Bsend' in loop_body[1], 'premature frame rearm')
    calls = re.findall(r'func.call @phase_alias_tag\(%phase_arena, (%\w+), (%\w+)\)', body)
    require(len(calls) == 2 and calls[0] == (constants[12800], constants[42405]) and calls[1][0] == constants[8192], 'incorrect A/B tagging lengths')
    cast = re.search(r'(%\w+) = arith.index_cast ' + re.escape(cmp[2]) + r' : index to i32', loop_body[1])
    require(cast and f'{calls[1][1]} = arith.addi {cast[1]}, {constants[23041]} : i32' in loop_body[1], 'incorrect stripe-dependent B tag')


def check_lowered(text, binary):
    """Check concrete registers, then reconstruct the complete transaction.

    Word layouts follow lib/Targets/AIETargetNPU.cpp append* functions. Exact
    byte comparison ties this retained sidecar to the instructions run by XRT.
    """
    globals_ = {name: [int(v.strip()) & 0xffffffff for v in values.split(',')]
                for name, values in re.findall(r'memref.global "private" constant @(\w+) : memref<\d+xi32> = dense<\[([^\]]+)\]>', text)}
    match = re.search(r'aie\.runtime_sequence\([^\n]+\)', text)
    require(match is not None, 'missing lowered runtime'); runtime, _ = body_after(text, match.end())
    refs = dict(re.findall(r'(%\w+) = memref.get_global @(\w+)', runtime))
    operations = []
    for match in re.finditer(r'aiex\.npu\.(\w+)(?:\((%\w+)\))?\s*\{([^}]+)\}', runtime):
        kind, value, attrs = match.groups()
        attrs = {k: int(v) for k, v in re.findall(r'(\w+) = (-?\d+) : [ui]*32', attrs)}
        if kind == 'blockwrite':
            require(value in refs and refs[value] in globals_, 'unknown blockwrite data')
            operations.append((kind, attrs['address'], globals_[refs[value]]))
        elif kind == 'maskwrite32':
            operations.append((kind, attrs['address'], attrs['value'], attrs['mask']))
        elif kind == 'write32':
            operations.append((kind, attrs['address'], attrs['value']))
        elif kind == 'address_patch':
            operations.append((kind, attrs['addr'], attrs['arg_idx'], attrs['arg_plus']))
        elif kind == 'sync':
            require(attrs.get('column') == 0 and attrs.get('column_num') == attrs.get('row_num') == 1, 'unexpected token sync extent')
            operations.append((kind, attrs['direction'], attrs['row'], attrs['channel']))
        else:
            raise ValueError(f'unexpected lowered opcode {kind}')
    require(len(operations) == len(re.findall(r'aiex\.npu\.', runtime)), 'unrecognized lowered operation')
    expected = []
    # Core BD register words: linear length, no chaining, valid descriptor,
    # acquire>=1/release1 lock pairs0→1,2→3,3→4,5→3 respectively.
    for index, (length, control) in enumerate([(6400,33832928), (6400,33849314), (4096,33857507), (4096,33849317)]):
        addr = 0x21d000 + index*32
        expected += [('blockwrite', addr, [length,0,0,0,0,control]), ('maskwrite32', addr, 0x2000000, 0xfffc000)]
    shim_words = [108800,0,0,0,0xc0000000,0x2000000,0,0x2000000]
    expected += [('blockwrite',0x1d000,shim_words), ('address_patch',0x1d004,0,0),
                 ('blockwrite',0x1d020,shim_words), ('address_patch',0x1d024,1,0),
                 ('write32',0x21de0c,0), ('write32',0x21de14,1),
                 ('maskwrite32',0x21de08,27<<8,0x1f00), ('write32',0x21de0c,0x80180002),
                 ('maskwrite32',0x21de10,27<<8,0x1f00), ('write32',0x21de14,0x80180003),
                 ('maskwrite32',0x1d200,15<<8,0x1f00), ('write32',0x1d204,0x80000001),
                 ('write32',0x1d214,0), ('sync',0,2,1), ('sync',1,2,0), ('sync',0,0,0)]
    require(operations == expected, 'lowered descriptor/address/queue/token sequence mismatch')
    words = []
    for op in operations:
        kind, *values = op
        if kind == 'blockwrite':
            addr, data = values; words += [1,0,addr,(4+len(data))*4]+data
        elif kind == 'write32':
            addr, value = values; words += [0,0,addr,0,value,24]
        elif kind == 'maskwrite32':
            addr, value, mask = values; words += [3,0,addr,0,value,mask,28]
        elif kind == 'address_patch':
            addr, arg, offset = values; words += [129,48,0,0,0,0,addr,0,arg,0,offset,0]
        else:
            direction, row, channel = values; words += [128,16,direction | row<<8,1<<8 | 1<<16 | channel<<24]
    words = [0x06040100,0x108,len(operations),(len(words)+4)*4]+words
    encoded = struct.pack('<'+'I'*len(words), *words)
    require(binary == encoded, 'actual instruction binary differs from checked lowered sidecar')
    return dict(binary_bytes=len(binary), binary_operations=len(operations), core_queue_entries_per_channel=2,
                core_completion_tokens=2, shim_completion_tokens=1, binary_matches_checked_runtime=True)


def check_build(root):
    project = root / 'phase_alias.mlir.prj'
    report = check_map((project / 'input_with_addresses.mlir').read_text())
    report['runtime'] = check_lowered((root / 'phase_alias_runtime.mlir').read_text(), (root / 'phase_alias.bin').read_bytes())
    elfs = list(project.glob('main_core_*.elf'))
    require([p.name for p in elfs] == ['main_core_0_2.elf'], 'expected exactly one worker ELF')
    report['elf'] = check_elf(elfs[0].read_bytes())
    require(report['elf']['text_bytes'] <= 131072, 'ELF exceeds program memory')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True, type=Path)
    print(json.dumps(check_build(parser.parse_args().build_dir), sort_keys=True))
