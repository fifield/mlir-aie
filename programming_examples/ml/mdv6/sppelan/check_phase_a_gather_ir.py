"""Independent actual-resource/runtime gate for numerical phase A to gather.

Checks this fixed experimental workload, not arbitrary MLIR or physical routes.
In full-build mode the checked lowered runtime is reconstructed byte-for-byte
and compared with the actual XRT instruction binary. Compile-only mode checks
components without claiming routing, ELF, binary, or hardware validation.
No hardware or AIE Python is required by this checker.
"""
import argparse
import itertools
import json
from pathlib import Path
import re
import struct

try:
    from .check_gather_ir import body_after
    from .check_phase_a_ir import check_elf, memref_size, require
    from .check_packet_aggregate_ir import dma_program
except ImportError:
    from check_gather_ir import body_after
    from check_phase_a_ir import check_elf, memref_size, require
    from check_packet_aggregate_ir import dma_program


def attrs(text):
    return {k: int(v) for k, v in re.findall(r'(\w+) = (-?\d+) : [ui]*32', text)}


def runtime_operations(text):
    globals_ = {name: [int(v.strip()) & 0xffffffff for v in values.split(',')]
                for name, values in re.findall(r'memref.global "private" constant @(\w+) : memref<\d+xi32> = dense<\[([^\]]+)\]>', text)}
    starts = list(re.finditer(r'aie\.runtime_sequence\([^\n]+\)', text))
    require(len(starts) == 1, 'expected exactly one lowered runtime')
    runtime, _ = body_after(text, starts[0].end())
    refs = dict(re.findall(r'(%\w+) = memref.get_global @(\w+)', runtime))
    operations = []
    for match in re.finditer(r'aiex\.npu\.(\w+)(?:\((%\w+)\))?\s*\{([^}]+)\}', runtime):
        kind, value, fields = match.groups(); a = attrs(fields)
        if kind in ('write32','maskwrite32','blockwrite') and 'column' in a and 'row' in a:
            a['address']=(a['column']<<25)|(a['row']<<20)|(a['address']&0xfffff)
        if kind == 'blockwrite':
            require(value in refs and refs[value] in globals_, 'unknown blockwrite payload')
            operations.append((kind, a['address'], globals_[refs[value]]))
        elif kind == 'maskwrite32':
            operations.append((kind, a['address'], a['value'], a['mask']))
        elif kind == 'write32':
            operations.append((kind, a['address'], a['value']))
        elif kind == 'address_patch':
            operations.append((kind, a['addr'], a['arg_idx'], a['arg_plus']))
        elif kind == 'sync':
            require(a.get('column_num') == a.get('row_num') == 1, 'unexpected completion-token extent')
            operations.append((kind, a['direction'], a['row'], a['channel'], a['column']))
        else:
            raise ValueError(f'unexpected lowered opcode {kind}')
    require(len(operations) == len(re.findall(r'aiex\.npu\.', runtime)), 'unrecognized lowered operation')
    return operations


def check_binary(text, binary):
    operations = runtime_operations(text); words = []
    for kind, *values in operations:
        if kind == 'blockwrite':
            addr, data = values; words += [1, 0, addr, (4+len(data))*4] + data
        elif kind == 'write32':
            addr, value = values; words += [0, 0, addr, 0, value, 24]
        elif kind == 'maskwrite32':
            addr, value, mask = values; words += [3, 0, addr, 0, value, mask, 28]
        elif kind == 'address_patch':
            addr, arg, offset = values; words += [129, 48, 0, 0, 0, 0, addr, 0, arg, 0, offset, 0]
        else:
            direction, row, channel, col = values
            words += [128, 16, direction | row<<8 | col<<16, 1<<8 | 1<<16 | channel<<24]
    words = [0x06040100, 0x108, len(operations), (len(words)+4)*4] + words
    require(binary == struct.pack('<'+'I'*len(words), *words), 'actual instruction binary differs from checked runtime')
    return operations


def addresses(offset, dimensions):
    return [offset + sum(i * stride for i, (_, stride) in zip(index, dimensions))
            for index in itertools.product(*(range(n) for n, _ in dimensions))]


def check_iteration_order():
    activation = [s*4096 + lane for s in range(25) for lane in range(4096)]
    require(activation == list(range(102400)), 'activation iteration must cover the complete input exactly')
    source = addresses(0, [(25,128),(4,12832),(4,3200),(128,1)])
    expected = [w*12832 + level*3200 + s*128 + p*8 + lane
                for s in range(25) for w in range(4) for level in range(4) for p in range(16) for lane in range(8)]
    require(source == expected, 'gather source order differs from feature stream')
    metadata = addresses(12800, [(4,12832),(32,1)])
    require(metadata == [w*12832 + 12800 + i for w in range(4) for i in range(32)] and
            sorted(source+metadata) == list(range(51328)), 'features and metadata must partition aggregate without overlap')


def check_bd_bank_union(static, runtime):
    """Check combined (BD ID, DMA channel) ownership, not allocator-local fit."""
    all_ids = [bd for bd, channel in static + runtime]
    require(len(all_ids) == len(set(all_ids)), 'static/runtime BD collision')
    require(all(0 <= bd < 48 and 0 <= channel < 6 and (bd >= 24) == bool(channel % 2)
                for bd, channel in static + runtime), 'memtile BD in inaccessible channel bank')


def check_queue_word(word, bd, repeat, token):
    require(0 <= bd < 48 and 0 <= repeat < 256, 'invalid memtile task queue contract')
    require(word == (bd | repeat << 16 | int(token) << 31), 'truncated/wrong memtile six-bit BD queue word')


def check_allocations(text):
    tiles = {v: (int(c), int(r)) for v,c,r in re.findall(r'(%\w+) = aie\.tile\((\d+), (\d+)\)', text)}
    require(len(tiles) == 24 and set(tiles.values()) == set(itertools.product(range(4),range(6))), 'expected four columns and sixteen workers')
    cores = list(re.finditer(r'aie\.core\((%\w+)\)', text))
    require(len(cores) == 16 and {tiles[m[1]] for m in cores} == set(itertools.product(range(4),range(2,6))), 'incorrect compute core placement')
    allocations = {tile: [] for tile in tiles}
    for core in cores:
        _, end = body_after(text, core.end())
        fields = re.match(r'\s*\{([^}]*)\}', text[end:])
        require(fields and attrs(fields[1]).get('stack_size') == 8192, 'expected explicit8192-byte worker stacks')
        allocations[core[1]].append((0,8192,'stack',None))
    records = re.findall(r'(%\w+) = aie\.buffer\((%\w+)\)\s*\{([^}]*)\}\s*:\s*memref<([^>]+)>', text)
    require(len(records) == 84 and len(re.findall(r'\baie\.buffer\(',text)) == 84, 'unexpected buffer count or unknown syntax')
    buffers = {}
    for value,tile,fields,shape in records:
        address = attrs(fields).get('address'); name = re.search(r'sym_name = "([^\"]+)"',fields)
        require(tile in tiles and tiles[tile][1] > 0 and address is not None and name and value not in buffers, 'unknown buffer allocation')
        begin=address; end=begin+memref_size(shape)
        require(0<=begin<end<=(524288 if tiles[tile][1]==1 else 65536), 'buffer exceeds actual tile memory')
        allocations[tile].append((begin,end,name[1],shape)); buffers[value]=(tile,shape,begin)
    for tile,rows in allocations.items():
        rows.sort()
        require(all(a[1]<=b[0] for a,b in zip(rows,rows[1:])), 'stack/buffer overlap in actual map')
        expected = [] if tiles[tile][1]==0 else (['102400xui16','8448xui16','51328xui16','8192xui16','32xui16'] if tiles[tile][1]==1
                    else ['12832xui16','4096xui16','2112xui16','32xui16'])
        require(sorted(t for _,_,_,t in rows if t) == sorted(expected), 'extra or missing resident buffer')
    return tiles,buffers,allocations,cores


def records(body):
    found = list(re.finditer(r'aie\.dma_bd\((%\w+) : memref<([^>]+)>, (\d+), (\d+)(?:, (\[[^\]]+\]))?\)\s*\{([^}]*)\}',body))
    require(len(found)==len(re.findall(r'\baie\.dma_bd\(',body)) and found,'unrecognized explicit BD')
    locks = re.findall(r'aie\.use_lock\((%\w+), (\w+), (\d+)\)',body)
    require(len(locks)==2*len(found),'every BD needs paired acquire/release locks')
    result=[]
    for index,match in enumerate(found):
        value,shape,offset,length,dimensions,fields=match.groups()
        acq,rel=locks[index*2:index*2+2]
        require(acq[1:]==('AcquireGreaterEqual','1') and rel[1:]==('Release','1'),'incorrect BD lock action')
        before=body[:match.start()]; after=body[match.end():]
        require(before.rfind(f'aie.use_lock({acq[0]}, AcquireGreaterEqual, 1)')>=0 and
                re.match(r'\s*aie.use_lock\('+re.escape(rel[0])+r', Release, 1\)',after),'BD locks must bracket transfer')
        a=attrs(fields); require('bd_id' in a,'missing explicit BD ID')
        packet=re.findall(r'packet = #aie.packet_info<pkt_type = 0, pkt_id = (\d+)>',fields)
        require(len(packet)<=1,'multiple packet headers')
        dims=[(int(n),int(s)) for n,s in re.findall(r'<size = (\d+), stride = (\d+)>',dimensions or '')]
        result.append((value,shape,int(offset),int(length),a['bd_id'],dims,None if not packet else int(packet[0]),acq[0],rel[0]))
    return result


def check_terminating_chain(body, count):
    body=re.sub(r'//[^\n]*','',body)
    labels=re.findall(r'^\s*\^(\w+):',body,re.M)
    targets=re.findall(r'aie.next_bd \^(\w+)',body)
    require(len(labels)==count-1 and len(set(labels))==len(labels) and targets==labels,
            'runtime BD chain must advance to each next block exactly once')
    require(len(re.findall(r'\baie.end\b',body))==1 and
            re.search(r'aie.end\s*$',body),'runtime BD chain must terminate in final block')


def check_projection_loop(body, suffix):
    body=re.sub(r'//[^\n]*','',body)
    constants={int(n):v for v,n in re.findall(r'(%\w+) = arith.constant (\d+) : index',body)}
    require(all(n in constants for n in (0,1,25)),'missing exact projection loop constants')
    match=re.search(r'\^(\w+)\((%\w+): index\):\s*(%\w+) = arith.cmpi slt, \2, '+re.escape(constants[25])+r' : index\s*cf.cond_br \3, \^(\w+), \^(\w+)',body)
    require(match is not None,'missing bounded projection loop header')
    header,iv,condition,work,done=match.groups()
    require(re.search(r'cf.br \^'+header+r'\('+re.escape(constants[0])+r' : index\)',body),'projection loop must start at zero')
    work_match=re.search(r'\^'+work+r':(.*?)\^'+done+r':',body,re.S)
    require(work_match is not None,'projection body/exit block order mismatch')
    work_body=work_match[1]
    cast=re.search(r'(%\w+) = arith.index_cast '+re.escape(iv)+r' : index to i32',work_body)
    require(cast is not None,'projection index must be the loop induction variable')
    call='func.call @phase_a_gather_project('+', '.join('%worker_'+n+'_'+suffix for n in ('input','weights','packet'))+', '+cast[1]+')'
    require(call in work_body and 'func.call @phase_a_gather_pool' not in work_body,'incorrect projection arguments or early pool')
    increment=re.search(r'(%\w+) = arith.addi '+re.escape(iv)+', '+re.escape(constants[1])+r' : index\s*cf.br \^'+header+r'\(\1 : index\)',work_body)
    require(increment is not None,'projection induction increment/backedge must be exactly one')
    acquire=f'aie.use_lock(%worker_input_ready_{suffix}, AcquireGreaterEqual, 1)'
    release=f'aie.use_lock(%worker_input_empty_{suffix}, Release, 1)'
    require(acquire in work_body and release in work_body and work_body.index(acquire)<work_body.index(call)<work_body.index(release),
            'input ownership must bracket each projection')
    require(body.index('func.call @phase_a_gather_pool')>work_match.end()-len(done)-2,'pool must be in loop exit')


def expected_tasks(tile_at):
    tasks=[]
    def record(name,size,offset,length,ident,acq,rel,dims=(),packet=None):
        return ('%'+name,f'{size}xui16',offset,length,ident,list(dims),packet,'%'+acq,'%'+rel)
    for col in range(4):
        for worker in range(4):
            suffix=f'{col}_{worker}';ct=tile_at[col,worker+2]
            for name,size,ident,acq,rel,repeat,token,direction,ch,packet in [
                ('worker_weights',2112,0,'weights_empty','weights_ready',0,False,'S2MM',1,None),
                ('worker_input',4096,1,'input_empty','input_ready',24,False,'S2MM',1,None),
                ('worker_grant',32,2,'grant_empty','grant_ready',0,True,'S2MM',1,None),
                ('worker_packet',12832,3,'planes_ready','planes_empty',0,True,'MM2S',0,16+worker)]:
                tasks.append((ct,direction,ch,repeat,token,[record(name+'_'+suffix,size,0,size,ident,'worker_'+acq+'_'+suffix,'worker_'+rel+'_'+suffix,packet=packet)]))
    for col in range(4):
        mt=tile_at[col,1]
        tasks += [(mt,'S2MM',5,0,True,[record(f'phase_input_{col}',102400,0,102400,28,f'frame_empty_{col}',f'I_ready_{col}'),
                                      record(f'phase_weights_{col}',8448,0,8448,29,f'I_ready_{col}',f'stage_ready_{col}')])]
        tasks += [(mt,'MM2S',5,0,False,[record(f'phase_weights_{col}',8448,w*2112,2112,30+w,
                                             f'stage_ready_{col}' if w==0 else f'weight_turn_{w}_{col}',
                                             f'activation_turn_0_{col}' if w==3 else f'weight_turn_{w+1}_{col}',packet=1<<w) for w in range(4)])]
        tasks += [(mt,'MM2S',5,24,False,[record(f'phase_input_{col}',102400,0,4096,34+w,
                                              f'activation_turn_{w}_{col}',f'activation_turn_{(w+1)%4}_{col}',
                                              [(25,4096),(1,0),(1,0),(4096,1)],1<<w) for w in range(4)])]
        tasks += [(mt,'MM2S',5,0,True,[record(f'grant_token_{col}',32,0,32,38+w,
                                            f'activation_turn_0_{col}' if w==0 else f'grant_ready_{w}_{col}',f'receive_ready_{w}_{col}',packet=1<<w) for w in range(4)])]
        tasks += [(mt,'S2MM',4,0,True,[record(f'phase_aggregate_{col}',51328,w*12832,12832,4+w,
                                            f'receive_ready_{w}_{col}',f'output_ready_{col}' if w==3 else f'grant_ready_{w+1}_{col}') for w in range(4)])]
        tasks += [(mt,'MM2S',4,24,False,[record(f'stripe_{col}',8192,0,8192,8,f'stripe_turn_4_{col}',f'stripe_turn_0_{col}')]),
                  (mt,'MM2S',4,0,True,[record(f'phase_aggregate_{col}',51328,12800,128,9,f'metadata_ready_{col}',f'frame_empty_{col}',[(1,0),(1,0),(4,12832),(32,1)])])]
    return tasks


def check_map(text):
    text=re.sub(r'//[^\n]*','',text)
    tiles,buffers,allocations,cores=check_allocations(text); tile_at={coord:value for value,coord in tiles.items()}
    expected=expected_tasks(tile_at)
    matches=list(re.finditer(r'aie\.runtime_sequence\(([^\n]*)\)',text))
    require(len(matches)==1 and re.findall(r'memref<([^>]+)>',matches[0][1])==['102400xui16','33792xui16','819200xui16','512xui16'],'incorrect four-BO ABI')
    runtime,_=body_after(text,matches[0].end())
    tasks=list(re.finditer(r'(%\w+) = aiex\.dma_configure_task\((%\w+), (S2MM|MM2S), (\d+)\)',runtime))
    require(len(tasks)==108 and len(re.findall(r'\baiex\.dma_configure_task\(',runtime))==108,'incorrect runtime task count')
    actual=[]
    for task in tasks[:92]:
        value,tile,direction,ch=task.groups();body,end=body_after(runtime,task.end())
        fields=re.match(r'\s*\{([^}]*)\}',runtime[end:]);fields=fields[1] if fields else ''
        recs=records(body)
        check_terminating_chain(body,len(recs))
        actual.append((tile,direction,int(ch),attrs(fields).get('repeat_count',0),'issue_token = true' in fields,recs))
    require(actual==expected,'runtime numerical/distribution/metadata ownership mismatch')
    for tile,direction,ch,repeat,issue,recs in actual:
        for value,shape,offset,length,ident,dims,packet,acq,rel in recs:
            require(value in buffers and buffers[value][:2]==(tile,shape),'runtime DMA borrows nonlocal allocation')
    _check_static_and_locks(text,tiles,tile_at,buffers,cores,actual)
    _check_shim_and_completion(runtime,tasks,actual,tiles,tile_at,matches[0][1])
    check_iteration_order()
    return dict(status='PASS',compute_cores=16,memtiles=4,runtime_bds=160,static_bds=20,
                worker_bytes_including_stack=46336,memtile_buffer_bytes=340800,
                allocations={tile:[dict(begin=b,end=e,name=n,type=t) for b,e,n,t in rows] for tile,rows in allocations.items()}),actual,buffers


def _check_static_and_locks(text,tiles,tile_at,buffers,cores,actual):
    expected_locks={}; expected_flows=[]; packets=[]; controllers=[]
    for col in range(4):
        mt=tile_at[col,1];shim=tile_at[col,0]
        names=['frame_empty','I_ready','stage_ready']+[f'weight_turn_{w}' for w in range(1,4)]
        names += [f'activation_turn_{w}' for w in range(4)]+[f'receive_ready_{w}' for w in range(4)]
        names += [f'grant_ready_{w}' for w in range(1,4)]+['output_ready','metadata_ready']+[f'stripe_turn_{i}' for i in range(5)]
        expected_locks[mt]=[(f'{name}_{col}',i,int(i in (0,19))) for i,name in enumerate(names)]
        chains=dma_program(text,mt,'memtile_dma')
        require(set(chains)=={('MM2S',col)}|{('S2MM',i) for i in range(4)} and all(len(v)==1 for v in chains.values()),'incorrect static gather channel/BD budget')
        static=[]
        for (direction,ch),blocks in chains.items():
            rec=records(blocks[0])[0];static.append((rec[4],ch))
            if direction=='MM2S':
                expected=(f'%phase_aggregate_{col}','51328xui16',0,51200,2 if col%2==0 else 26,[(25,128),(4,12832),(4,3200),(128,1)],None,f'%output_ready_{col}',f'%metadata_ready_{col}')
            else:
                expected=(f'%stripe_{col}','8192xui16',ch*32,2048,[0,24,1,25][ch],[(4,8),(4,128),(16,512),(8,1)],None,f'%stripe_turn_{ch}_{col}',f'%stripe_turn_{ch+1}_{col}')
            require(rec==expected,'incorrect static gather stream/metadata-retention contract')
            require(buffers[rec[0]][:2]==(mt,rec[1]),'static DMA borrows neighboring buffer')
        runtime=[(r[4],ch) for tile,direction,ch,repeat,issue,recs in actual if tile==mt for r in recs]
        require(len(runtime)==20,'incorrect finite memtile BD budget');check_bd_bank_union(static,runtime)
        expected_flows += [(shim,0,mt,5),(mt,4,shim,0)]+[(mt,col,tile_at[d,1],col) for d in range(4)]
        controllers += [(mt,shim,26),(shim,shim,15)]
        for worker in range(4):
            ct=tile_at[col,worker+2];suffix=f'{col}_{worker}'
            cnames=['weights_empty','weights_ready','input_empty','input_ready','grant_empty','grant_ready','planes_empty','planes_ready']
            expected_locks[ct]=[(f'worker_{name}_{suffix}',i,int(i in (0,2,4,6))) for i,name in enumerate(cnames)]
            packets += [(1<<worker,mt,5,ct,1),(16+worker,ct,0,mt,4)]
            controllers += [(ct,shim,[27,29,30,31][worker])]
            core=next(m for m in cores if m[1]==ct);body,_=body_after(text,core.end())
            check_projection_loop(body,suffix)
            expected_core=[(f'%worker_{name}_{suffix}',action,'1') for name,action in [
                ('weights_ready','AcquireGreaterEqual'),('planes_empty','AcquireGreaterEqual'),('input_ready','AcquireGreaterEqual'),
                ('input_empty','Release'),('weights_empty','Release'),('grant_ready','AcquireGreaterEqual'),('planes_ready','Release'),('grant_empty','Release')]]
            require(re.findall(r'aie\.use_lock\((%\w+), (\w+), (\d+)\)',body)==expected_core,'incorrect weight/input/packet ownership')
            constant=re.search(r'(%\w+) = arith.constant 25 : index',body)
            require(constant and re.search(r'arith.cmpi slt, %\w+, '+re.escape(constant[1])+r' : index',body),'missing bounded25 projection loop')
            require(len(re.findall(r'func.call @phase_a_gather_project\(',body))==1 and len(re.findall(r'func.call @phase_a_gather_pool\(',body))==1 and
                    body.index('func.call @phase_a_gather_project') < body.index('func.call @phase_a_gather_pool') < body.index(f'aie.use_lock(%worker_weights_empty_{suffix}'),'pool must follow projection before weight release')
    require(not re.search(r'\baie.mem\(',text),'core DMA must be finite runtime tasks')
    locks=re.findall(r'(%\w+) = aie\.lock\((%\w+), (\d+)\)\s*\{init = (\d+) : i32, sym_name = "([^\"]+)"\}',text)
    require(len(locks)==224 and len(re.findall(r'\baie.lock\(',text))==224,'incorrect lock budget')
    for tile,expected in expected_locks.items():
        require([(name,int(i),int(init)) for v,t,i,init,name in locks if t==tile and v=='%'+name]==expected,'incorrect initial lock state/IDs')
    flows=[(s,int(sc),d,int(dc)) for s,sc,d,dc in re.findall(r'aie.flow\((%\w+), DMA : (\d+), (%\w+), DMA : (\d+)\)',text)]
    require(sorted(flows)==sorted(expected_flows) and len(re.findall(r'\baie.flow\(',text))==24,'incorrect circuit route set')
    data=re.findall(r'aie.packet_flow\((\d+)\)\s*\{\s*aie.packet_source<(%\w+), DMA : (\d+)>\s*aie.packet_dest<(%\w+), DMA : (\d+)>\s*\}\s*\{keep_pkt_header = false\}',text)
    require(sorted((int(i),s,int(sc),d,int(dc)) for i,s,sc,d,dc in data)==sorted(packets),'incorrect packet data route/header policy')
    for tile,shim,ident in controllers:
        require(re.search(re.escape(tile)+r' = aie.tile\([^\n]+controller_id = #aie.packet_info<pkt_type = 0, pkt_id = '+str(ident)+'>',text),'missing controller ID')
        extra='keep_pkt_header = true, priority_route = true'
        require(re.search(r'aie.packet_flow\('+str(ident)+r'\)\s*\{\s*aie.packet_source<'+re.escape(tile)+r', TileControl : 0>\s*aie.packet_dest<'+re.escape(shim)+r', South : 0>\s*\}\s*\{'+re.escape(extra)+r'\}',text),'missing completion-token route')
    require(len(re.findall(r'\baie.packet_flow\(',text))==56,'unexpected packet/control route count')


def _check_shim_and_completion(runtime,tasks,actual,tiles,tile_at,signature):
    arguments=re.findall(r'(%\w+): memref<[^>]+>',signature); inputs=[];outputs=[]
    for i,task in enumerate(tasks[92:]):
        col=i//4;role=i%4;body,end=body_after(runtime,task.end());value,tile,direction,ch=task.groups()
        require((tile,direction,ch)==(tile_at[col,0],'MM2S' if role<2 else 'S2MM','0'),'wrong shim queue role')
        fields=re.match(r'\s*\{([^}]*)\}',runtime[end:]);fields=fields[1] if fields else ''
        require(('issue_token = true' in fields)==(role>=2) and attrs(fields).get('repeat_count',0)==0,'incorrect shim token/repetition')
        size=[102400,33792,819200,512][role];length=[102400,8448,204800,128][role];offset=0 if role==0 else col*length
        require(re.search(r'aie.dma_bd\('+re.escape(arguments[role])+rf' : memref<{size}xui16>, {offset}, {length}\)',body) and
                len(re.findall(r'\baie.dma_bd\(',body))==1 and attrs(body).get('burst_length')==0,'incorrect shim address/size')
        (inputs if role<2 else outputs).append(value)
    core_values=[m[1] for m in tasks[:64]]
    core_done=[tasks[col*16+worker*4+role][1] for worker in range(4) for col in range(4) for role in (2,3)]
    ingress=[tasks[64+col*7][1] for col in range(4)]
    mt_done=[tasks[64+col*7+role][1] for col in range(4) for role in (3,4,6)]
    free=[m[1] for m,t in zip(tasks[:92],actual) if not t[4]]
    require(re.findall(r'aiex.dma_start_task\((%\w+)\)',runtime)==core_values+outputs+inputs,'incorrect core/shim task start coverage')
    require(re.findall(r'aiex.dma_await_task\((%\w+)\)',runtime)==ingress+core_done+mt_done+outputs,'all finite core/MT tokens and eight host outputs must be awaited in causal order')
    require(re.findall(r'aiex.dma_free_task\((%\w+)\)',runtime)==inputs+free,'incorrect explicit free coverage')
    require(runtime.rindex('aiex.dma_start_task') < runtime.index('aiex.dma_await_task') and runtime.rindex('aiex.dma_await_task') < runtime.index('aiex.dma_free_task'),'premature wait/free')


def descriptor_operations(text, actual, buffers):
    """Independently encode this fixed workload's BD words and address patches.

    All lengths are 32-bit words; buffer offsets in the addressed IR are bf16
    elements. Register field layouts follow AIE2 core/MT descriptors. This is
    deliberately not a general MLIR lowering implementation.
    """
    tiles=check_allocations(text)[0]
    lock_ids={value:int(ident) for value,ident in re.findall(r'(%\w+) = aie.lock\(%\w+, (\d+)\)',text)}
    result=[]
    for tile,direction,channel,repeat,token,recs in actual:
        col,row=tiles[tile];base=(col<<25)|(row<<20)
        for index,rec in enumerate(recs):
            name,shape,offset,length,ident,dims,packet,acq,rel=rec
            following=recs[index+1][4] if index+1<len(recs) else 0
            chained=int(index+1<len(recs));a=lock_ids[acq];r=lock_ids[rel]
            address=base+(0xa0000 if row==1 else 0x1d000)+32*ident
            if row==1:
                d0=d1=stride=iteration=0
                if 34<=ident<=37:
                    iteration=(24<<17)|2047
                elif ident==9:
                    d0=16;d1=4;stride=6415
                words=[length//2 | (0 if packet is None else (1<<31)|(packet<<23)),
                       following<<20 | chained<<19 | offset*2,
                       d0<<17,(d1<<17)|stride,0,0,iteration,
                       (1<<31)|(1<<24)|((r+64)<<16)|(1<<15)|(127<<8)|(a+64)]
                patch=('maskwrite32',address+4,(0x80000+buffers[name][2]+offset*2)//4,0x7ffff)
            else:
                words=[length//2,0 if packet is None else (1<<30)|(packet<<19),0,0,0,
                       following<<27 | chained<<26 | (1<<25)|(1<<18)|(r<<13)|(1<<12)|(127<<5)|a]
                patch=('maskwrite32',address,((buffers[name][2]+offset*2)//4)<<14,0xfffc000)
            result.extend([('blockwrite',address,words),patch])
    for col in range(4):
        for role,length in enumerate([102400,8448,204800,128]):
            address=(col<<25)+0x1d000+32*role
            result.extend([('blockwrite',address,[length//2,0 if role==0 else col*length*2,0,0,0xc0000000,0x2000000,0,0x2000000]),
                           ('address_patch',address+4,role,0 if role==0 else col*length*2)])
    return result


def check_lowered(text, lowered):
    """Check actual descriptor words, full six-bit queues and completion fences."""
    report,actual,buffers=check_map(text)
    tiles=check_allocations(text)[0]
    operations=runtime_operations(lowered)
    prefix=descriptor_operations(text,actual,buffers)
    require(operations[:len(prefix)]==prefix,'lowered descriptor length/address/iteration/lock mismatch')
    expected=[]
    for tile,direction,channel,repeat,token,recs in actual:
        col,row=tiles[tile];base=(col<<25)|(row<<20)
        if row==1:
            control=base+0xa0600+channel*8+(0x30 if direction=='MM2S' else 0)
            controller=26
        else:
            control=base+0x1de00+channel*8+(0x10 if direction=='MM2S' else 0)
            controller=[27,29,30,31][row-2]
        if token: expected.append(('maskwrite32',control,controller<<8,0x1f00))
        expected.append(('write32',control+4,recs[0][4]|repeat<<16|int(token)<<31))
    for roles in ((2,3),(0,1)):
        for col in range(4):
            for role in roles:
                if role>=2: expected.append(('maskwrite32',(col<<25)+0x1d200,15<<8,0x1f00))
                expected.append(('write32',(col<<25)+(0x1d204 if role>=2 else 0x1d214),role|(int(role>=2)<<31)))
    expected += [('sync',0,1,5,col) for col in range(4)]
    expected += [('sync',direction,row,1 if direction==0 else 0,col)
                 for row in range(2,6) for col in range(4) for direction in (0,1)]
    expected += [('sync',direction,1,channel,col) for col in range(4) for direction,channel in ((1,5),(0,4),(1,4))]
    expected += [('sync',0,0,0,col) for col in range(4) for _ in range(2)]
    require(operations[len(prefix):]==expected,'incorrect lowered queue IDs/order/controller or completion fences')
    report.update(status='COMPONENT_PASS',hardware_approved=False,
                  runtime_operations=len(operations),runtime_descriptor_writes=160)
    return report


def check_build(build_dir):
    root=Path(build_dir);stem='spp_phase_a_gather';project=root/(stem+'.mlir.prj')
    required=[root/(stem+suffix) for suffix in ('.xclbin','.bin','_tasks.mlir','_runtime.mlir')]
    required.append(project/'input_with_addresses.mlir')
    require(all(path.is_file() and path.stat().st_size for path in required),'missing compiled artifact; compile-only checks do not authorize hardware')
    text=required[-1].read_text();lowered=(root/(stem+'_runtime.mlir')).read_text()
    report=check_lowered(text,lowered)
    check_binary(lowered,(root/(stem+'.bin')).read_bytes())
    elfs=sorted(project.glob('main_core_*.elf'))
    require({p.name for p in elfs}=={f'main_core_{c}_{r}.elf' for c in range(4) for r in range(2,6)},'missing or extra worker ELF')
    report['elfs']={p.name:check_elf(p.read_bytes()) for p in elfs}
    report['instruction_bytes']=(root/(stem+'.bin')).stat().st_size
    report['status']='COMPILED_GATE_PASS'
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',required=True,type=Path)
    parser.add_argument('--compile-only',action='store_true',help='component checks only; NOT hardware approval')
    args=parser.parse_args()
    if args.compile_only:
        stem='spp_phase_a_gather'
        report=check_lowered((args.build_dir/(stem+'.mlir.prj/input_with_addresses.mlir')).read_text(),
                             (args.build_dir/(stem+'_runtime.mlir')).read_text())
        report['unchecked']=['physical routing','worker ELF sizes','actual instruction binary','hardware execution']
    else:
        report=check_build(args.build_dir)
    print(json.dumps(report,sort_keys=True))
