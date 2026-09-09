"""Numerical 16-shard phase A, resident packet aggregation, and gather.

I[400,256], W[4,4,2,1056], O[4,25,16,512], M[4,4,32], all uint16.
Static gather BDs and finite runtime tasks have explicit disjoint ID banks.
"""
import sys
import numpy as np
from aie.dialects.aie import (AIEDevice, DMAChannelDir, WireBundle, LockAction,
    device, tile, buffer, lock, flow, packetflow, memtile_dma, core,
    dma_start, dma_bd, use_lock, next_bd, EndOp, external_func)
from aie.dialects.aiex import (runtime_sequence, dma_configure_task, bds,
    shim_dma_bd, dma_start_task, dma_await_task, dma_free_task,
    NpuWrite32Op, NpuMaskWrite32Op)
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.dialects import arith
from aie.extras import types as T
from aie.ir import Attribute, BoolAttr


def start_memtile_task_6bit(col, direction, channel, bd_id, repeat=0, token=False):
    """Local NPU2 workaround: generic task-start lowering masks BD IDs to4bits.

    xaie2pgbl_params.h MEM_TILE_MODULE_DMA_*_START_QUEUE_MASK=0x80FF003F.
    Preserve six physical BD-ID bits; configure/await/free remain normal tasks.
    """
    if not (0 <= col < 4 and 0 <= channel < 6 and 0 <= bd_id < 48):
        raise ValueError("invalid memtile queue target")
    if (bd_id >= 24) != bool(channel & 1):
        raise ValueError("memtile BD bank/channel parity mismatch")
    if not 0 <= repeat <= 255:
        raise ValueError("invalid queue repeat count")
    control = 0xA0600 + channel * 8 + (0x30 if direction == DMAChannelDir.MM2S else 0)
    if token:
        NpuMaskWrite32Op(control, 26 << 8, 0x1F00, column=col, row=1)
    word = bd_id | (repeat << 16) | (0x80000000 if token else 0)
    NpuWrite32Op(control + 4, word, column=col, row=1)


def generate():
    ty = lambda n: np.ndarray[(n,), np.dtype[np.uint16]]
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def body():
            shims = [tile(c, 0) for c in range(4)]
            mts = [tile(c, 1) for c in range(4)]
            project = external_func("phase_a_gather_project",
                                    [ty(4096), ty(2112), ty(12832), np.int32],
                                    link_with="phase_a_gather.o")
            pool = external_func("phase_a_gather_pool", [ty(12832)], link_with="phase_a_gather.o")
            mt_state, core_state = [], []

            def controller(t, shim, ident):
                t.operation.attributes["controller_id"] = Attribute.parse(
                    f"#aie.packet_info<pkt_type = 0, pkt_id = {ident}>")
                p = packetflow(ident,t,WireBundle.TileControl,0,
                               {"dest":shim,"port":WireBundle.South,"channel":0},
                               keep_pkt_header=True)
                p.operation.attributes["priority_route"] = BoolAttr.get(True)

            for col in range(4):
                flow(shims[col], WireBundle.DMA, 0, mts[col], WireBundle.DMA, 5)
                flow(mts[col], WireBundle.DMA, 4, shims[col], WireBundle.DMA, 0)
                for dest in range(4):
                    flow(mts[col], WireBundle.DMA, col, mts[dest], WireBundle.DMA, col)

            for col, mt in enumerate(mts):
                controller(mt, shims[col], 26)
                inp = buffer(mt, ty(102400), name=f"phase_input_{col}")
                weights = buffer(mt, ty(8448), name=f"phase_weights_{col}")
                aggregate = buffer(mt, ty(51328), name=f"phase_aggregate_{col}")
                stripe = buffer(mt, ty(8192), name=f"stripe_{col}")
                token = buffer(mt, ty(32), name=f"grant_token_{col}", initial_value=np.zeros(32,np.uint16))
                names = ["frame_empty", "I_ready", "stage_ready"]
                names += [f"weight_turn_{r}" for r in range(1,4)]
                names += [f"activation_turn_{r}" for r in range(4)]
                names += [f"receive_ready_{r}" for r in range(4)]
                names += [f"grant_ready_{r}" for r in range(1,4)]
                names += ["output_ready", "metadata_ready"]
                names += [f"stripe_turn_{r}" for r in range(5)]
                lk = [lock(mt, lock_id=i, init=int(i in (0,19)), sym_name=f"{n}_{col}")
                      for i,n in enumerate(names)]
                mt_state.append((inp,weights,aggregate,stripe,token,lk))

                for row in range(4):
                    ct = tile(col, row+2)
                    controller(ct, shims[col], [27,29,30,31][row])
                    packetflow(1 << row, mt, WireBundle.DMA, 5,
                               {"dest":ct,"port":WireBundle.DMA,"channel":1}, keep_pkt_header=False)
                    packetflow(16+row, ct, WireBundle.DMA, 0,
                               {"dest":mt,"port":WireBundle.DMA,"channel":4}, keep_pkt_header=False)
                    ci = buffer(ct, ty(4096), name=f"worker_input_{col}_{row}")
                    cw = buffer(ct, ty(2112), name=f"worker_weights_{col}_{row}")
                    co = buffer(ct, ty(12832), name=f"worker_packet_{col}_{row}")
                    cg = buffer(ct, ty(32), name=f"worker_grant_{col}_{row}")
                    cnames = ["weights_empty","weights_ready","input_empty","input_ready",
                              "grant_empty","grant_ready","planes_empty","planes_ready"]
                    cl = [lock(ct, lock_id=i, init=int(i in (0,2,4,6)),
                               sym_name=f"worker_{n}_{col}_{row}") for i,n in enumerate(cnames)]
                    core_state.append((ct,ci,cw,co,cg,cl,row))

                    @core(ct, stack_size=8192)
                    def worker():
                        for _ in range_(sys.maxsize):
                            use_lock(cl[1], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(cl[6], LockAction.AcquireGreaterEqual, value=1)
                            for s in range_(25):
                                use_lock(cl[3], LockAction.AcquireGreaterEqual, value=1)
                                project(ci,cw,co,arith.index_cast(T.i32(),s))
                                use_lock(cl[2], LockAction.Release, value=1)
                            pool(co)
                            use_lock(cl[0], LockAction.Release, value=1)
                            use_lock(cl[5], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(cl[7], LockAction.Release, value=1)
                            use_lock(cl[4], LockAction.Release, value=1)

                # Static gather only. These IDs are never rewritten at runtime.
                @memtile_dma(mt)
                def gather_program(block):
                    specs = [(DMAChannelDir.MM2S,col,aggregate,0,51200,
                              [(25,128),(4,12832),(4,3200),(128,1)],lk[17],lk[18],2 if col%2==0 else 26)]
                    for src in range(4):
                        specs.append((DMAChannelDir.S2MM,src,stripe,src*32,2048,
                                      [(4,8),(4,128),(16,512),(8,1)],lk[19+src],lk[20+src],[0,24,1,25][src]))
                    for i,(direction,ch,buf,off,length,dims,acq,rel,ident) in enumerate(specs):
                        if i==0:
                            dma_start(direction,ch,dest=block[1],chain=block[2])
                        else:
                            with block[2*i]:
                                dma_start(direction,ch,dest=block[2*i+1],chain=block[2*i+2])
                        with block[2*i+1]:
                            use_lock(acq,LockAction.AcquireGreaterEqual,value=1)
                            dma_bd(buf,offset=off,len=length,dimensions=dims,bd_id=ident)
                            use_lock(rel,LockAction.Release,value=1)
                            next_bd(block[2*i+1])
                    with block[2*len(specs)]:
                        EndOp()

            @runtime_sequence(ty(102400),ty(33792),ty(819200),ty(512))
            def sequence(I,W,O,M):
                def task(t,direction,ch,records,repeat=0,issue=False):
                    result = dma_configure_task(t,direction,ch,repeat_count=repeat,issue_token=issue)
                    with bds(result) as blocks:
                        for j,(buf,off,length,acq,rel,ident,dims,pkt) in enumerate(records):
                            with blocks[j]:
                                use_lock(acq,LockAction.AcquireGreaterEqual,value=1)
                                dma_bd(buf,offset=off,len=length,bd_id=ident,dimensions=dims,packet=pkt)
                                use_lock(rel,LockAction.Release,value=1)
                                if j+1<len(records):
                                    next_bd(blocks[j+1])
                                else:
                                    EndOp()
                    return result

                cores, core_done, core_free = [],[],[]
                for index,(ct,ci,cw,co,cg,cl,row) in enumerate(core_state):
                    wr = task(ct,DMAChannelDir.S2MM,1,[(cw,0,2112,cl[0],cl[1],0,None,None)])
                    ir = task(ct,DMAChannelDir.S2MM,1,[(ci,0,4096,cl[2],cl[3],1,None,None)],24)
                    gr = task(ct,DMAChannelDir.S2MM,1,[(cg,0,32,cl[4],cl[5],2,None,None)],issue=True)
                    out = task(ct,DMAChannelDir.MM2S,0,[(co,0,12832,cl[7],cl[6],3,None,(0,16+row))],issue=True)
                    cores += [wr,ir,gr,out]
                    core_done.append((row,index//4,gr,out))
                    core_free += [wr,ir]

                mt_starts, mt_ingress, mt_done, mt_free = [],[],[],[]
                for col,(inp,weights,aggregate,stripe,token,lk) in enumerate(mt_state):
                    def mt_task(direction,ch,recs,repeat=0,issue=False):
                        t = task(mts[col],direction,ch,recs,repeat,issue)
                        mt_starts.append((col,direction,ch,recs[0][5],repeat,issue))
                        if issue and direction == DMAChannelDir.S2MM and ch == 5:
                            mt_ingress.append(t)
                        else:
                            (mt_done if issue else mt_free).append(t)
                        return t
                    mt_task(DMAChannelDir.S2MM,5,[
                        (inp,0,102400,lk[0],lk[1],28,None,None),
                        (weights,0,8448,lk[1],lk[2],29,None,None)],issue=True)
                    mt_task(DMAChannelDir.MM2S,5,[
                        (weights,r*2112,2112,lk[2+r],lk[3+r],30+r,None,(0,1<<r)) for r in range(4)])
                    mt_task(DMAChannelDir.MM2S,5,[
                        (inp,0,4096,lk[6+r],lk[6+(r+1)%4],34+r,
                         [(25,4096),(1,0),(1,0),(4096,1)],(0,1<<r)) for r in range(4)],24)
                    mt_task(DMAChannelDir.MM2S,5,[
                        (token,0,32,lk[6] if r==0 else lk[13+r],lk[10+r],38+r,None,(0,1<<r)) for r in range(4)],issue=True)
                    mt_task(DMAChannelDir.S2MM,4,[
                        (aggregate,r*12832,12832,lk[10+r],lk[14+r] if r<3 else lk[17],4+r,None,None) for r in range(4)],issue=True)
                    mt_task(DMAChannelDir.MM2S,4,[(stripe,0,8192,lk[23],lk[19],8,None,None)],24)
                    mt_task(DMAChannelDir.MM2S,4,[(aggregate,12800,128,lk[18],lk[0],9,
                                                [(1,0),(1,0),(4,12832),(32,1)],None)],issue=True)

                inputs, outputs = [],[]
                for col in range(4):
                    for host,off,n,ident in [(I,0,102400,0),(W,col*8448,8448,1)]:
                        t = dma_configure_task(shims[col],DMAChannelDir.MM2S,0)
                        with bds(t) as bd:
                            with bd[0]:
                                shim_dma_bd(host,offset=off,sizes=[1,1,1,n],strides=[0,0,0,1])
                                EndOp()
                        inputs.append(t)
                    for host,off,n,ident in [(O,col*204800,204800,2),(M,col*128,128,3)]:
                        t = dma_configure_task(shims[col],DMAChannelDir.S2MM,0,issue_token=True)
                        with bds(t) as bd:
                            with bd[0]:
                                shim_dma_bd(host,offset=off,sizes=[1,1,1,n],strides=[0,0,0,1])
                                EndOp()
                        outputs.append(t)
                dma_start_task(*cores)
                for args in mt_starts:
                    start_memtile_task_6bit(*args)
                dma_start_task(*outputs,*inputs)
                # Drain causally early tokens before waiting on end-to-end
                # outputs; avoid leaving all producer TCTs unsolicited.
                ordered_core_done = [t for _,_,gr,out in sorted(core_done) for t in (gr,out)]
                dma_await_task(*mt_ingress,*ordered_core_done,*mt_done,*outputs)
                dma_free_task(*inputs,*core_free,*mt_free)
    return ctx.module


if __name__ == "__main__":
    print(generate())
