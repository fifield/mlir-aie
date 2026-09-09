"""One-core finite A/25B phase alias sentinel; compile/runtime proof in progress.

Native uint16 I/O each217600: A12800 then25 B8192 stripes. One explicit
50240-byte arena at8192 and8192-byte stack. Four terminating runtime core BDs.
"""
import sys
import numpy as np
from aie.dialects.aie import (AIEDevice, DMAChannelDir, WireBundle, LockAction,
    device, tile, buffer, lock, flow, packetflow, core, dma_bd, use_lock, EndOp, external_func)
from aie.dialects.aiex import (runtime_sequence, dma_configure_task, bds,
    shim_dma_bd, dma_start_task, dma_await_task, dma_free_task)
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.dialects import arith
from aie.extras import types as T
from aie.ir import Attribute, BoolAttr


def generate():
    ty = lambda n: np.ndarray[(n,), np.dtype[np.uint16]]
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def body():
            shim, ct = tile(0, 0), tile(0, 2)
            ct.operation.attributes["controller_id"] = Attribute.parse(
                "#aie.packet_info<pkt_type = 0, pkt_id = 27>")
            arena = buffer(ct, ty(25120), name="phase_arena", address=8192)
            tag = external_func("phase_alias_tag", [ty(25120), np.int32, np.int32],
                                link_with="phase_alias.o")
            frame_empty = lock(ct, lock_id=0, init=1, sym_name="frame_empty")
            a_ready = lock(ct, lock_id=1, init=0, sym_name="Aready")
            a_send = lock(ct, lock_id=2, init=0, sym_name="Asend")
            b_empty = lock(ct, lock_id=3, init=0, sym_name="Bempty")
            b_ready = lock(ct, lock_id=4, init=0, sym_name="Bready")
            b_send = lock(ct, lock_id=5, init=0, sym_name="Bsend")
            flow(shim, WireBundle.DMA, 0, ct, WireBundle.DMA, 1)
            flow(ct, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)
            # Explicit core task-completion route; the default overlay adds
            # only the separate shim controller route. Preserve TCT headers.
            tct = packetflow(27, ct, WireBundle.TileControl, 0,
                             {"dest": shim, "port": WireBundle.South, "channel": 0},
                             keep_pkt_header=True)
            tct.operation.attributes["ctrl_pkt_flow"] = BoolAttr.get(True)

            @core(ct, stack_size=8192)
            def worker():
                for _ in range_(sys.maxsize):
                    use_lock(a_ready, LockAction.AcquireGreaterEqual, value=1)
                    tag(arena, 12800, 0xA5A5)
                    use_lock(a_send, LockAction.Release, value=1)
                    for stripe in range_(25):
                        use_lock(b_ready, LockAction.AcquireGreaterEqual, value=1)
                        # 1..25 has no bits in common with 0x5A00.
                        mask = arith.index_cast(T.i32(), stripe) + 0x5A01
                        tag(arena, 8192, mask)
                        use_lock(b_send, LockAction.Release, value=1)
                    use_lock(b_empty, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(frame_empty, LockAction.Release, value=1)

            @runtime_sequence(ty(217600), ty(217600))
            def sequence(I, O):
                def core_task(direction, channel, length, acquire, release, repeat):
                    task = dma_configure_task(ct, direction, channel,
                                              repeat_count=repeat, issue_token=bool(repeat))
                    with bds(task) as bd:
                        with bd[0]:
                            use_lock(acquire, LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(arena, offset=0, len=length)
                            use_lock(release, LockAction.Release, value=1)
                            EndOp()
                    return task

                a_rx = core_task(DMAChannelDir.S2MM, 1, 12800, frame_empty, a_ready, 0)
                a_tx = core_task(DMAChannelDir.MM2S, 0, 12800, a_send, b_empty, 0)
                b_rx = core_task(DMAChannelDir.S2MM, 1, 8192, b_empty, b_ready, 24)
                b_tx = core_task(DMAChannelDir.MM2S, 0, 8192, b_send, b_empty, 24)
                input_task = dma_configure_task(shim, DMAChannelDir.MM2S, 0)
                with bds(input_task) as bd:
                    with bd[0]:
                        shim_dma_bd(I, sizes=[1,1,1,217600], strides=[0,0,0,1])
                        EndOp()
                output_task = dma_configure_task(shim, DMAChannelDir.S2MM, 0, issue_token=True)
                with bds(output_task) as bd:
                    with bd[0]:
                        shim_dma_bd(O, sizes=[1,1,1,217600], strides=[0,0,0,1])
                        EndOp()
                dma_start_task(a_rx, a_tx, b_rx, b_tx, output_task, input_task)
                # B completion tokens fence both finite core queues; data
                # arrival at the shim alone is not used as a queue-idle proof.
                dma_await_task(b_rx, b_tx, output_task)
                # Allocation bookkeeping only, not a device completion fence.
                dma_free_task(input_task, a_rx, a_tx)
    return ctx.module


if __name__ == "__main__":
    print(generate())
