"""One-column, four-worker packet aggregation with receive-gated grants.

ABI I/O: uint16 [worker4,level4,pixel400,lane8]. Output is input XOR
0x1111*(worker+1). Reserved memtile channels: S2MM5 ingress, S2MM4 worker
aggregation, MM2S5 addressed payload/grants, MM2S1 final egress. Gather's
S2MM0..3 and MM2S0, plus activation MM2S4, remain unused by this cut.
"""
import sys
import numpy as np
from aie.dialects.aie import (AIEDevice, DMAChannelDir, WireBundle, LockAction,
    device, tile, buffer, lock, flow, packetflow, memtile_dma, mem, core,
    dma_start, dma_bd, use_lock, next_bd, EndOp, external_func)
from aie.dialects.aiex import (runtime_sequence, dma_configure_task, bds,
    shim_dma_bd, dma_start_task, dma_await_task, dma_free_task)
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def dma_chains(block, chains):
    """Emit permanent channel chains; each record is one BD, not a host task."""
    cursor = 0
    for channel_index, (direction, channel, records) in enumerate(chains):
        first, following = cursor + 1, cursor + 1 + len(records)
        if channel_index == 0:
            dma_start(direction, channel, dest=block[first], chain=block[following])
        else:
            with block[cursor]:
                dma_start(direction, channel, dest=block[first], chain=block[following])
        for index, (buf, offset, length, acquire, release, packet) in enumerate(records):
            with block[first + index]:
                if acquire is not None:
                    use_lock(acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf, offset=offset, len=length, packet=packet)
                if release is not None:
                    use_lock(release, LockAction.Release, value=1)
                next_bd(block[first + ((index + 1) % len(records))])
        cursor = following
    with block[cursor]:
        EndOp()


def generate():
    ty = lambda n: np.ndarray[(n,), np.dtype[np.uint16]]
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def body():
            shim, mt = tile(0, 0), tile(0, 1)
            workers = [tile(0, row + 2) for row in range(4)]
            tag = external_func("packet_aggregate_tag", [ty(12800), np.int32],
                                link_with="packet_aggregate.o")
            source = buffer(mt, ty(51200), name="packet_source")
            aggregate = buffer(mt, ty(51200), name="packet_aggregate")
            token = buffer(mt, ty(32), name="grant_token",
                           initial_value=np.zeros(32, np.uint16))
            stage_empty = lock(mt, lock_id=0, init=1, sym_name="stage_empty")
            stage_ready = lock(mt, lock_id=1, init=0, sym_name="stage_ready")
            grants = [lock(mt, lock_id=2 + row, init=0, sym_name=f"grant_ready_{row}") for row in range(4)]
            receivers = [lock(mt, lock_id=6 + row, init=0, sym_name=f"receive_ready_{row}") for row in range(4)]
            output_ready = lock(mt, lock_id=10, init=0, sym_name="output_ready")
            payload_ready = [lock(mt, lock_id=11 + row, init=0,
                                  sym_name=f"payload_ready_{row}") for row in range(3)]

            flow(shim, WireBundle.DMA, 0, mt, WireBundle.DMA, 5)
            flow(mt, WireBundle.DMA, 1, shim, WireBundle.DMA, 0)
            for row, ct in enumerate(workers):
                # One-hot destinations prevent merged packet masks from
                # matching a different worker along the shared north path.
                packetflow(1 << row, mt, WireBundle.DMA, 5,
                           {"dest": ct, "port": WireBundle.DMA, "channel": 1},
                           keep_pkt_header=False)
                packetflow(16 + row, ct, WireBundle.DMA, 0,
                           {"dest": mt, "port": WireBundle.DMA, "channel": 4},
                           keep_pkt_header=False)
                planes = buffer(ct, ty(12800), name=f"worker_planes_{row}")
                grant = buffer(ct, ty(32), name=f"worker_grant_{row}")
                feature_empty = lock(ct, lock_id=0, init=1, sym_name=f"feature_empty_{row}")
                feature_ready = lock(ct, lock_id=1, init=0, sym_name=f"feature_ready_{row}")
                grant_empty = lock(ct, lock_id=2, init=1, sym_name=f"grant_empty_{row}")
                grant_ready = lock(ct, lock_id=3, init=0, sym_name=f"worker_grant_ready_{row}")
                send_ready = lock(ct, lock_id=4, init=0, sym_name=f"send_ready_{row}")

                @core(ct, stack_size=4096)
                def worker():
                    for _ in range_(sys.maxsize):
                        use_lock(feature_ready, LockAction.AcquireGreaterEqual, value=1)
                        tag(planes, row)
                        use_lock(grant_ready, LockAction.AcquireGreaterEqual, value=1)
                        use_lock(send_ready, LockAction.Release, value=1)
                        use_lock(grant_empty, LockAction.Release, value=1)

                @mem(ct)
                def worker_dma(block):
                    dma_chains(block, [
                        (DMAChannelDir.S2MM, 1, [
                            (planes, 0, 12800, feature_empty, feature_ready, None),
                            (grant, 0, 32, grant_empty, grant_ready, None)]),
                        (DMAChannelDir.MM2S, 0, [
                            (planes, 0, 12800, send_ready, feature_empty, (0, 16 + row))])])

            @memtile_dma(mt)
            def mem_program(block):
                sends = []
                # Reverse readiness order deliberately differs from the
                # required receive/grant order; no host repacking is used.
                for row in reversed(range(4)):
                    sends.append((source, row * 12800, 12800,
                                  stage_ready if row == 3 else payload_ready[row],
                                  grants[0] if row == 0 else payload_ready[row - 1],
                                  (0, 1 << row)))
                for row in range(4):
                    sends.append((token, 0, 32, grants[row], receivers[row], (0, 1 << row)))
                receives = [(aggregate, row * 12800, 12800, receivers[row],
                             grants[row + 1] if row < 3 else output_ready, None)
                            for row in range(4)]
                dma_chains(block, [
                    (DMAChannelDir.S2MM, 5, [(source, 0, 51200, stage_empty, stage_ready, None)]),
                    (DMAChannelDir.MM2S, 5, sends),
                    (DMAChannelDir.S2MM, 4, receives),
                    (DMAChannelDir.MM2S, 1, [(aggregate, 0, 51200, output_ready, stage_empty, None)])])

            @runtime_sequence(ty(51200), ty(51200))
            def sequence(I, O):
                input_task = dma_configure_task(shim, DMAChannelDir.MM2S, 0)
                with bds(input_task) as bd:
                    with bd[0]:
                        shim_dma_bd(I, sizes=[1, 1, 1, 51200], strides=[0, 0, 0, 1])
                        EndOp()
                output_task = dma_configure_task(shim, DMAChannelDir.S2MM, 0, issue_token=True)
                with bds(output_task) as bd:
                    with bd[0]:
                        shim_dma_bd(O, sizes=[1, 1, 1, 51200], strides=[0, 0, 0, 1])
                        EndOp()
                dma_start_task(output_task, input_task)
                dma_await_task(output_task)
                dma_free_task(input_task)
    return ctx.module


if __name__ == "__main__":
    print(generate())
