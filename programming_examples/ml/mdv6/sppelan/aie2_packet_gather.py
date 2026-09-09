"""Resident full-plane packet aggregation followed by four-column gather.

I uint16 [source4,worker4,level4,pixel400,lane8], O uint16
[destination4,stripe25,pixel16,K512]. Workers apply opaque XOR tags only.
No host round trip occurs between aggregation and gather. This does not
implement numerical SPP, a finite phase-A/B controller, or L1 phase aliasing.
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
    """Permanent cyclic BDs, each with one ownership acquire/release pair."""
    cursor = 0
    for channel_index, (direction, channel, records) in enumerate(chains):
        first, following = cursor + 1, cursor + 1 + len(records)
        if channel_index == 0:
            dma_start(direction, channel, dest=block[first], chain=block[following])
        else:
            with block[cursor]:
                dma_start(direction, channel, dest=block[first], chain=block[following])
        for index, (buf, offset, length, acquire, release, packet, dims) in enumerate(records):
            with block[first + index]:
                use_lock(acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf, offset=offset, len=length, packet=packet, dimensions=dims)
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
            shims = [tile(c, 0) for c in range(4)]
            mts = [tile(c, 1) for c in range(4)]
            tag = external_func("packet_gather_tag", [ty(12800), np.int32],
                                link_with="packet_gather.o")
            for col in range(4):
                flow(shims[col], WireBundle.DMA, 0, mts[col], WireBundle.DMA, 5)
                flow(mts[col], WireBundle.DMA, 4, shims[col], WireBundle.DMA, 0)
                for dest in range(4):
                    # The local memtile loopback must use equal DMA channels.
                    flow(mts[col], WireBundle.DMA, col, mts[dest], WireBundle.DMA, col)

            for col, mt in enumerate(mts):
                source = buffer(mt, ty(51200), name=f"packet_source_{col}")
                aggregate = buffer(mt, ty(51200), name=f"packet_aggregate_{col}")
                token = buffer(mt, ty(32), name=f"grant_token_{col}",
                               initial_value=np.zeros(32, np.uint16))
                stripe = buffer(mt, ty(8192), name=f"stripe_{col}")
                stage_empty = lock(mt, lock_id=0, init=1, sym_name=f"stage_empty_{col}")
                stage_ready = lock(mt, lock_id=1, init=0, sym_name=f"stage_ready_{col}")
                grants = [lock(mt, lock_id=2+r, init=0, sym_name=f"grant_ready_{col}_{r}") for r in range(4)]
                receivers = [lock(mt, lock_id=6+r, init=0, sym_name=f"receive_ready_{col}_{r}") for r in range(4)]
                output_ready = lock(mt, lock_id=10, init=0, sym_name=f"output_ready_{col}")
                payload_ready = [lock(mt, lock_id=11+r, init=0, sym_name=f"payload_ready_{col}_{r}") for r in range(3)]
                turns = [lock(mt, lock_id=14+r, init=int(r == 0), sym_name=f"stripe_turn_{col}_{r}") for r in range(5)]

                for row in range(4):
                    ct = tile(col, row + 2)
                    # Reused IDs remain confined to each column's packet paths.
                    packetflow(1 << row, mt, WireBundle.DMA, 5,
                               {"dest": ct, "port": WireBundle.DMA, "channel": 1}, keep_pkt_header=False)
                    packetflow(16 + row, ct, WireBundle.DMA, 0,
                               {"dest": mt, "port": WireBundle.DMA, "channel": 4}, keep_pkt_header=False)
                    planes = buffer(ct, ty(12800), name=f"worker_planes_{col}_{row}")
                    grant = buffer(ct, ty(32), name=f"worker_grant_{col}_{row}")
                    feature_empty = lock(ct, lock_id=0, init=1, sym_name=f"feature_empty_{col}_{row}")
                    feature_ready = lock(ct, lock_id=1, init=0, sym_name=f"feature_ready_{col}_{row}")
                    grant_empty = lock(ct, lock_id=2, init=1, sym_name=f"grant_empty_{col}_{row}")
                    grant_ready = lock(ct, lock_id=3, init=0, sym_name=f"worker_grant_ready_{col}_{row}")
                    send_ready = lock(ct, lock_id=4, init=0, sym_name=f"send_ready_{col}_{row}")

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
                                (planes, 0, 12800, feature_empty, feature_ready, None, None),
                                (grant, 0, 32, grant_empty, grant_ready, None, None)]),
                            (DMAChannelDir.MM2S, 0, [
                                (planes, 0, 12800, send_ready, feature_empty, (0, 16 + row), None)])])

                @memtile_dma(mt)
                def mem_program(block):
                    sends = []
                    # Deliberately reverse input readiness relative to grants.
                    for row in reversed(range(4)):
                        sends.append((source, row*12800, 12800,
                                      stage_ready if row == 3 else payload_ready[row],
                                      grants[0] if row == 0 else payload_ready[row-1],
                                      (0, 1 << row), None))
                    for row in range(4):
                        sends.append((token, 0, 32, grants[row], receivers[row], (0, 1 << row), None))
                    receives = [(aggregate, row*12800, 12800, receivers[row],
                                 grants[row+1] if row < 3 else output_ready, None, None) for row in range(4)]
                    chains = [
                        (DMAChannelDir.S2MM, 5, [(source, 0, 51200, stage_empty, stage_ready, None, None)]),
                        (DMAChannelDir.MM2S, 5, sends),
                        (DMAChannelDir.S2MM, 4, receives),
                        # Retain the aggregate until the entire multicast ends.
                        (DMAChannelDir.MM2S, col, [(aggregate, 0, 51200, output_ready, stage_empty,
                                                   None, [(25,128),(4,12800),(4,3200),(128,1)])])]
                    for src in range(4):
                        chains.append((DMAChannelDir.S2MM, src, [
                            (stripe, src*32, 2048, turns[src], turns[src+1],
                             None, [(4,8),(4,128),(16,512),(8,1)])]))
                    chains.append((DMAChannelDir.MM2S, 4, [
                        (stripe, 0, 8192, turns[4], turns[0], None, None)]))
                    dma_chains(block, chains)

            @runtime_sequence(ty(204800), ty(819200))
            def sequence(I, O):
                inputs, outputs = [], []
                for col in range(4):
                    task = dma_configure_task(shims[col], DMAChannelDir.MM2S, 0)
                    with bds(task) as bd:
                        with bd[0]:
                            shim_dma_bd(I, offset=col*51200, sizes=[1,1,1,51200], strides=[0,0,0,1])
                            EndOp()
                    inputs.append(task)
                for col in range(4):
                    task = dma_configure_task(shims[col], DMAChannelDir.S2MM, 0, issue_token=True)
                    with bds(task) as bd:
                        with bd[0]:
                            shim_dma_bd(O, offset=col*204800, sizes=[1,1,1,204800], strides=[0,0,0,1])
                            EndOp()
                    outputs.append(task)
                dma_start_task(*outputs, *inputs)
                dma_await_task(*outputs)
                dma_free_task(*inputs)
    return ctx.module


if __name__ == "__main__":
    print(generate())
