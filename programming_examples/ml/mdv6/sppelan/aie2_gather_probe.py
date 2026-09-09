"""Direct memtile gather-only proof, with no arithmetic or compute workers.

I uint16 [source4,row4,level4,pixel400,lane8], O uint16
[destination4,stripe25,pixel16,K512]. --remote-only compiles one source at
column 0 to destination column 1, with I/O each 51200 uint16 elements.
"""
import argparse
import numpy as np
from aie.dialects.aie import (AIEDevice, DMAChannelDir, WireBundle, LockAction,
    device, tile, buffer, lock, flow, memtile_dma, dma_start, dma_bd, use_lock,
    next_bd, EndOp)
from aie.dialects.aiex import (runtime_sequence, dma_configure_task, bds,
    shim_dma_bd, dma_start_task, dma_await_task, dma_free_task)
from aie.extras.context import mlir_mod_ctx


def generate(remote_only=False):
    sources = [0] if remote_only else list(range(4))
    destinations = [1] if remote_only else list(range(4))
    source_size = 51200
    part_size = 2048
    stripe_size = part_size if remote_only else 8192
    ty = lambda size: np.ndarray[(size,), np.dtype[np.uint16]]
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def body():
            columns = sorted(set(sources + destinations))
            shims = {c: tile(c, 0) for c in columns}
            mems = {c: tile(c, 1) for c in columns}
            for src in sources:
                flow(shims[src], WireBundle.DMA, 0, mems[src], WireBundle.DMA, 4)
                for dest in destinations:
                    # Local memtile loopback requires matching DMA channels.
                    flow(mems[src], WireBundle.DMA, src,
                         mems[dest], WireBundle.DMA, src)
            for dest in destinations:
                flow(mems[dest], WireBundle.DMA, 4, shims[dest], WireBundle.DMA, 0)

            for col in columns:
                source = buffer(mems[col], ty(source_size), name=f"source_{col}") if col in sources else None
                stripe = buffer(mems[col], ty(stripe_size), name=f"stripe_{col}") if col in destinations else None
                empty = lock(mems[col], lock_id=16, init=1, sym_name=f"source_empty_{col}") if source is not None else None
                ready = lock(mems[col], lock_id=17, init=0, sym_name=f"source_ready_{col}") if source is not None else None
                turns = [lock(mems[col], lock_id=i, init=int(i == 0),
                              sym_name=f"stripe_turn_{col}_{i}")
                         for i in range(len(sources) + 1)] if stripe is not None else []

                # Each record is one permanent self-looping BD with one
                # acquire/release pair; no dynamic descriptor reprogramming.
                specs = []
                if source is not None:
                    specs.extend([
                        (DMAChannelDir.S2MM, 4, source, 0, source_size, None, empty, ready),
                        (DMAChannelDir.MM2S, col, source, 0, source_size,
                         [(25, 128), (4, 12800), (4, 3200), (128, 1)], ready, empty)])
                if stripe is not None:
                    for index, src in enumerate(sources):
                        dims = None if remote_only else [(4, 8), (4, 128), (16, 512), (8, 1)]
                        specs.append((DMAChannelDir.S2MM, src, stripe,
                                      0 if remote_only else src * 32, part_size,
                                      dims, turns[index], turns[index + 1]))
                    specs.append((DMAChannelDir.MM2S, 4, stripe, 0, stripe_size,
                                  None, turns[-1], turns[0]))

                @memtile_dma(mems[col])
                def mem_program(block):
                    for index, (direction, channel, buf, offset, length, dims, acquire, release) in enumerate(specs):
                        if index == 0:
                            dma_start(direction, channel, dest=block[1], chain=block[2])
                        else:
                            with block[2 * index]:
                                dma_start(direction, channel, dest=block[2 * index + 1], chain=block[2 * index + 2])
                        with block[2 * index + 1]:
                            use_lock(acquire, LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(buf, offset=offset, len=length, dimensions=dims)
                            use_lock(release, LockAction.Release, value=1)
                            next_bd(block[2 * index + 1])
                    with block[2 * len(specs)]:
                        EndOp()

            @runtime_sequence(ty(source_size * len(sources)), ty(25 * stripe_size * len(destinations)))
            def sequence(I, O):
                input_tasks, output_tasks = [], []
                for index, src in enumerate(sources):
                    task = dma_configure_task(shims[src], DMAChannelDir.MM2S, 0)
                    with bds(task) as bd:
                        with bd[0]:
                            shim_dma_bd(I, offset=index * source_size,
                                        sizes=[1, 1, 1, source_size], strides=[0, 0, 0, 1])
                            EndOp()
                    input_tasks.append(task)
                for index, dest in enumerate(destinations):
                    task = dma_configure_task(shims[dest], DMAChannelDir.S2MM, 0, issue_token=True)
                    with bds(task) as bd:
                        with bd[0]:
                            shim_dma_bd(O, offset=index * 25 * stripe_size,
                                        sizes=[1, 1, 1, 25 * stripe_size], strides=[0, 0, 0, 1])
                            EndOp()
                    output_tasks.append(task)
                dma_start_task(*output_tasks, *input_tasks)
                dma_await_task(*output_tasks)
                dma_free_task(*input_tasks)
    return ctx.module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-only", action="store_true")
    print(generate(parser.parse_args().remote_only))
