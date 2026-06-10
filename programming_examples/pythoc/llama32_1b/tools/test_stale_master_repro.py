#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Minimal repro: stale switch state from one PDI deadlocks the next device
(RESIDENT_DEVICE_EVOLUTION.md C2 blocker, commit ff1b778f0).

All devices are kernel-less and self-contained:

  park_mini_rgr   ring-shifted circuit trees (shim c MM2S1 -> ct row2 col c+5,
                  MM2S0 -> ct row3 col c+3, drains back per col) — long E/W
                  walks on the shim row park circuit masters rgr-style.
  victim_pkt      8-col packet-broadcast passthrough (C2-style routing)
  victim_circuit  same shape, circuit broadcast

Protocol: park -> victims. On a FRESH device (driver reset / boot) all three
run individually; after the park, a victim deadlock = the bug (LoadPDI only
writes the ports the new device uses, so the parked masters keep forking the
victim stream into disarmed destinations).

PRECONDITION: a blank device. Any production decode run leaves leftovers
that deadlock every newcomer (including the park itself) — that IS the bug,
but it makes the self-contained sequence unrunnable afterwards without a
driver reset. On a machine where decode already ran, use `--park-rgr` to
park via the production rms_gemv_rope ELF (rgr runs from any state;
deterministically reproduces today: rgr ok -> both victims deadlock).

Usage: test_stale_master_repro.py [--compile-only] [--no-park] [--park-rgr]
Exit 0 = no bug, 1 = bug reproduced, 2 = device unusable.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

N_COLS = 8
N = 256  # elements per column


def build_module(want_packet: bool) -> str:
    from aie.dialects.aie import (
        AIEDevice, DMAChannelDir, LockAction, WireBundle, buffer, core,
        device, dma_bd, dma_start, flow, lock, mem, next_bd, packetflow,
        shim_dma_allocation, tile, use_lock,
    )
    from aie.dialects.aiex import (
        EndOp, bds, dma_await_task, dma_configure_task_for, dma_free_task,
        dma_start_task, runtime_sequence,
    )
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp
    from aie.extras.context import mlir_mod_ctx
    from aie.ir import InsertionPoint
    from builders._emit import bf16_memref, bf16_np

    host_args = [bf16_np(N)] + [bf16_np(N) for _ in range(N_COLS)]
    sym = "victim_pkt" if want_packet else "victim_circuit"

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2, sym_name=sym)
        def _dev():
            shims = [tile(c, 0) for c in range(N_COLS)]
            cts = [tile(c, 2) for c in range(N_COLS)]
            lks = {}
            bufs = {}
            for c in range(N_COLS):
                lks[c] = {
                    "in_avail": lock(cts[c], lock_id=1, init=1),
                    "in_ready": lock(cts[c], lock_id=0, init=0),
                }
                bufs[c] = buffer(cts[c], datatype=bf16_memref(N, memory_space=2))

            for c in range(N_COLS):
                def _mk(_ct, _lk, _b):
                    @mem(_ct)
                    def _m(block):
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                        with block[1]:
                            use_lock(_lk["in_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_b, offset=0, len=N)
                            use_lock(_lk["in_ready"], LockAction.Release, value=1)
                            next_bd(block[1])
                        with block[2]:
                            EndOp()
                        with block[3]:
                            dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[2])
                        with block[4]:
                            use_lock(_lk["in_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if want_packet:
                                dma_bd(_b, offset=0, len=N, packet=(0, 8))
                            else:
                                dma_bd(_b, offset=0, len=N)
                            use_lock(_lk["in_avail"], LockAction.Release, value=1)
                            next_bd(block[4])

                _mk(cts[c], lks[c], bufs[c])

                @core(cts[c])
                def _c():
                    pass  # DMA-only tile

            if want_packet:
                packetflow(
                    pkt_id=1,
                    source=shims[0], source_port=WireBundle.DMA, source_channel=1,
                    dests=[{"dest": cts[c], "port": WireBundle.DMA, "channel": 0}
                           for c in range(N_COLS)],
                )
                for c in range(N_COLS):
                    packetflow(
                        pkt_id=8,
                        source=cts[c], source_port=WireBundle.DMA, source_channel=0,
                        dests={"dest": shims[c], "port": WireBundle.DMA, "channel": 0},
                    )
            else:
                for c in range(N_COLS):
                    flow(shims[0], WireBundle.DMA, 1, cts[c], WireBundle.DMA, 0)
                    flow(cts[c], WireBundle.DMA, 0, shims[c], WireBundle.DMA, 0)

            shim_dma_allocation(f"{sym}_x", shims[0], DMAChannelDir.MM2S, 1)
            for c in range(N_COLS):
                shim_dma_allocation(f"{sym}_y_{c}", shims[c], DMAChannelDir.S2MM, 0)

            @runtime_sequence(*host_args, sym_name=f"{sym}_sequence")
            def _seq(*args):
                xt = dma_configure_task_for(f"{sym}_x", repeat_count=0)
                with bds(xt) as bd:
                    with bd[0]:
                        if want_packet:
                            dma_bd(args[0], offset=0, len=N, packet=(0, 1))
                        else:
                            dma_bd(args[0], offset=0, len=N)
                        EndOp()
                dma_start_task(xt)
                outs = []
                for c in range(N_COLS):
                    t = dma_configure_task_for(f"{sym}_y_{c}", issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(args[1 + c], offset=0, len=N)
                            EndOp()
                    dma_start_task(t)
                    outs.append(t)
                for t in reversed(outs):
                    dma_await_task(t)
                dma_free_task(xt)

        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(*host_args, sym_name=sym + "_top")
            def _outer(*args):
                cfg = ConfigureOp(symbol=sym)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(runtime_sequence_symbol=f"{sym}_sequence", args=list(args))

        return str(ctx.module)



def build_park_module() -> str:
    """Mini-rgr park device: kernel-less, but with rgr's structural footprint.

    Two circuit broadcast trees across the full shim row (shim0 MM2S1 -> all
    8 ct(c,2); shim1 MM2S1 -> all 8 ct(c,3)) plus per-column config-only
    W chains (shim MM2S0 -> mem S2MM0, mem MM2S1 -> ct S2MM1) -- those carry
    no traffic, they only park switch masters the way rgr's weight chains do.
    Each ct loops 256 bf16 back to its shim (row2 -> S2MM0, row3 -> S2MM1).
    """
    from aie.dialects.aie import (
        AIEDevice, DMAChannelDir, LockAction, WireBundle, buffer, core,
        device, dma_bd, dma_start, flow, lock, mem, next_bd,
        shim_dma_allocation, tile, use_lock,
    )
    from aie.dialects.aiex import (
        EndOp, bds, dma_await_task, dma_configure_task_for, dma_free_task,
        dma_start_task, runtime_sequence,
    )
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp
    from aie.extras.context import mlir_mod_ctx
    from aie.ir import InsertionPoint
    from builders._emit import bf16_memref, bf16_np

    host_args = [bf16_np(N)] + [bf16_np(N) for _ in range(N_COLS)]
    sym = "park_mini_rgr"

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2, sym_name=sym)
        def _dev():
            shims = [tile(c, 0) for c in range(N_COLS)]
            mems = [tile(c, 1) for c in range(N_COLS)]
            r2 = [tile(c, 2) for c in range(N_COLS)]
            r3 = [tile(c, 3) for c in range(N_COLS)]

            def passthrough(ct):
                lk_av = lock(ct, lock_id=1, init=1)
                lk_rd = lock(ct, lock_id=0, init=0)
                b = buffer(ct, datatype=bf16_memref(N, memory_space=2))

                @mem(ct)
                def _m(block):
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(lk_av, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(b, offset=0, len=N)
                        use_lock(lk_rd, LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[2])
                    with block[4]:
                        use_lock(lk_rd, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(b, offset=0, len=N)
                        use_lock(lk_av, LockAction.Release, value=1)
                        next_bd(block[4])

                @core(ct)
                def _c():
                    pass

            for c in range(N_COLS):
                passthrough(r2[c])
                passthrough(r3[c])
                # config-only mem-tile W chain (parks masters, carries nothing)
                m_av = lock(mems[c], lock_id=1, init=1)
                m_rd = lock(mems[c], lock_id=0, init=0)
                mb = buffer(mems[c], datatype=bf16_memref(N, memory_space=1))

                def _mk_mem(_mt, _av, _rd, _b):
                    from aie.dialects.aie import memtile_dma
                    @memtile_dma(_mt)
                    def _mt_dma(block):
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                        with block[1]:
                            use_lock(_av, LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_b, offset=0, len=N)
                            use_lock(_rd, LockAction.Release, value=1)
                            next_bd(block[1])
                        with block[2]:
                            EndOp()
                        with block[3]:
                            dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[2])
                        with block[4]:
                            use_lock(_rd, LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_b, offset=0, len=N)
                            use_lock(_av, LockAction.Release, value=1)
                            next_bd(block[4])
                _mk_mem(mems[c], m_av, m_rd, mb)

            # Ring-shifted trees: shim c feeds ct row2 of col (c+5)%8 over
            # MM2S1 and ct row3 of col (c+3)%8 over MM2S0 -- long E/W walks
            # on row 0 park masters across the whole shim row (rgr-like).
            for c in range(N_COLS):
                flow(shims[c], WireBundle.DMA, 1, r2[(c + 5) % N_COLS], WireBundle.DMA, 0)
                flow(shims[c], WireBundle.DMA, 0, r3[(c + 3) % N_COLS], WireBundle.DMA, 0)
                flow(r2[c], WireBundle.DMA, 0, shims[c], WireBundle.DMA, 0)
                flow(r3[c], WireBundle.DMA, 0, shims[c], WireBundle.DMA, 1)
                # config-only mem chain (parks row-1 masters, carries nothing)
                flow(mems[c], WireBundle.DMA, 1, r2[c], WireBundle.DMA, 1)

            for c in range(N_COLS):
                shim_dma_allocation(f"{sym}_x2_{c}", shims[c], DMAChannelDir.MM2S, 1)
                shim_dma_allocation(f"{sym}_x3_{c}", shims[c], DMAChannelDir.MM2S, 0)
                shim_dma_allocation(f"{sym}_y2_{c}", shims[c], DMAChannelDir.S2MM, 0)
                shim_dma_allocation(f"{sym}_y3_{c}", shims[c], DMAChannelDir.S2MM, 1)

            @runtime_sequence(*host_args, sym_name=f"{sym}_sequence")
            def _seq(*args):
                xts = []
                for c in range(N_COLS):
                    for nm in (f"{sym}_x2_{c}", f"{sym}_x3_{c}"):
                        t = dma_configure_task_for(nm, repeat_count=0)
                        with bds(t) as bd:
                            with bd[0]:
                                dma_bd(args[0], offset=0, len=N)
                                EndOp()
                        dma_start_task(t)
                        xts.append(t)
                outs = []
                for c in range(N_COLS):
                    for nm in (f"{sym}_y2_{c}", f"{sym}_y3_{c}"):
                        t = dma_configure_task_for(nm, issue_token=True)
                        with bds(t) as bd:
                            with bd[0]:
                                dma_bd(args[1 + c], offset=0, len=N)
                                EndOp()
                        dma_start_task(t)
                        outs.append(t)
                for t in reversed(outs):
                    dma_await_task(t)
                for t in xts:
                    dma_free_task(t)

        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(*host_args, sym_name=sym + "_top")
            def _outer(*args):
                cfg = ConfigureOp(symbol=sym)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(runtime_sequence_symbol=f"{sym}_sequence", args=list(args))

        return str(ctx.module)


def main() -> int:
    os.chdir(PROJECT_DIR / "build_peano")
    from kernel_builder.cache import KernelCache

    cache = KernelCache(cache_dir=PROJECT_DIR / "build_peano" / "stale_master_cache",
                        verbose=False)
    if not (cache.load_manifest() and "victim_pkt" in cache.artifacts
            and "victim_circuit" in cache.artifacts
            and "park_mini_rgr" in cache.artifacts):
        cache.compile_and_cache("victim_pkt", build_module(True),
                                instance_name="victim_pkt")
        cache.compile_and_cache("victim_circuit", build_module(False),
                                instance_name="victim_circuit")
        cache.compile_and_cache("park_mini_rgr", build_park_module(),
                                instance_name="park_mini_rgr")
        cache._save_manifest()
        print("compiled", flush=True)
    if "--compile-only" in sys.argv:
        return 0

    x = (np.arange(N) % 251 + 1).astype(bfloat16)
    ys = [np.zeros(N, dtype=bfloat16) for _ in range(N_COLS)]
    out_idx = list(range(1, 1 + N_COLS))

    use_rgr = "--park-rgr" in sys.argv
    if use_rgr:
        EMB, KV = 2048, 512
        z = lambda n: np.zeros(n, dtype=bfloat16)
        rgr_args = [z(EMB), z(EMB), z(EMB), z((EMB, EMB)), z(EMB), z((KV, EMB)),
                    z(KV), z((KV, EMB)), z(KV), z(64), z(64), z(EMB), z(KV)]
        dc = KernelCache(cache_dir=PROJECT_DIR / "build_peano" / "decode_kernel_cache",
                         verbose=False)
        if not dc.load_manifest():
            print("--park-rgr needs decode_kernel_cache (`make compile`)")
            return 2

    def park_retry(tag):
        for i in range(3):
            try:
                if use_rgr:
                    dc.load_and_run("rms_gemv_rope", None, *rgr_args,
                                    output_indices=[11, 12], bo_key=f"{tag}{i}")
                else:
                    cache.load_and_run("park_mini_rgr", None, x, *ys,
                                       output_indices=out_idx, bo_key=f"{tag}{i}")
                return True
            except RuntimeError:
                print(f"park {tag}{i} consumed a wedge", flush=True)
        return False

    if "--no-park" not in sys.argv:
        # rgr doubles as the sacrifice: it runs from any device state.
        if not park_retry("s"):
            print("device unusable (park never ran) — inconclusive")
            return 2
        print("park ok — circuit trees parked", flush=True)

    verdicts = []
    for victim in ("victim_pkt", "victim_circuit"):
        try:
            t0 = time.perf_counter()
            res = cache.load_and_run(victim, None, x, *ys, output_indices=out_idx,
                                     bo_key=victim)
            bad = [c for c in range(N_COLS)
                   if not np.array_equal(res[1 + c].view(np.uint16),
                                         x.view(np.uint16))]
            print(f"{victim}: {'ok' if not bad else f'BAD cols {bad}'} "
                  f"({time.perf_counter() - t0:.2f}s)")
            verdicts.append(not bad)
        except RuntimeError:
            print(f"{victim}: DEADLOCKED — stale-master bug reproduced")
            verdicts.append(False)
            if "--no-park" not in sys.argv:
                park_retry("r")  # re-park before the next victim

    print("NO BUG" if all(verdicts) else "BUG REPRODUCED")
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
