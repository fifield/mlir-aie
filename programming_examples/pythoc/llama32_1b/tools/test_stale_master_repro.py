#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Minimal repro: stale switch state from a production PDI deadlocks the next
tiny device (RESIDENT_DEVICE_EVOLUTION.md C2 blocker, commit ff1b778f0).

Park: production rms_gemv_rope ELF (decode_kernel_cache) — broad shim E/W
circuit broadcast tree. Victim: a kernel-less 8-column passthrough (shim0
MM2S1 broadcast -> per-col compute tile -> shim S2MM0, 256 bf16 per column)
in either routing mode:

  victim_pkt      packet broadcast / packet results (C2-style)
  victim_circuit  circuit broadcast (default-kernel-style)

Both victims complete on a blank device (run with --no-park to check), both
DEADLOCK after rgr: LoadPDI only writes the ports the new device uses, so
rgr's parked circuit masters keep forking the victim stream into disarmed
destinations. Production kernels survive only because their route trees
happen to overwrite rgr's ports. rgr doubles as the wedge sacrifice (it runs
from any device state).

Usage: python3 tools/test_stale_master_repro.py [--compile-only] [--no-park]
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


def main() -> int:
    os.chdir(PROJECT_DIR / "build_peano")
    from kernel_builder.cache import KernelCache

    cache = KernelCache(cache_dir=PROJECT_DIR / "build_peano" / "stale_master_cache",
                        verbose=False)
    if not (cache.load_manifest() and "victim_pkt" in cache.artifacts
            and "victim_circuit" in cache.artifacts):
        cache.compile_and_cache("victim_pkt", build_module(True),
                                instance_name="victim_pkt")
        cache.compile_and_cache("victim_circuit", build_module(False),
                                instance_name="victim_circuit")
        cache._save_manifest()
        print("compiled", flush=True)
    if "--compile-only" in sys.argv:
        return 0

    x = (np.arange(N) % 251 + 1).astype(bfloat16)
    ys = [np.zeros(N, dtype=bfloat16) for _ in range(N_COLS)]
    out_idx = list(range(1, 1 + N_COLS))

    EMB, KV = 2048, 512
    z = lambda n: np.zeros(n, dtype=bfloat16)
    rgr_args = [z(EMB), z(EMB), z(EMB), z((EMB, EMB)), z(EMB), z((KV, EMB)),
                z(KV), z((KV, EMB)), z(KV), z(64), z(64), z(EMB), z(KV)]
    dc = KernelCache(cache_dir=PROJECT_DIR / "build_peano" / "decode_kernel_cache",
                     verbose=False)
    if not dc.load_manifest():
        print("decode_kernel_cache missing — run `make compile` first")
        return 2

    def park_retry(tag):
        for i in range(3):
            try:
                dc.load_and_run("rms_gemv_rope", None, *rgr_args,
                                output_indices=[11, 12], bo_key=f"{tag}{i}")
                return True
            except RuntimeError:
                print(f"rgr {tag}{i} consumed a wedge", flush=True)
        return False

    if "--no-park" not in sys.argv:
        # rgr doubles as the sacrifice: it runs from any device state.
        if not park_retry("s"):
            print("device unusable (rgr never ran) — inconclusive")
            return 2
        print("rgr ok — production circuit tree parked", flush=True)

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
