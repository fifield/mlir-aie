#!/usr/bin/env python3
# dma_overhead_experiment.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --mode all --device npu2 --work-dir ./dma_overhead_build | FileCheck %s
# CHECK: PASS!

"""Measure the overhead of a core programming its own tile DMA BD chains.

See DMA_OVERHEAD_EXPERIMENT.md for the full methodology. In brief: the compute
core "pulls" M blocks of B int32 words off the input stream by programming the
S2MM (receive) buffer descriptors itself, with a *finite* repeat — never an
infinitely-looping chain — so the tile DMA is idle at the end. We compare three
ways of arming the receive BDs while holding the wait/compute/data identical:

  mode 0 (static)   : host runtime_sequence arms the BD chain (the "usual way")
  mode 1 (once)     : core arms the whole BD chain once, before its block loop
  mode 2 (per-iter) : core re-arms a fresh BD (chain depth D) every block

The core brackets each phase with reads of its own Timer_Low cycle counter and
emits a small telemetry buffer, so we get per-phase (program / wait / compute)
cycle counts plus an idle-at-end check (DMA_S2MM_Status_0 bit 19).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aie.dialects.aie import (
    AIEDevice,
    buffer,
    core,
    device,
    DMAChannelDir,
    flow,
    lock,
    tile,
    WireBundle,
)
from aie.dialects.aiex import (
    dma_await_task,
    dma_free_task,
    dma_start_task,
    npu_maskwrite32,
    npu_write32,
    runtime_sequence,
    shim_dma_single_bd_task,
)
from aie.extras.context import mlir_mod_ctx
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils.compile import compile_mlir_module
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.regdb import AIEAddressDecoder
import aie.iron as iron

from pythoc import ptr, i32
from pythoc.aie.operations import read_tm, write_tm

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "dma_overhead_build"

# Telemetry buffer size (int32 words). Layout documented in unpack_telemetry().
TELEM_N = 32

# ── Mode codes ───────────────────────────────────────────────────────────────
MODE_STATIC = 0  # host arms the BD chain (baseline "usual way")
MODE_ONCE = 1  # core arms the BD chain once before the loop
MODE_PERITER = 2  # core re-arms the BD(s) every block iteration
MODE_NAMES = {MODE_STATIC: "static", MODE_ONCE: "once", MODE_PERITER: "per-iter"}
MODE_CODES = {v: k for k, v in MODE_NAMES.items()}

# ── Register addresses from the register database ────────────────────────────

_decoder = AIEAddressDecoder()
_reg = _decoder.get_register_offset

DMA_BD_BASE = _reg("DMA_BD0_0", "memory")  # BD0 word0; stride 0x20 per BD
DMA_S2MM_0_START_QUEUE = _reg("DMA_S2MM_0_Start_Queue", "memory")
DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory")
DMA_S2MM_STATUS_0 = _reg("DMA_S2MM_Status_0", "memory")
LOCK0_VALUE = _reg("Lock0_value", "memory")
LOCK1_VALUE = _reg("Lock1_value", "memory")
TIMER_LOW = _reg("Timer_Low", "core")
CORE_PROCESSOR_BUS = _reg("Core_Processor_Bus", "core")

_REGDB_GLOBALS = {
    "DMA_BD_BASE": DMA_BD_BASE,
    "DMA_S2MM_0_START_QUEUE": DMA_S2MM_0_START_QUEUE,
    "DMA_MM2S_0_START_QUEUE": DMA_MM2S_0_START_QUEUE,
    "DMA_S2MM_STATUS_0": DMA_S2MM_STATUS_0,
    "LOCK0_VALUE": LOCK0_VALUE,
    "LOCK1_VALUE": LOCK1_VALUE,
    "TIMER_LOW": TIMER_LOW,
    "TELEM_N": TELEM_N,
    "MODE_STATIC": MODE_STATIC,
    "MODE_ONCE": MODE_ONCE,
    "MODE_PERITER": MODE_PERITER,
}


# ── PythoC kernel and helpers ─────────────────────────────────────────────────


@aie_kernel
def prog_s2mm_bd(
    bd_id: i32,
    base_words: i32,
    len_words: i32,
    next_bd: i32,
    use_next: i32,
    lock_rel: i32,
):
    """Program one S2MM buffer descriptor (6 words) via processor-bus writes.

    Args:
        bd_id:     BD index (0..15); selects the 0x20-strided register block
        base_words: buffer base address in 32-bit words
        len_words:  transfer length in 32-bit words
        next_bd:    BD to continue with when use_next != 0
        use_next:   1 to chain to next_bd, 0 to stop after this BD
        lock_rel:   1 to release lock 0 (+1) on completion, 0 for no release
    """
    bd: i32 = DMA_BD_BASE + bd_id * 32  # 0x20 stride between BDs

    # word0: [27:14] Base_Address (words), [13:0] Buffer_Length (words, actual)
    write_tm((base_words << 14) | len_words, bd)
    # words 1-4: contiguous 1D, no packet, no iteration stepping
    write_tm(0, bd + 4)
    write_tm(0, bd + 8)
    write_tm(0, bd + 12)
    write_tm(0, bd + 16)
    # word5: Valid_BD(25) | Use_Next_BD(26) | Next_BD(30:27)
    #        | Lock_Rel_Value(24:18)=lock_rel | Lock_Rel_ID(16:13)=0
    valid: i32 = 1 << 25
    rel: i32 = lock_rel << 18  # +1 release when lock_rel==1
    nxt: i32 = (use_next << 26) | (next_bd << 27)
    write_tm(valid | rel | nxt, bd + 20)


@aie_kernel
def dma_overhead_kernel(
    in_buf: ptr[i32, True],
    out_buf: ptr[i32, True],
    telem: ptr[i32, True],
    mode: i32,
    num_blocks: i32,
    block_words: i32,
    chain_depth: i32,
    sub_words: i32,
    in_addr_words: i32,
    telem_addr_words: i32,
):
    """Pull `num_blocks` blocks of `block_words` words and add `compute_passes`.

    The receive BDs are armed according to `mode`; the per-block wait + compute
    is identical across modes. Per-phase cycle counts go into `telem`.
    """
    # ── timer-read bias (cost of two back-to-back Timer_Low reads) ──────────
    bias_a: i32 = read_tm(TIMER_LOW)
    bias_b: i32 = read_tm(TIMER_LOW)
    telem[8] = bias_b - bias_a

    # ── one-time BD arming for `once` mode (core programs whole chain) ──────
    # `static` mode: host already armed + started the chain (do nothing here).
    # `once`/`per-iter`: chain is a sequence of `num_blocks` (resp. chain_depth)
    # BDs; here for `once` we lay out one BD per block, each releasing lock 0.
    prog_once_cycles: i32 = 0
    if mode == MODE_ONCE:
        t_p0: i32 = read_tm(TIMER_LOW)
        b: i32 = 0
        while b < num_blocks:
            last_use: i32 = 1
            if b == num_blocks - 1:
                last_use = 0
            prog_s2mm_bd(
                b,
                in_addr_words + b * block_words,
                block_words,
                b + 1,
                last_use,
                1,
            )
            b = b + 1
        # start the chain once: Start_BD_ID=0, Repeat_Count=0 (chain walks all)
        write_tm(0, DMA_S2MM_0_START_QUEUE)
        prog_once_cycles = read_tm(TIMER_LOW) - t_p0

    # ── per-block loop ──────────────────────────────────────────────────────
    program_cycles: i32 = prog_once_cycles
    wait_cycles: i32 = 0
    compute_cycles: i32 = 0
    prog_min: i32 = 0x7FFFFFFF
    prog_max: i32 = 0
    work: i32 = 1  # sequential scalar accumulator (compute-intensity load)

    t_start: i32 = read_tm(TIMER_LOW)

    i: i32 = 0
    while i < num_blocks:
        base_i: i32 = in_addr_words + i * block_words

        # ---- program phase (per-iter only) ----
        t0: i32 = read_tm(TIMER_LOW)
        if mode == MODE_PERITER:
            # Arm a fresh chain of `chain_depth` BDs covering this block.
            # Only the last BD releases lock 0, so exactly one release/block.
            j: i32 = 0
            while j < chain_depth:
                use_n: i32 = 1
                rel_j: i32 = 0
                if j == chain_depth - 1:
                    use_n = 0
                    rel_j = 1
                prog_s2mm_bd(j, base_i + j * sub_words, sub_words, j + 1, use_n, rel_j)
                j = j + 1
            # single-shot start: Start_BD_ID=0, Repeat_Count=0
            write_tm(0, DMA_S2MM_0_START_QUEUE)
        t1: i32 = read_tm(TIMER_LOW)

        # ---- wait phase: block i has landed when lock0 >= i+1 ----
        target: i32 = i + 1
        lk: i32 = read_tm(LOCK0_VALUE)
        while lk < target:
            lk = read_tm(LOCK0_VALUE)
        t2: i32 = read_tm(TIMER_LOW)

        # ---- compute phase ----
        # Verifiable output (out = in + 1) is a vector add (supported).
        # The compute-intensity load is a *sequential scalar* recurrence:
        # work = work*work + v carries a dependence across k (so the k-loop
        # does not vectorize, avoiding the unsupported <16 x i32> G_MUL) and
        # is non-affine in CPASS (so the optimizer keeps all CPASS scalar
        # multiplies). CPASS is a compile-time constant (see extra_globals).
        k: i32 = 0
        while k < block_words:
            v: i32 = in_buf[base_i - in_addr_words + k]
            out_buf[i * block_words + k] = v + 1
            p: i32 = 0
            while p < CPASS:
                work = work * work + v
                p = p + 1
            k = k + 1
        t3: i32 = read_tm(TIMER_LOW)

        dp: i32 = t1 - t0
        program_cycles = program_cycles + dp
        wait_cycles = wait_cycles + (t2 - t1)
        compute_cycles = compute_cycles + (t3 - t2)
        if mode == MODE_PERITER:
            if dp < prog_min:
                prog_min = dp
            if dp > prog_max:
                prog_max = dp
        i = i + 1

    # ── idle-at-end check: poll S2MM Channel_Running (bit 19) until 0 ───────
    t_idle0: i32 = read_tm(TIMER_LOW)
    st: i32 = read_tm(DMA_S2MM_STATUS_0)
    running: i32 = (st >> 19) & 1
    while running == 1:
        st = read_tm(DMA_S2MM_STATUS_0)
        running = (st >> 19) & 1
    t_end: i32 = read_tm(TIMER_LOW)

    # ── fill telemetry ──────────────────────────────────────────────────────
    telem[0] = t_end - t_start
    telem[1] = program_cycles
    telem[2] = wait_cycles
    telem[3] = compute_cycles
    telem[4] = prog_min
    telem[5] = prog_max
    telem[6] = t_end - t_idle0
    telem[7] = st  # final S2MM status (bit19 must be 0)
    telem[9] = num_blocks
    telem[10] = block_words
    telem[11] = mode
    telem[13] = chain_depth
    telem[14] = CPASS
    telem[16] = work  # observable, so the compute-intensity loop isn't DCE'd

    # ── epilogue: send [telem | results] back over MM2S ch0 (one BD) ────────
    # telem and results are contiguous in tile memory; one send covers both.
    # Measure the cost of programming this single send (constant across modes).
    t_e0: i32 = read_tm(TIMER_LOW)
    # MM2S BD uses bd_id 15 to avoid clobbering the S2MM chain BDs.
    # base = telem buffer address (in_addr_words is the input; telem is fixed
    # below the results, see host layout). We pass telem/out via their pointers
    # for compute, but program the send by absolute address words in telem[15].
    send_base: i32 = telem_addr_words  # telem buffer base in words
    send_len: i32 = TELEM_N + num_blocks * block_words
    bd15: i32 = DMA_BD_BASE + 15 * 32
    write_tm((send_base << 14) | send_len, bd15)
    write_tm(0, bd15 + 4)
    write_tm(0, bd15 + 8)
    write_tm(0, bd15 + 12)
    write_tm(0, bd15 + 16)
    # Valid(25) | Lock_Rel_Value(24:18)=+1 | Lock_Rel_ID(16:13)=1
    write_tm((1 << 25) | (1 << 18) | (1 << 13), bd15 + 20)
    telem[12] = read_tm(TIMER_LOW) - t_e0
    # rewrite telem[12] into the buffer is already done; start the send.
    write_tm(15, DMA_MM2S_0_START_QUEUE)  # Start_BD_ID=15
    # wait for MM2S completion (lock1 released by the send BD)
    done: i32 = 0
    while done == 0:
        done = read_tm(LOCK1_VALUE)


# ── Telemetry unpacking ────────────────────────────────────────────────────


def unpack_telemetry(telem):
    """Decode the telemetry int32 array into a dict."""
    t = [int(x) for x in telem]
    return {
        "total_cycles": t[0],
        "program_cycles": t[1],
        "wait_cycles": t[2],
        "compute_cycles": t[3],
        "prog_min": t[4] if t[4] != 0x7FFFFFFF else 0,
        "prog_max": t[5],
        "idle_drain_cycles": t[6],
        "s2mm_status": t[7],
        "timer_bias": t[8],
        "num_blocks": t[9],
        "block_words": t[10],
        "mode": t[11],
        "epilogue_prog_cycles": t[12],
        "chain_depth": t[13],
        "compute_passes": t[14],
        "channel_idle": ((t[7] >> 19) & 1) == 0,
    }


# ── Design construction ──────────────────────────────────────────────────────


def _mem_layout(num_blocks, block_words):
    """Return byte addresses for telem, results, in_buf (contiguous telem|res)."""
    mb = num_blocks * block_words
    telem_addr = 0x1000
    results_addr = telem_addr + TELEM_N * 4
    in_addr = results_addr + mb * 4
    return telem_addr, results_addr, in_addr, mb


def build_mlir_module(dev, kernel, mode, num_blocks, block_words, chain_depth):
    mb = num_blocks * block_words
    telem_addr, results_addr, in_addr, _ = _mem_layout(num_blocks, block_words)

    in_ty = np.ndarray[(mb,), np.dtype[np.int32]]
    out_all_ty = np.ndarray[(TELEM_N + mb,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            kernel.resolve()

            t00 = tile(0, 0)
            t02 = tile(0, 2)

            telem_buf = buffer(
                t02, datatype=np.ndarray[(TELEM_N,), np.dtype[np.int32]],
                name="telem", address=telem_addr,
            )
            results_buf = buffer(
                t02, datatype=np.ndarray[(mb,), np.dtype[np.int32]],
                name="results", address=results_addr,
            )
            in_buf = buffer(t02, datatype=in_ty, name="in_buf", address=in_addr)

            lock(t02, lock_id=0, init=0, sym_name="s2mm_done")
            lock(t02, lock_id=1, init=0, sym_name="mm2s_done")

            flow(t00, WireBundle.DMA, 0, t02, WireBundle.DMA, 0)  # input stream
            flow(t02, WireBundle.DMA, 0, t00, WireBundle.DMA, 0)  # output stream

            from aie.dialects.aie import shim_dma_allocation

            shim_dma_allocation("in_alloc", t00, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("out_alloc", t00, DMAChannelDir.S2MM, 0)

            @core(t02)
            def core_body():
                kernel(
                    in_buf,
                    results_buf,
                    telem_buf,
                    mode,
                    num_blocks,
                    block_words,
                    chain_depth,
                    block_words // chain_depth,
                    in_addr // 4,
                    telem_addr // 4,
                )

            @runtime_sequence(in_ty, out_all_ty)
            def sequence(A, C):
                # Enable processor bus so the core's write_tm reaches the DMA.
                npu_maskwrite32(
                    address=CORE_PROCESSOR_BUS, value=0x1, mask=0x1,
                    column=0, row=2,
                )
                # static mode: host arms the S2MM BD chain (one BD per block).
                if mode == MODE_STATIC:
                    for b in range(num_blocks):
                        bd_base = DMA_BD_BASE + b * 32
                        base_w = (in_addr // 4) + b * block_words
                        npu_write32(address=bd_base + 0,
                                    value=(base_w << 14) | block_words,
                                    column=0, row=2)
                        npu_write32(address=bd_base + 4, value=0, column=0, row=2)
                        npu_write32(address=bd_base + 8, value=0, column=0, row=2)
                        npu_write32(address=bd_base + 12, value=0, column=0, row=2)
                        npu_write32(address=bd_base + 16, value=0, column=0, row=2)
                        use_next = 1 if b < num_blocks - 1 else 0
                        w5 = (1 << 25) | (1 << 18) | (use_next << 26) | ((b + 1) << 27)
                        npu_write32(address=bd_base + 20, value=w5, column=0, row=2)
                    # start the chain: Start_BD_ID=0, Repeat_Count=0
                    npu_write32(address=DMA_S2MM_0_START_QUEUE, value=0,
                                column=0, row=2)

                in_task = shim_dma_single_bd_task("in_alloc", A, sizes=[1, 1, 1, mb])
                out_task = shim_dma_single_bd_task(
                    "out_alloc", C, sizes=[1, 1, 1, TELEM_N + mb], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)

        if not ctx.module.operation.verify():
            raise RuntimeError("Generated MLIR failed verification")
        return ctx.module


# ── Compile & Run ─────────────────────────────────────────────────────────────


def build_and_run(dev, target_arch, mode, num_blocks, block_words, chain_depth,
                  compute_passes, work_dir, verbose):
    mb = num_blocks * block_words
    in_ty = np.ndarray[(mb,), np.dtype[np.int32]]
    out_all_ty = np.ndarray[(TELEM_N + mb,), np.dtype[np.int32]]

    kernel = PythocKernel(
        dma_overhead_kernel,
        [in_ty, np.ndarray[(mb,), np.dtype[np.int32]],
         np.ndarray[(TELEM_N,), np.dtype[np.int32]],
         np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
        target_arch=target_arch,
        extra_globals={**_REGDB_GLOBALS, "CPASS": compute_passes},
        helpers=[prog_s2mm_bd],
    )

    module = build_mlir_module(
        dev, kernel, mode, num_blocks, block_words, chain_depth
    )
    tag = f"{MODE_NAMES[mode]}_M{num_blocks}_B{block_words}_D{chain_depth}_C{compute_passes}"
    sub = work_dir / tag
    sub.mkdir(parents=True, exist_ok=True)
    with open(sub / "design.mlir", "w") as f:
        print(module, file=f)

    insts_path = sub / "insts.bin"
    xclbin_path = sub / "final.xclbin"
    compile_mlir_module(
        mlir_module=module, insts_path=str(insts_path),
        xclbin_path=str(xclbin_path), work_dir=str(sub), verbose=verbose,
    )

    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    in_data = np.arange(1, mb + 1, dtype=np.int32)
    in_t = iron.tensor(in_data, dtype=np.int32)
    out_t = iron.zeros(TELEM_N + mb, dtype=np.int32)
    DefaultNPURuntime.run(handle, [in_t, out_t])

    out_all = np.array(out_t.numpy())
    telem = out_all[:TELEM_N]
    results = out_all[TELEM_N:]
    # Output is in + 1 regardless of compute_passes (the compute-intensity
    # loop only adds scalar work; it does not change the result).
    expected = (in_data + 1).astype(np.int32)
    ok = bool(np.array_equal(results, expected))
    info = unpack_telemetry(telem)
    info["correct"] = ok
    return info


# ── Reporting ──────────────────────────────────────────────────────────────


def print_result(info):
    bias = info["timer_bias"]
    pb = info["program_cycles"] / max(1, info["num_blocks"])
    print(
        f"  {MODE_NAMES[info['mode']]:8s} "
        f"M={info['num_blocks']:3d} B={info['block_words']:5d} "
        f"D={info['chain_depth']} C={info['compute_passes']:3d} | "
        f"total={info['total_cycles']:8d}  prog={info['program_cycles']:7d} "
        f"(/blk={pb:7.1f})  wait={info['wait_cycles']:8d}  "
        f"compute={info['compute_cycles']:8d}  "
        f"idle={'Y' if info['channel_idle'] else 'N'}  "
        f"{'OK' if info['correct'] else 'FAIL'}  (bias={bias})"
    )


def summarize_sweep(results):
    """Print a per-axis comparison of the three modes.

    The clean overhead signal is program_cycles/block (pure core cost). wait
    and total also include DMA/stream timing (spin-wait that overlaps the
    transfer), so they are noisier; program_cycles and compute_cycles are
    cycle-stable. We report per-iter overhead as a fraction of total runtime.
    """
    print("\n" + "=" * 78)
    print("SUMMARY  (prog/blk = pure core programming cost; ovhd = per-iter prog/total)")
    print("=" * 78)
    by_axis = {}
    for r in results:
        by_axis.setdefault(r.get("axis", "single"), []).append(r)
    for axis, rows in by_axis.items():
        print(f"\n[{axis}]")
        # group rows by config (M,B,D,C), each having up to 3 modes
        cfgs = {}
        for r in rows:
            key = (r["num_blocks"], r["block_words"], r["chain_depth"], r["compute_passes"])
            cfgs.setdefault(key, {})[r["mode"]] = r
        hdr = (f"  {'M':>3} {'B':>5} {'D':>2} {'C':>3} | "
               f"{'static/blk':>10} {'once/blk':>9} {'periter/blk':>11} | "
               f"{'periter ovhd':>12}")
        print(hdr)
        for key in sorted(cfgs):
            M, B, D, C = key
            modes = cfgs[key]
            def pb(m):
                return modes[m]["program_cycles"] / max(1, M) if m in modes else float("nan")
            pi = modes.get(MODE_PERITER)
            ovhd = (pi["program_cycles"] / max(1, pi["total_cycles"])) if pi else float("nan")
            print(f"  {M:>3} {B:>5} {D:>2} {C:>3} | "
                  f"{pb(MODE_STATIC):>10.1f} {pb(MODE_ONCE):>9.1f} "
                  f"{pb(MODE_PERITER):>11.1f} | {ovhd*100:>11.2f}%")


# ── Sweep configurations ─────────────────────────────────────────────────────


def sweep_configs():
    """Yield (axis, M, B, D, C) tuples. M*B kept <= 4096 words (memory)."""
    cfgs = []
    # Axis 1: block size B (fixed M=4, D=1, C=1)
    for B in (64, 128, 256, 512, 1024):
        cfgs.append(("block_size", 4, B, 1, 1))
    # Axis 2: number of blocks M (fixed B=128, D=1, C=1)
    for M in (2, 4, 8, 16):
        cfgs.append(("num_blocks", M, 128, 1, 1))
    # Axis 3: BD-chain depth D (fixed M=4, B=256, C=1); B must divide by D
    for D in (1, 2, 4):
        cfgs.append(("chain_depth", 4, 256, D, 1))
    # Axis 4: compute intensity C (fixed M=4, B=256, D=1)
    for C in (1, 4, 16, 64):
        cfgs.append(("compute", 4, 256, 1, C))
    return cfgs


def parse_args():
    p = argparse.ArgumentParser(description="Core-programmed DMA overhead experiment")
    p.add_argument("--device", choices=("npu", "npu1", "npu2"), default="npu2")
    p.add_argument("--mode", choices=("static", "once", "per-iter", "all"), default="all")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--block-words", type=int, default=256)
    p.add_argument("--chain-depth", type=int, default=1)
    p.add_argument("--compute-passes", type=int, default=1)
    p.add_argument("--sweep", action="store_true", help="Run the full sweep")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def pick_device(name):
    if name.lower() == "npu2":
        return AIEDevice.npu2, "aie2p"
    return AIEDevice.npu1_1col, "aie2"


def modes_for(arg):
    if arg == "all":
        return [MODE_STATIC, MODE_ONCE, MODE_PERITER]
    return [MODE_CODES[arg]]


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dev, target_arch = pick_device(args.device)
    modes = modes_for(args.mode)

    all_ok = True
    results = []
    try:
        if args.sweep:
            for axis, M, B, D, C in sweep_configs():
                print(f"\n[{axis}] M={M} B={B} D={D} C={C}")
                for m in modes:
                    info = build_and_run(dev, target_arch, m, M, B, D, C,
                                         work_dir, args.verbose)
                    info["axis"] = axis
                    print_result(info)
                    results.append(info)
                    all_ok = all_ok and info["correct"] and info["channel_idle"]
            summarize_sweep(results)
        else:
            print(f"\nM={args.num_blocks} B={args.block_words} "
                  f"D={args.chain_depth} C={args.compute_passes}")
            for m in modes:
                info = build_and_run(dev, target_arch, m, args.num_blocks,
                                     args.block_words, args.chain_depth,
                                     args.compute_passes, work_dir, args.verbose)
                print_result(info)
                results.append(info)
                all_ok = all_ok and info["correct"] and info["channel_idle"]

        if all_ok:
            print("\nPASS!")
            return 0
        print("\nFAILED: some runs incorrect or channel not idle")
        return 1
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
