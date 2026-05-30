#!/usr/bin/env python3
# memtile_program_cost.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cost of programming a memtile DMA: core (control packets) vs CDO vs host.

Companion to ../dma_overhead (which measured a core arming its OWN tile DMA).
Here we measure the runtime cost of getting an *idle memtile's* MM2S DMA to
stream N int32 words to the compute core, three ways:

  core : the core programs+starts the memtile MM2S BD via control packets
         (reset/unreset -> BD -> enable -> start-queue push). The buffer length
         is baked as a compile-time constant (MEM_N) to avoid the Peano store
         scheduling bug; see ../ctrl_packet_dma/issue_peano_dma_sched.
  cdo  : a static aie.memtile_dma configures the channel at PDI/CDO time; the
         core's runtime programming cost is ~0 (paid once at load).
  host : the runtime_sequence programs the memtile BD+start via npu_write32
         (shim MMIO) before the core runs; core programming cost is ~0.

The compute core brackets two phases with its own Timer_Low counter:
  program_cyc  = cycles the core spends issuing the control-packet program+start
                 (core mode only; ~bias for cdo/host).
  transfer_cyc = cycles from "armed" to "N words received" (the DMA itself).

Telemetry is returned to the host. Sweeping N shows the fixed programming
overhead vs the size-dependent transfer, i.e. the amortization point.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import aie.dialects.aiex as aiex
import aie.iron as iron
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    EndOp,
    LockAction,
    WireBundle,
    buffer,
    core,
    device,
    dma_bd,
    dma_start,
    flow,
    lock,
    memtile_dma,
    next_bd,
    packetflow,
    shim_dma_allocation,
    tile,
    use_lock,
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
from aie.iron.pythoc import PythocKernel, aie_kernel
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module
from aie.utils.regdb import AIEAddressDecoder
from pythoc import i32, ptr
from pythoc.aie.operations import read_tm, write_tm

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "build"

TELEM_N = 16
CTRL_WORDS = 23  # single-transfer layout: 6 commands + 5 embedded NoC headers
CTRL_PACKET_ID = 5
CTRL_PACKET_TYPE = 0


def _noc_ctrl_header():
    """On-stream NoC/stream packet header for the control flow (cf.
    AIETranslateControlPacketsToUI32Vec in lib/Targets/AIETargetNPU.cpp):
    hdr = (pktType & 0x7)<<12 | (pktId & 0xff), odd-parity bit at bit31.

    The core DMA inserts this header for the *first* packet of a transfer only;
    when all control commands are sent as one transfer, commands 2..N must carry
    this header word themselves.
    """
    hdr = ((CTRL_PACKET_TYPE & 0x7) << 12) | (CTRL_PACKET_ID & 0xFF)
    n, ones = hdr, 0
    while n:
        ones += n & 1
        n >>= 1
    pb = 1 if (ones % 2) == 0 else 0
    return (hdr | (pb << 31)) & 0xFFFFFFFF


NOC_CTRL_HDR = _noc_ctrl_header()              # 0x80000005 for id=5,type=0
NOC_CTRL_HDR_S = NOC_CTRL_HDR - (1 << 32)      # signed i32 form for kernel globals

CORE_CTRL_ADDR = 0x1000   # ctrl_buf (up to 8+10*DEPTH words)
CORE_TELEM_ADDR = 0x1400  # telem (16 words) — clear of the deepest ctrl_buf
CORE_IN_ADDR = 0x2000     # received data (N or DEPTH*N words)
MEM_SRC_ADDR = 0x1000     # memtile source (N words)

_decoder = AIEAddressDecoder()
_reg = _decoder.get_register_offset

DMA_BD_BASE = _reg("DMA_BD0_0", "memory")
DMA_S2MM_1_START_QUEUE = _reg("DMA_S2MM_1_Start_Queue", "memory")
DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory")
DMA_MM2S_1_START_QUEUE = _reg("DMA_MM2S_1_Start_Queue", "memory")
LOCK0_VALUE = _reg("Lock0_value", "memory")
LOCK1_VALUE = _reg("Lock1_value", "memory")
LOCK2_VALUE = _reg("Lock2_value", "memory")
TIMER_LOW = _reg("Timer_Low", "core")
CORE_PROCESSOR_BUS = _reg("Core_Processor_Bus", "core")

MEMTILE_DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory_tile")
MEMTILE_DMA_MM2S_0_CTRL = _reg("DMA_MM2S_0_Ctrl", "memory_tile")
MEMTILE_DMA_BD_BASE = _reg("DMA_BD0_0", "memory_tile")
MEMTILE_OWN = 0x20000


def _globals(mem_n, depth=1, nchan=1):
    return {
        "DEPTH": depth,
        "NCHAN": nchan,
        "MEMTILE_DMA_MM2S_1_CTRL": _reg("DMA_MM2S_1_Ctrl", "memory_tile"),
        "MEMTILE_DMA_MM2S_1_START_QUEUE": _reg("DMA_MM2S_1_Start_Queue", "memory_tile"),
        # memtile odd DMA channels (MM2S1/S2MM1) may use BD ID [24-47] only
        # (AIETargetModel isBdChannelAccessible); MM2S1 -> BD24.
        "MEMTILE_DMA_BD1_BASE": _reg("DMA_BD0_0", "memory_tile") + 24 * 32,
        "DMA_S2MM_0_START_QUEUE": _reg("DMA_S2MM_0_Start_Queue", "memory"),
        "CORE_IN2_ADDR_WORDS": (CORE_IN_ADDR // 4) + (mem_n * nchan),
        "DMA_BD_BASE": DMA_BD_BASE,
        "DMA_S2MM_1_START_QUEUE": DMA_S2MM_1_START_QUEUE,
        "DMA_MM2S_0_START_QUEUE": DMA_MM2S_0_START_QUEUE,
        "DMA_MM2S_1_START_QUEUE": DMA_MM2S_1_START_QUEUE,
        "LOCK0_VALUE": LOCK0_VALUE,
        "LOCK1_VALUE": LOCK1_VALUE,
        "LOCK2_VALUE": LOCK2_VALUE,
        "TIMER_LOW": TIMER_LOW,
        "MEMTILE_DMA_MM2S_0_START_QUEUE": MEMTILE_DMA_MM2S_0_START_QUEUE,
        "MEMTILE_DMA_MM2S_0_CTRL": MEMTILE_DMA_MM2S_0_CTRL,
        "MEMTILE_DMA_BD_BASE": MEMTILE_DMA_BD_BASE,
        "MEMTILE_OWN": MEMTILE_OWN,
        "MEM_SRC_ADDR": MEM_SRC_ADDR,
        "CTRL_PACKET_ID": CTRL_PACKET_ID,
        "CTRL_PACKET_TYPE": CTRL_PACKET_TYPE,
        "NOC_CTRL_HDR": NOC_CTRL_HDR_S,
        "TELEM_ADDR_WORDS": CORE_TELEM_ADDR // 4,
        "TELEM_N": TELEM_N,
        "MEM_N": mem_n,  # baked constant -> constant BD-length store (workaround)
    }


def py_packet_header(address, beats):
    """Host-side mirror of make_write_packet_header (odd-parity ctrl header)."""
    raw = ((beats << 20) | address) & 0xFFFFFFFF
    n, ones = raw, 0
    while n:
        ones += n & 1
        n >>= 1
    pb = 1 if (ones & 1) == 0 else 0
    return (raw | (pb << 31)) & 0xFFFFFFFF


def ctrl_init_words(mem_n):
    """The 23-word single-channel memtile MM2S program, computed on the host.

    The 6 control commands are concatenated into ONE control-packet transfer.
    The core DMA inserts the NoC stream header only for the first command, so
    commands 2..6 carry an explicit `NOC_CTRL_HDR` word (see _noc_ctrl_header).
    Baked into a statically-initialized ctrl_buf so the core never *writes* it at
    runtime -> no data-memory store for the launched DMA to race against.
    """
    bd = MEMTILE_DMA_BD_BASE
    ctl = MEMTILE_DMA_MM2S_0_CTRL
    sq = MEMTILE_DMA_MM2S_0_START_QUEUE
    base = (MEM_SRC_ADDR >> 2) + MEMTILE_OWN
    h = NOC_CTRL_HDR
    w = [
        py_packet_header(ctl, 0), 2,                        # cmd0 reset (DMA adds NoC hdr)
        h, py_packet_header(ctl, 0), 0,                     # cmd1 unreset
        h, py_packet_header(bd, 3), mem_n, base, 0, 0,      # cmd2 BD words 0..3
        h, py_packet_header(bd + 16, 3), 0, 0, 0, 1 << 31,  # cmd3 BD words 4..7 (valid)
        h, py_packet_header(ctl, 0), 1,                     # cmd4 enable
        h, py_packet_header(sq, 0), 0,                      # cmd5 start-queue push (BD0)
    ]
    return np.array([x if x < 2**31 else x - 2**32 for x in w], dtype=np.int32)


# ── shared helpers ──────────────────────────────────────────────────────────


@aie_kernel
def odd_parity_header(raw: i32) -> i32:
    n: i32 = raw
    ones: i32 = 0
    while n != 0:
        ones = ones + (n & 1)
        n = n >> 1
    parity_bit: i32 = 0
    if (ones & 1) == 0:
        parity_bit = 1
    return raw | (parity_bit << 31)


@aie_kernel
def make_write_packet_header(address: i32, beats: i32) -> i32:
    raw: i32 = (beats << 20) | address
    return odd_parity_header(raw)


@aie_kernel
def program_dma_and_start(
    bd_id: i32, start_queue_addr: i32, base_addr_words: i32, length: i32, lock_rel_id: i32
):
    bd: i32 = DMA_BD_BASE + (bd_id * 32)
    write_tm((base_addr_words << 14) | length, bd)
    write_tm(0, bd + 4)
    write_tm(0, bd + 8)
    write_tm(0, bd + 12)
    write_tm(0, bd + 16)
    write_tm((1 << 25) | (1 << 18) | (lock_rel_id << 13), bd + 20)
    write_tm(bd_id, start_queue_addr)


@aie_kernel
def program_packet_mm2s(base_addr_words: i32, num_words: i32):
    bd: i32 = DMA_BD_BASE
    write_tm((base_addr_words << 14) | num_words, bd)
    write_tm((1 << 30) | (CTRL_PACKET_ID << 19) | (CTRL_PACKET_TYPE << 16), bd + 4)
    write_tm(0, bd + 8)
    write_tm(0, bd + 12)
    write_tm(0, bd + 16)
    write_tm((1 << 25) | (1 << 18) | (1 << 13), bd + 20)
    write_tm(0, DMA_MM2S_0_START_QUEUE)


@aie_kernel
def spin_lock_ge(lock_addr: i32, target: i32):
    v: i32 = read_tm(lock_addr)
    guard: i32 = 0
    while v < target and guard < 4000000:
        v = read_tm(lock_addr)
        guard = guard + 1


@aie_kernel
def spin_lock_ge_bounded(lock_addr: i32, target: i32) -> i32:
    v: i32 = read_tm(lock_addr)
    guard: i32 = 0
    while v < target and guard < 2000000:
        v = read_tm(lock_addr)
        guard = guard + 1
    return v


# ── core mode: program the memtile MM2S via control packets, timed ───────────


@aie_kernel
def bench_core(
    ctrl_buf: ptr[i32, True],
    in_buf: ptr[i32, True],
    telem: ptr[i32, True],
    ctrl_addr_words: i32,
    in_addr_words: i32,
):
    ba: i32 = read_tm(TIMER_LOW)
    bb: i32 = read_tm(TIMER_LOW)

    # Arm the receive path (core S2MM1, releases core lock0 on completion).
    program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_addr_words, MEM_N, 0)

    t0: i32 = read_tm(TIMER_LOW)
    # Build the 23-word single-transfer control program for the memtile MM2S0:
    # 6 commands concatenated, with the NoC stream header (NOC_CTRL_HDR) baked in
    # front of commands 2..6 (the core DMA only inserts the header for the first
    # command of the transfer; see _noc_ctrl_header / AIETargetNPU.cpp). Length is
    # baked as the MEM_N constant -> constant store; see issue_peano_dma_sched.
    tf: i32 = read_tm(TIMER_LOW)
    ctrl_buf[0] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[1] = 2  # cmd0 reset (DMA prepends NoC header)
    ctrl_buf[2] = NOC_CTRL_HDR
    ctrl_buf[3] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[4] = 0  # cmd1 unreset
    ctrl_buf[5] = NOC_CTRL_HDR
    ctrl_buf[6] = make_write_packet_header(MEMTILE_DMA_BD_BASE, 3)
    ctrl_buf[7] = MEM_N
    ctrl_buf[8] = (MEM_SRC_ADDR >> 2) + MEMTILE_OWN
    ctrl_buf[9] = 0
    ctrl_buf[10] = 0  # cmd2 BD words 0..3
    ctrl_buf[11] = NOC_CTRL_HDR
    ctrl_buf[12] = make_write_packet_header(MEMTILE_DMA_BD_BASE + 16, 3)
    ctrl_buf[13] = 0
    ctrl_buf[14] = 0
    ctrl_buf[15] = 0
    ctrl_buf[16] = 1 << 31  # cmd3 BD words 4..7 (valid)
    ctrl_buf[17] = NOC_CTRL_HDR
    ctrl_buf[18] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[19] = 1  # cmd4 enable
    ctrl_buf[20] = NOC_CTRL_HDR
    ctrl_buf[21] = make_write_packet_header(MEMTILE_DMA_MM2S_0_START_QUEUE, 0)
    ctrl_buf[22] = 0  # cmd5 start-queue push (BD 0)

    # ── program phase: all 6 commands in ONE transfer (one MM2S0 completion) ──
    program_packet_mm2s(ctrl_addr_words + 0, 23)
    spin_lock_ge(LOCK1_VALUE, 1)
    t1: i32 = read_tm(TIMER_LOW)

    # ── transfer phase: wait for all MEM_N words to land (core lock0) ──
    spin_lock_ge(LOCK0_VALUE, 1)
    t2: i32 = read_tm(TIMER_LOW)

    telem[0] = bb - ba          # timer-read bias
    telem[1] = t1 - t0          # program cycles
    telem[2] = t2 - t1          # transfer cycles
    telem[3] = MEM_N
    telem[4] = 0                # mode = core
    telem[5] = in_buf[0]        # sanity: first received word
    telem[6] = in_buf[MEM_N - 1]
    telem[7] = t0 - tf          # ctrl_buf fill cost (core builds the packets)

    # Return telem to the host (core MM2S1 -> shim), release lock1 -> 7.
    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, TELEM_N, 1)
    spin_lock_ge(LOCK1_VALUE, 7)


# ── core_static: ctrl_buf is statically initialized; core only *sends* it ─────


@aie_kernel
def bench_core_static(
    ctrl_buf: ptr[i32, True],
    in_buf: ptr[i32, True],
    telem: ptr[i32, True],
    ctrl_addr_words: i32,
    in_addr_words: i32,
    num_words: i32,
):
    ba: i32 = read_tm(TIMER_LOW)
    bb: i32 = read_tm(TIMER_LOW)

    program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_addr_words, num_words, 0)

    # NOTE: no stores to ctrl_buf -- it was initialized in the PDI. The core only
    # streams the pre-baked control packets, so there is no runtime data-memory
    # store for the launched MM2S to race against (the write_tm barrier issue).
    # All 6 commands ship as ONE control-packet transfer (23 words, with the
    # NoC headers for commands 2..6 baked into the buffer); one transfer => one
    # core-MM2S0 completion => lock1 advances by 1.
    t0: i32 = read_tm(TIMER_LOW)
    program_packet_mm2s(ctrl_addr_words + 0, 23)
    spin_lock_ge(LOCK1_VALUE, 1)
    t1: i32 = read_tm(TIMER_LOW)

    spin_lock_ge(LOCK0_VALUE, 1)
    t2: i32 = read_tm(TIMER_LOW)

    telem[0] = bb - ba
    telem[1] = t1 - t0
    telem[2] = t2 - t1
    telem[3] = num_words
    telem[4] = 10
    telem[5] = in_buf[0]
    telem[6] = in_buf[num_words - 1]

    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, TELEM_N, 1)
    spin_lock_ge(LOCK1_VALUE, 7)


# ── chain-depth mode: core programs a DEPTH-deep memtile MM2S BD chain ───────


@aie_kernel
def bench_core_chain(
    ctrl_buf: ptr[i32, True],
    in_buf: ptr[i32, True],
    telem: ptr[i32, True],
    ctrl_addr_words: i32,
    in_addr_words: i32,
):
    ba: i32 = read_tm(TIMER_LOW)
    bb: i32 = read_tm(TIMER_LOW)

    # One core S2MM BD receives the whole DEPTH*MEM_N chain output.
    program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_addr_words, DEPTH * MEM_N, 0)

    # Build a DEPTH-deep chain: BD_i -> BD_{i+1} (use_next), last -> end. Sent as
    # lock-synced separate control packets: a single concatenated transfer does
    # NOT deliver when the head BD has use_next set (chained-BD writes need the
    # paced/committed path), unlike the independent single/multi-channel cases.
    ctrl_buf[0] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[1] = 2
    ctrl_buf[2] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[3] = 0
    i: i32 = 0
    while i < DEPTH:
        off: i32 = 4 + i * 10
        bd_addr: i32 = MEMTILE_DMA_BD_BASE + i * 32
        word1: i32 = (MEM_SRC_ADDR >> 2) + MEMTILE_OWN
        if i < DEPTH - 1:
            word1 = word1 | (1 << 19) | ((i + 1) << 20)
        ctrl_buf[off + 0] = make_write_packet_header(bd_addr, 3)
        ctrl_buf[off + 1] = MEM_N
        ctrl_buf[off + 2] = word1
        ctrl_buf[off + 3] = 0
        ctrl_buf[off + 4] = 0
        ctrl_buf[off + 5] = make_write_packet_header(bd_addr + 16, 3)
        ctrl_buf[off + 6] = 0
        ctrl_buf[off + 7] = 0
        ctrl_buf[off + 8] = 0
        ctrl_buf[off + 9] = 1 << 31
        i = i + 1
    e_off: i32 = 4 + DEPTH * 10
    ctrl_buf[e_off + 0] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[e_off + 1] = 1
    ctrl_buf[e_off + 2] = make_write_packet_header(MEMTILE_DMA_MM2S_0_START_QUEUE, 0)
    ctrl_buf[e_off + 3] = 0

    # ── program phase ──
    t0: i32 = read_tm(TIMER_LOW)
    program_packet_mm2s(ctrl_addr_words + 0, 2)
    spin_lock_ge(LOCK1_VALUE, 1)
    program_packet_mm2s(ctrl_addr_words + 2, 2)
    spin_lock_ge(LOCK1_VALUE, 2)
    sent: i32 = 2
    i = 0
    while i < DEPTH:
        off2: i32 = 4 + i * 10
        program_packet_mm2s(ctrl_addr_words + off2, 5)
        sent = sent + 1
        spin_lock_ge(LOCK1_VALUE, sent)
        program_packet_mm2s(ctrl_addr_words + off2 + 5, 5)
        sent = sent + 1
        spin_lock_ge(LOCK1_VALUE, sent)
        i = i + 1
    program_packet_mm2s(ctrl_addr_words + e_off, 2)
    sent = sent + 1
    spin_lock_ge(LOCK1_VALUE, sent)
    program_packet_mm2s(ctrl_addr_words + e_off + 2, 2)
    sent = sent + 1
    spin_lock_ge(LOCK1_VALUE, sent)
    t1: i32 = read_tm(TIMER_LOW)

    spin_lock_ge(LOCK0_VALUE, 1)
    t2: i32 = read_tm(TIMER_LOW)

    telem[0] = bb - ba
    telem[1] = t1 - t0
    telem[2] = t2 - t1
    telem[3] = MEM_N
    telem[4] = DEPTH
    telem[5] = in_buf[0]
    telem[6] = in_buf[DEPTH * MEM_N - 1]

    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, TELEM_N, 1)
    spin_lock_ge(LOCK1_VALUE, sent + 1)


# ── multi-channel mode: core programs 2 memtile MM2S channels in parallel ────


@aie_kernel
def bench_core_2ch(
    ctrl_buf: ptr[i32, True],
    in_buf: ptr[i32, True],
    in_buf2: ptr[i32, True],
    telem: ptr[i32, True],
    ctrl_addr_words: i32,
    in_addr_words: i32,
    in2_addr_words: i32,
):
    ba: i32 = read_tm(TIMER_LOW)
    bb: i32 = read_tm(TIMER_LOW)

    # Two receive channels: S2MM1 (-> lock0) and S2MM0 (-> lock2).
    # S2MM0 uses BD3 (not BD0) so it doesn't clash with the control-packet
    # sender's MM2S0 BD0 (program_packet_mm2s reprograms BD0 on every send).
    program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_addr_words, MEM_N, 0)
    program_dma_and_start(3, DMA_S2MM_0_START_QUEUE, in2_addr_words, MEM_N, 2)

    src1: i32 = (MEM_SRC_ADDR >> 2) + MEMTILE_OWN
    # Both channels' programs (12 commands) concatenated into ONE control
    # transfer; a NoC stream header (NOC_CTRL_HDR) is baked in front of every
    # command except the first (the DMA supplies that one). 47 words total.
    # channel 0: memtile MM2S0 / BD0
    ctrl_buf[0] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[1] = 2
    ctrl_buf[2] = NOC_CTRL_HDR
    ctrl_buf[3] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[4] = 0
    ctrl_buf[5] = NOC_CTRL_HDR
    ctrl_buf[6] = make_write_packet_header(MEMTILE_DMA_BD_BASE, 3)
    ctrl_buf[7] = MEM_N
    ctrl_buf[8] = src1
    ctrl_buf[9] = 0
    ctrl_buf[10] = 0
    ctrl_buf[11] = NOC_CTRL_HDR
    ctrl_buf[12] = make_write_packet_header(MEMTILE_DMA_BD_BASE + 16, 3)
    ctrl_buf[13] = 0
    ctrl_buf[14] = 0
    ctrl_buf[15] = 0
    ctrl_buf[16] = 1 << 31
    ctrl_buf[17] = NOC_CTRL_HDR
    ctrl_buf[18] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
    ctrl_buf[19] = 1
    ctrl_buf[20] = NOC_CTRL_HDR
    ctrl_buf[21] = make_write_packet_header(MEMTILE_DMA_MM2S_0_START_QUEUE, 0)
    ctrl_buf[22] = 0
    # channel 1: memtile MM2S1 / BD1
    ctrl_buf[23] = NOC_CTRL_HDR
    ctrl_buf[24] = make_write_packet_header(MEMTILE_DMA_MM2S_1_CTRL, 0)
    ctrl_buf[25] = 2
    ctrl_buf[26] = NOC_CTRL_HDR
    ctrl_buf[27] = make_write_packet_header(MEMTILE_DMA_MM2S_1_CTRL, 0)
    ctrl_buf[28] = 0
    ctrl_buf[29] = NOC_CTRL_HDR
    ctrl_buf[30] = make_write_packet_header(MEMTILE_DMA_BD1_BASE, 3)
    ctrl_buf[31] = MEM_N
    ctrl_buf[32] = src1
    ctrl_buf[33] = 0
    ctrl_buf[34] = 0
    ctrl_buf[35] = NOC_CTRL_HDR
    ctrl_buf[36] = make_write_packet_header(MEMTILE_DMA_BD1_BASE + 16, 3)
    ctrl_buf[37] = 0
    ctrl_buf[38] = 0
    ctrl_buf[39] = 0
    ctrl_buf[40] = 1 << 31
    ctrl_buf[41] = NOC_CTRL_HDR
    ctrl_buf[42] = make_write_packet_header(MEMTILE_DMA_MM2S_1_CTRL, 0)
    ctrl_buf[43] = 1
    ctrl_buf[44] = NOC_CTRL_HDR
    ctrl_buf[45] = make_write_packet_header(MEMTILE_DMA_MM2S_1_START_QUEUE, 0)
    ctrl_buf[46] = 24  # push BD24 (MM2S1 / odd channel uses BD [24-47])

    t0: i32 = read_tm(TIMER_LOW)
    program_packet_mm2s(ctrl_addr_words + 0, 47)
    lk1: i32 = spin_lock_ge_bounded(LOCK1_VALUE, 1)
    t1: i32 = read_tm(TIMER_LOW)

    l0v: i32 = spin_lock_ge_bounded(LOCK0_VALUE, 1)
    l2v: i32 = spin_lock_ge_bounded(LOCK2_VALUE, 1)
    t2: i32 = read_tm(TIMER_LOW)

    telem[0] = bb - ba
    telem[1] = t1 - t0
    telem[2] = t2 - t1
    telem[3] = MEM_N
    telem[4] = 2
    telem[5] = in_buf[0]
    telem[6] = in_buf2[MEM_N - 1]
    telem[7] = lk1   # lock1 after program phase (expect 1: one transfer)
    telem[8] = l0v   # lock0 (ch0 / S2MM1 done, expect 1)
    telem[9] = l2v   # lock2 (ch1 / S2MM0 done, expect 1)

    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, TELEM_N, 1)
    spin_lock_ge(LOCK1_VALUE, 2)


# ── cdo / host mode: memtile MM2S preconfigured; core just receives, timed ───


@aie_kernel
def bench_passive(
    in_buf: ptr[i32, True],
    telem: ptr[i32, True],
    in_addr_words: i32,
    mode: i32,
):
    ba: i32 = read_tm(TIMER_LOW)
    bb: i32 = read_tm(TIMER_LOW)

    program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_addr_words, MEM_N, 0)

    t0: i32 = read_tm(TIMER_LOW)
    # No core-side programming of the memtile in cdo/host modes.
    t1: i32 = t0

    spin_lock_ge(LOCK0_VALUE, 1)
    t2: i32 = read_tm(TIMER_LOW)

    telem[0] = bb - ba
    telem[1] = t1 - t0          # ~ arm S2MM only (no memtile programming)
    telem[2] = t2 - t1
    telem[3] = MEM_N
    telem[4] = mode
    telem[5] = in_buf[0]
    telem[6] = in_buf[MEM_N - 1]

    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, TELEM_N, 1)
    spin_lock_ge(LOCK1_VALUE, 1)


def _set_controller_id(tile_op, packet_type, packet_id):
    tile_op.attributes["controller_id"] = aiex.AttrBuilder.get("PacketInfoAttr")(
        [packet_type, packet_id]
    )


def build_module(dev, kernel, mode, mem_n, depth=1):
    telem_ty = np.ndarray[(TELEM_N,), np.dtype[np.int32]]
    src_ty = np.ndarray[(mem_n,), np.dtype[np.int32]]
    in_n = mem_n * depth if mode == "chain" else mem_n
    in_ty = np.ndarray[(in_n,), np.dtype[np.int32]]
    ctrl_n = {"chain": 8 + 10 * depth, "multi": 47}.get(mode, CTRL_WORDS)
    ctrl_ty = np.ndarray[(ctrl_n,), np.dtype[np.int32]]
    in2_addr = CORE_IN_ADDR + mem_n * 4
    mem_init = np.arange(100, 100 + mem_n, dtype=np.int32)

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            kernel.resolve()
            t00 = tile(0, 0)
            t01 = tile(0, 1)
            t02 = tile(0, 2)

            mem_src = buffer(
                t01, datatype=src_ty, name="mem_src", address=MEM_SRC_ADDR,
                initial_value=mem_init,
            )
            telem = buffer(t02, datatype=telem_ty, name="telem", address=CORE_TELEM_ADDR)
            in_buf = buffer(t02, datatype=in_ty, name="in_buf", address=CORE_IN_ADDR)

            lock(t02, lock_id=0, init=0, sym_name="s2mm_done")
            lock(t02, lock_id=1, init=0, sym_name="mm2s_done")

            flow(t01, WireBundle.DMA, 0, t02, WireBundle.DMA, 1)
            flow(t02, WireBundle.DMA, 1, t00, WireBundle.DMA, 0)
            shim_dma_allocation("telem_alloc", t00, DMAChannelDir.S2MM, 0)

            if mode == "multi":
                lock(t02, lock_id=2, init=0, sym_name="s2mm0_done")
                in_buf2 = buffer(t02, datatype=in_ty, name="in_buf2", address=in2_addr)
                flow(t01, WireBundle.DMA, 1, t02, WireBundle.DMA, 0)  # memtile MM2S1 -> core S2MM0
                _set_controller_id(t01, CTRL_PACKET_TYPE, CTRL_PACKET_ID)
                ctrl_buf = buffer(t02, datatype=ctrl_ty, name="ctrl_buf", address=CORE_CTRL_ADDR)
                packetflow(
                    CTRL_PACKET_ID, t02, WireBundle.DMA, 0,
                    {"dest": t01, "port": WireBundle.TileControl, "channel": 0},
                )

                @core(t02)
                def core_body():
                    kernel(ctrl_buf, in_buf, in_buf2, telem,
                           CORE_CTRL_ADDR // 4, CORE_IN_ADDR // 4, in2_addr // 4)
            elif mode == "core_static":
                _set_controller_id(t01, CTRL_PACKET_TYPE, CTRL_PACKET_ID)
                ctrl_buf = buffer(
                    t02, datatype=ctrl_ty, name="ctrl_buf", address=CORE_CTRL_ADDR,
                    initial_value=ctrl_init_words(mem_n),
                )
                packetflow(
                    CTRL_PACKET_ID, t02, WireBundle.DMA, 0,
                    {"dest": t01, "port": WireBundle.TileControl, "channel": 0},
                )

                @core(t02)
                def core_body():
                    kernel(ctrl_buf, in_buf, telem,
                           CORE_CTRL_ADDR // 4, CORE_IN_ADDR // 4, mem_n)
            elif mode in ("core", "chain"):
                _set_controller_id(t01, CTRL_PACKET_TYPE, CTRL_PACKET_ID)
                ctrl_buf = buffer(
                    t02, datatype=ctrl_ty, name="ctrl_buf", address=CORE_CTRL_ADDR
                )
                packetflow(
                    CTRL_PACKET_ID, t02, WireBundle.DMA, 0,
                    {"dest": t01, "port": WireBundle.TileControl, "channel": 0},
                )

                @core(t02)
                def core_body():
                    kernel(ctrl_buf, in_buf, telem, CORE_CTRL_ADDR // 4, CORE_IN_ADDR // 4)
            else:
                if mode == "cdo":
                    # CDO configures + runs the memtile MM2S0 once (ungated).
                    @memtile_dma(t01)
                    def mt(block):
                        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
                        with block[1]:
                            dma_bd(mem_src, offset=0, len=mem_n)
                            next_bd(block[2])
                        with block[2]:
                            EndOp()

                @core(t02)
                def core_body():
                    kernel(in_buf, telem, CORE_IN_ADDR // 4, 1 if mode == "cdo" else 2)

            @runtime_sequence(telem_ty)
            def sequence(C):
                npu_maskwrite32(address=CORE_PROCESSOR_BUS, value=1, mask=1, column=0, row=2)
                if mode == "host":
                    # Host programs the memtile MM2S0 BD + start via MMIO.
                    bd = 0xA0000
                    npu_write32(column=0, row=1, address=bd + 0, value=mem_n)
                    npu_write32(column=0, row=1, address=bd + 4, value=(MEM_SRC_ADDR >> 2) + 0x20000)
                    npu_write32(column=0, row=1, address=bd + 8, value=0)
                    npu_write32(column=0, row=1, address=bd + 12, value=0)
                    npu_write32(column=0, row=1, address=bd + 16, value=0)
                    npu_write32(column=0, row=1, address=bd + 20, value=0)
                    npu_write32(column=0, row=1, address=bd + 24, value=0)
                    npu_write32(column=0, row=1, address=bd + 28, value=0x80000000)
                    npu_write32(column=0, row=1, address=0xA0634, value=0)
                    npu_write32(column=0, row=1, address=0xA0630, value=1)
                out_task = shim_dma_single_bd_task(
                    "telem_alloc", C, sizes=[1, 1, 1, TELEM_N], issue_token=True
                )
                dma_start_task(out_task)
                dma_await_task(out_task)
                dma_free_task(out_task)

        if not ctx.module.operation.verify():
            raise RuntimeError("MLIR verify failed")
        return ctx.module


def build_and_run(mode, mem_n, work_dir, depth=1, verbose=False):
    dev, arch = AIEDevice.npu2, "aie2p"
    g = _globals(mem_n, depth=depth)
    i32t = np.dtype[np.int32]
    telem_t = np.ndarray[(TELEM_N,), i32t]
    helpers = [odd_parity_header, make_write_packet_header,
               program_dma_and_start, program_packet_mm2s, spin_lock_ge,
               spin_lock_ge_bounded]
    if mode == "multi":
        kernel = PythocKernel(
            bench_core_2ch,
            [np.ndarray[(47,), i32t], np.ndarray[(mem_n,), i32t],
             np.ndarray[(mem_n,), i32t], telem_t, np.int32, np.int32, np.int32],
            target_arch=arch, extra_globals=g, helpers=helpers,
        )
    elif mode == "chain":
        kernel = PythocKernel(
            bench_core_chain,
            [np.ndarray[(8 + 10 * depth,), i32t], np.ndarray[(mem_n * depth,), i32t],
             telem_t, np.int32, np.int32],
            target_arch=arch, extra_globals=g, helpers=helpers,
        )
    elif mode == "core_static":
        kernel = PythocKernel(
            bench_core_static,
            [np.ndarray[(CTRL_WORDS,), i32t], np.ndarray[(mem_n,), i32t],
             telem_t, np.int32, np.int32, np.int32],
            target_arch=arch, extra_globals=g, helpers=helpers,
        )
    elif mode == "core":
        kernel = PythocKernel(
            bench_core,
            [np.ndarray[(CTRL_WORDS,), i32t], np.ndarray[(mem_n,), i32t],
             telem_t, np.int32, np.int32],
            target_arch=arch, extra_globals=g,
            helpers=[odd_parity_header, make_write_packet_header,
                     program_dma_and_start, program_packet_mm2s, spin_lock_ge],
        )
    else:
        kernel = PythocKernel(
            bench_passive,
            [np.ndarray[(mem_n,), i32t], telem_t, np.int32, np.int32],
            target_arch=arch, extra_globals=g,
            helpers=[program_dma_and_start, spin_lock_ge],
        )
    module = build_module(dev, kernel, mode, mem_n, depth=depth)
    wd = Path(work_dir) / f"{mode}_n{mem_n}_d{depth}"
    wd.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHOC_DEBUG_DIR"] = str(wd / "pythoc_objects")
    insts, xclbin = wd / "insts.bin", wd / "final.xclbin"
    compile_mlir_module(mlir_module=module, insts_path=str(insts),
                        xclbin_path=str(xclbin), work_dir=str(wd), verbose=verbose)
    npu_kernel = NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu_kernel)
    out = iron.zeros(TELEM_N, dtype=np.int32)
    DefaultNPURuntime.run(h, [out])
    t = [int(x) for x in np.array(out.numpy())]
    return {"bias": t[0], "program": t[1], "transfer": t[2], "n": t[3],
            "mode": t[4], "first": t[5], "last": t[6], "fill": t[7]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="core,core_static,cdo,host",
                   help="comma list of core,cdo,host,multi for the size sweep")
    p.add_argument("--sizes", default="8,32,128,512,2048,8192")
    p.add_argument("--depths", default="", help="chain-depth sweep (e.g. 1,2,4,8) at --chain-n")
    p.add_argument("--chain-n", type=int, default=256)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    def run(mode, n, depth=1):
        info = build_and_run(mode, n, args.work_dir, depth=depth, verbose=args.verbose)
        exp_last = 100 + (n if mode != "chain" else n) - 1
        ok = info["first"] == 100 and info["last"] == exp_last
        return info, ok

    rows = []  # (label, mode, program, transfer)
    for n in [int(x) for x in args.sizes.split(",")]:
        for mode in args.modes.split(","):
            try:
                info, ok = run(mode, n)
                rows.append((f"N={n}", mode, info["program"], info["transfer"]))
                print(f"  N={n:5d} {mode:5s}  program={info['program']:6d}  "
                      f"transfer={info['transfer']:8d}  {'OK' if ok else 'DATA-FAIL'}")
            except Exception as e:
                print(f"  N={n:5d} {mode:5s}  ERROR: {e}")

    if args.depths:
        print(f"\n  -- BD chain-depth sweep (core, N={args.chain_n}) --")
        for d in [int(x) for x in args.depths.split(",")]:
            try:
                info, ok = run("chain", args.chain_n, depth=d)
                rows.append((f"D={d}", "chain", info["program"], info["transfer"]))
                print(f"  D={d:2d}  program={info['program']:6d}  "
                      f"transfer={info['transfer']:8d}  {'OK' if ok else 'DATA-FAIL'}")
            except Exception as e:
                print(f"  D={d:2d}  ERROR: {e}")

    print("\n=== memtile DMA programming cost (core cycles) ===")
    print(f"{'config':>8} {'mode':>6} {'program':>9} {'transfer':>9} {'prog%total':>11}")
    for label, mode, prog, xfer in rows:
        tot = prog + xfer
        pct = (100.0 * prog / tot) if tot > 0 else 0.0
        print(f"{label:>8} {mode:>6} {prog:>9} {xfer:>9} {pct:>10.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
