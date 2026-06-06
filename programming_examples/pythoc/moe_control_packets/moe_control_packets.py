#!/usr/bin/env python3
# moe_control_packets.py -*- Python -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --scores 10,90,30,40,99,20,5,70 --work-dir ./moe_control_packets_build | FileCheck %s
# CHECK: top-k select : OK
# CHECK: fetched data : OK
# CHECK: PASS

"""Runtime, data-dependent DMA-BD reprogramming via AIE control packets.

This is a small, self-contained duplicate of the *clever bit* of the GPT-OSS-20B
MoE decode layer (see ../../../../GPTOSS_LAYER_ARCH.md, section 6). In that design
the `router_slave` compute core selects the top-k experts for a token at runtime,
then **rewrites shim-tile DMA buffer descriptors on the fly** by emitting AIE
*control packets* over statically-routed packet flows to the expert-weight shim
tiles' `TileControl` ports -- so each shim fetches exactly the selected experts'
weights from DDR, with no host round-trip.

The control-packet mechanism is identical whether the target is a shim tile (DDR,
as in GPT-OSS) or a memtile (on-chip). The companion microbench
../microbench/memtile_program_cost drives a memtile MM2S DMA this way; we build on
that proven path here, and add the MoE-defining ingredient: the *which BD address
to program* is computed at runtime from data.

Pipeline (all in column 0):

  shim(0,0) ──scores──► core(0,2) S2MM0          (host supplies per-expert scores)
  core(0,2) ── top-k argmax  (the "router")
  core(0,2) MM2S0 ──control packets──► memtile(0,1) TileControl
                                        └─ reprograms memtile MM2S0 BD:
                                           src = experts[idx] , len = CHUNK
  memtile(0,1) MM2S0 ──experts[idx]──► core(0,2) S2MM1
  core(0,2) MM2S1 ──results──► shim(0,0) S2MM0   (selected idxs + fetched chunks)

The memtile holds NUM_EXPERT "expert weight" chunks (CHUNK int32 each, baked with
a known pattern: expert e, element i == 1000*e + i). The host picks the winners by
setting `scores`; the core discovers them at runtime and control-packet-fetches
exactly those chunks. The host then verifies both the selected indices and the
fetched data -- proving the BD address field was patched from runtime data.

GPT-OSS correspondence:
  scores            ~ the 32 bf16 router logits (+bias)
  top-k argmax      ~ select_top_k_with_index (sort + stable softmax), here top-1..k
  memtile MM2S BD   ~ the expert-weight shim BD (configure_BD_x_MM2S_y_dma_bd_len)
  control packets   ~ control_packet_gen / packet_flow_gen (packet_control_gen.hpp)
  TileControl flow  ~ packetflow(pkt_id, router_slave -> IT[col] TileControl)
The only deliberate simplification is the *target* (memtile vs shim/DDR) and the
selection (argmax vs sort+softmax); the control-packet path is the same. The shim
register offsets needed to retarget this at a shim tile are listed at the bottom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import aie.dialects.aiex as aiex
import aie.iron as iron
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    WireBundle,
    buffer,
    core,
    device,
    flow,
    lock,
    packetflow,
    shim_dma_allocation,
    tile,
)
from aie.dialects.aiex import (
    dma_await_task,
    dma_free_task,
    dma_start_task,
    npu_maskwrite32,
    runtime_sequence,
    shim_dma_single_bd_task,
)
from aie.extras.context import mlir_mod_ctx
from aie.iron.pythoc import PythocKernel, aie_kernel
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module
from aie.utils.regdb import AIEAddressDecoder
from pythoc import i32, ptr, volatile
from pythoc.aie.operations import read_tm, write_tm

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "build"

# ── problem shape ────────────────────────────────────────────────────────────
NUM_EXPERT = 8     # "experts" resident in the memtile
CHUNK = 16         # int32 words per expert (the per-expert "weight" slab)
TOPK = 2           # experts the router selects+fetches per run
INT_MIN = -(2**31)  # argmax sentinel for "already picked"

# ── control-packet flow identity (matches memtile_program_cost) ──────────────
CTRL_PACKET_ID = 5
CTRL_PACKET_TYPE = 0

# ── core data-memory layout (byte addresses) ─────────────────────────────────
CORE_CTRL_ADDR = 0x1000    # ctrl_buf scratch (23 words)
CORE_SCORES_ADDR = 0x1400  # scores in (NUM_EXPERT words)
CORE_OUT_ADDR = 0x1800     # results out (OUT_N words)
CORE_IN_ADDR = 0x2000      # fetched expert chunks (TOPK*CHUNK words)
MEM_SRC_ADDR = 0x1000      # memtile expert store (NUM_EXPERT*CHUNK words)
MEMTILE_OWN = 0x20000      # memtile DMA "own address space" offset (words)

OUT_N = TOPK + TOPK * CHUNK  # [idx0..idx_{k-1}] + fetched chunks

# 23-word single-transfer memtile MM2S0 control program (see ctrl_init_words /
# bench_core in memtile_program_cost for the derivation of this layout).
CTRL_WORDS = 23


def _noc_ctrl_header():
    """On-stream NoC header the core DMA inserts for the *first* control command
    only; commands 2..N must carry it themselves (cf. _noc_ctrl_header in
    memtile_program_cost / AIETranslateControlPacketsToUI32Vec)."""
    hdr = ((CTRL_PACKET_TYPE & 0x7) << 12) | (CTRL_PACKET_ID & 0xFF)
    n, ones = hdr, 0
    while n:
        ones += n & 1
        n >>= 1
    pb = 1 if (ones % 2) == 0 else 0
    return (hdr | (pb << 31)) & 0xFFFFFFFF


NOC_CTRL_HDR = _noc_ctrl_header()
NOC_CTRL_HDR_S = NOC_CTRL_HDR - (1 << 32)  # signed i32 form for kernel globals

# ── register offsets (decoded once, baked as kernel globals) ─────────────────
_decoder = AIEAddressDecoder()
_reg = _decoder.get_register_offset

DMA_BD_BASE = _reg("DMA_BD0_0", "memory")
DMA_S2MM_0_START_QUEUE = _reg("DMA_S2MM_0_Start_Queue", "memory")
DMA_S2MM_1_START_QUEUE = _reg("DMA_S2MM_1_Start_Queue", "memory")
DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory")
DMA_MM2S_1_START_QUEUE = _reg("DMA_MM2S_1_Start_Queue", "memory")
LOCK0_VALUE = _reg("Lock0_value", "memory")
LOCK1_VALUE = _reg("Lock1_value", "memory")
LOCK2_VALUE = _reg("Lock2_value", "memory")
CORE_PROCESSOR_BUS = _reg("Core_Processor_Bus", "core")

MEMTILE_DMA_MM2S_0_START_QUEUE = _reg("DMA_MM2S_0_Start_Queue", "memory_tile")
MEMTILE_DMA_MM2S_0_CTRL = _reg("DMA_MM2S_0_Ctrl", "memory_tile")
MEMTILE_DMA_BD_BASE = _reg("DMA_BD0_0", "memory_tile")


def _globals():
    return {
        "volatile": volatile,
        "NUM_EXPERT": NUM_EXPERT,
        "CHUNK": CHUNK,
        "TOPK": TOPK,
        "INT_MIN": INT_MIN,
        "OUT_N": OUT_N,
        "MEM_SRC_ADDR": MEM_SRC_ADDR,
        "MEMTILE_OWN": MEMTILE_OWN,
        "CTRL_PACKET_ID": CTRL_PACKET_ID,
        "CTRL_PACKET_TYPE": CTRL_PACKET_TYPE,
        "NOC_CTRL_HDR": NOC_CTRL_HDR_S,
        "DMA_BD_BASE": DMA_BD_BASE,
        "DMA_S2MM_0_START_QUEUE": DMA_S2MM_0_START_QUEUE,
        "DMA_S2MM_1_START_QUEUE": DMA_S2MM_1_START_QUEUE,
        "DMA_MM2S_0_START_QUEUE": DMA_MM2S_0_START_QUEUE,
        "DMA_MM2S_1_START_QUEUE": DMA_MM2S_1_START_QUEUE,
        "LOCK0_VALUE": LOCK0_VALUE,
        "LOCK1_VALUE": LOCK1_VALUE,
        "LOCK2_VALUE": LOCK2_VALUE,
        "MEMTILE_DMA_MM2S_0_START_QUEUE": MEMTILE_DMA_MM2S_0_START_QUEUE,
        "MEMTILE_DMA_MM2S_0_CTRL": MEMTILE_DMA_MM2S_0_CTRL,
        "MEMTILE_DMA_BD_BASE": MEMTILE_DMA_BD_BASE,
    }


# ── kernel helpers (mirrors memtile_program_cost) ────────────────────────────


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
    """Arm a *core* DMA channel: write its BD then push the start queue. The BD
    releases lock `lock_rel_id` on completion."""
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
    """Arm the core MM2S0 BD0 to stream `num_words` control words as a control
    packet (packet-id CTRL_PACKET_ID); releases lock1 on completion."""
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


# ── the router core ──────────────────────────────────────────────────────────


@aie_kernel
def moe_router(
    scores: ptr[i32, True],
    # volatile: the control program is consumed by the MM2S DMA, not read back by
    # the core, so LLVM would otherwise dead-store-eliminate these writes (it can't
    # see the DMA reading ctrl_buf). volatile keeps every word and its ordering.
    ctrl_buf: ptr[volatile[i32], True],
    in_buf: ptr[i32, True],
    out_buf: ptr[i32, True],
    scores_addr_words: i32,
    ctrl_addr_words: i32,
    in_addr_words: i32,
    out_addr_words: i32,
):
    # 1. receive per-expert scores from the host (core S2MM0 -> lock2).
    program_dma_and_start(3, DMA_S2MM_0_START_QUEUE, scores_addr_words, NUM_EXPERT, 2)
    spin_lock_ge(LOCK2_VALUE, 1)

    # 2. top-k select (the "router"): repeated argmax, masking winners with INT_MIN.
    for k in range(TOPK):
        best: i32 = INT_MIN
        best_i: i32 = 0
        e: i32 = 0
        while e < NUM_EXPERT:
            s: i32 = scores[e]
            if s > best:
                best = s
                best_i = e
            e = e + 1
        out_buf[k] = best_i
        scores[best_i] = INT_MIN  # don't pick this expert again

    # 3. fetch each selected expert by control-packet-reprogramming the memtile
    #    MM2S0 BD to that expert's slab, then receiving it on core S2MM1.
    #
    # Two non-obvious requirements (both learned the hard way; see README):
    #   - This must be a `for ... in range(TOPK)` (trace-time unrolled to straight
    #     line code), NOT a runtime `while` loop. A real loop around the
    #     control-DMA launch does not deliver on this path.
    #   - ctrl_buf is `volatile` (see signature). The control program is consumed
    #     by the DMA, never read back by the core, so without volatile LLVM
    #     dead-store-eliminates the constant words and the memtile gets garbage.
    # ctrl_buf[8] (the source base) is the runtime, data-dependent patch -- this is
    # the per-expert DMA-BD-address rewrite that GPT-OSS performs.
    for k in range(TOPK):
        idx: i32 = out_buf[k]
        src_base: i32 = (MEM_SRC_ADDR >> 2) + MEMTILE_OWN + idx * CHUNK
        # arm the receive (core S2MM1 -> in_buf[k*CHUNK:], releases lock0)
        program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_addr_words + k * CHUNK, CHUNK, 0)
        # build the data-dependent control program in place
        ctrl_buf[0] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
        ctrl_buf[1] = 2  # cmd0 reset (DMA prepends the NoC header)
        ctrl_buf[2] = NOC_CTRL_HDR
        ctrl_buf[3] = make_write_packet_header(MEMTILE_DMA_MM2S_0_CTRL, 0)
        ctrl_buf[4] = 0  # cmd1 unreset
        ctrl_buf[5] = NOC_CTRL_HDR
        ctrl_buf[6] = make_write_packet_header(MEMTILE_DMA_BD_BASE, 3)
        ctrl_buf[7] = CHUNK  # BD word0: transfer length (words)
        ctrl_buf[8] = src_base  # BD word1: source base (runtime, data-dependent)
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
        ctrl_buf[22] = 0  # cmd5 start-queue push (BD0)
        # send all 6 commands as ONE control-packet transfer (core MM2S0 -> lock1)
        program_packet_mm2s(ctrl_addr_words, 23)
        spin_lock_ge(LOCK1_VALUE, k + 1)  # control transfer landed
        spin_lock_ge(LOCK0_VALUE, k + 1)  # expert chunk arrived

    # 4. stage results: selected indices (already in out_buf[0:TOPK]) + chunks.
    i: i32 = 0
    while i < TOPK * CHUNK:
        out_buf[TOPK + i] = in_buf[i]
        i = i + 1

    # return results to the host (core MM2S1 -> shim, releases lock1).
    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, out_addr_words, OUT_N, 1)
    spin_lock_ge(LOCK1_VALUE, TOPK + 1)


def _set_controller_id(tile_op, packet_type, packet_id):
    tile_op.attributes["controller_id"] = aiex.AttrBuilder.get("PacketInfoAttr")(
        [packet_type, packet_id]
    )


def build_module(dev, kernel):
    scores_ty = np.ndarray[(NUM_EXPERT,), np.dtype[np.int32]]
    out_ty = np.ndarray[(OUT_N,), np.dtype[np.int32]]
    in_ty = np.ndarray[(TOPK * CHUNK,), np.dtype[np.int32]]
    src_ty = np.ndarray[(NUM_EXPERT * CHUNK,), np.dtype[np.int32]]
    ctrl_ty = np.ndarray[(CTRL_WORDS,), np.dtype[np.int32]]

    # expert e, element i == 1000*e + i  (a known, per-expert pattern)
    experts = (np.arange(NUM_EXPERT)[:, None] * 1000 + np.arange(CHUNK)[None, :]).astype(
        np.int32
    ).reshape(-1)

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            kernel.resolve()
            t00 = tile(0, 0)  # shim
            t01 = tile(0, 1)  # memtile (expert store)
            t02 = tile(0, 2)  # router compute core

            mem_src = buffer(
                t01, datatype=src_ty, name="mem_src", address=MEM_SRC_ADDR,
                initial_value=experts,
            )
            scores = buffer(t02, datatype=scores_ty, name="scores", address=CORE_SCORES_ADDR)
            ctrl_buf = buffer(t02, datatype=ctrl_ty, name="ctrl_buf", address=CORE_CTRL_ADDR)
            in_buf = buffer(t02, datatype=in_ty, name="in_buf", address=CORE_IN_ADDR)
            out_buf = buffer(t02, datatype=out_ty, name="out_buf", address=CORE_OUT_ADDR)

            lock(t02, lock_id=0, init=0, sym_name="s2mm1_done")   # expert chunk in
            lock(t02, lock_id=1, init=0, sym_name="mm2s_done")    # control + out
            lock(t02, lock_id=2, init=0, sym_name="scores_done")  # scores in

            # data flows
            flow(t00, WireBundle.DMA, 0, t02, WireBundle.DMA, 0)  # shim -> core (scores)
            flow(t01, WireBundle.DMA, 0, t02, WireBundle.DMA, 1)  # memtile -> core (expert)
            flow(t02, WireBundle.DMA, 1, t00, WireBundle.DMA, 0)  # core -> shim (results)
            shim_dma_allocation("scores_alloc", t00, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("out_alloc", t00, DMAChannelDir.S2MM, 0)

            # control-packet flow: core MM2S0 -> memtile TileControl (reprogram BDs)
            _set_controller_id(t01, CTRL_PACKET_TYPE, CTRL_PACKET_ID)
            packetflow(
                CTRL_PACKET_ID, t02, WireBundle.DMA, 0,
                {"dest": t01, "port": WireBundle.TileControl, "channel": 0},
            )

            @core(t02)
            def core_body():
                kernel(
                    scores, ctrl_buf, in_buf, out_buf,
                    CORE_SCORES_ADDR // 4, CORE_CTRL_ADDR // 4,
                    CORE_IN_ADDR // 4, CORE_OUT_ADDR // 4,
                )

            @runtime_sequence(scores_ty, out_ty)
            def sequence(S, O):
                # release the core from reset so it starts executing.
                npu_maskwrite32(address=CORE_PROCESSOR_BUS, value=1, mask=1, column=0, row=2)
                in_task = shim_dma_single_bd_task(
                    "scores_alloc", S, sizes=[1, 1, 1, NUM_EXPERT], issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    "out_alloc", O, sizes=[1, 1, 1, OUT_N], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(in_task, out_task)
                dma_free_task(in_task, out_task)

        if not ctx.module.operation.verify():
            raise RuntimeError("MLIR verify failed")
        return ctx.module


def build_and_run(scores_in, work_dir, verbose=False):
    dev, arch = AIEDevice.npu2, "aie2p"
    i32t = np.dtype[np.int32]
    kernel = PythocKernel(
        moe_router,
        [
            np.ndarray[(NUM_EXPERT,), i32t],
            np.ndarray[(CTRL_WORDS,), i32t],
            np.ndarray[(TOPK * CHUNK,), i32t],
            np.ndarray[(OUT_N,), i32t],
            np.int32, np.int32, np.int32, np.int32,
        ],
        target_arch=arch,
        extra_globals=_globals(),
        helpers=[
            odd_parity_header, make_write_packet_header, program_dma_and_start,
            program_packet_mm2s, spin_lock_ge,
        ],
    )
    module = build_module(dev, kernel)
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    insts, xclbin = wd / "insts.bin", wd / "final.xclbin"
    compile_mlir_module(
        mlir_module=module, insts_path=str(insts), xclbin_path=str(xclbin),
        work_dir=str(wd), verbose=verbose,
    )
    npu_kernel = NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu_kernel)
    s_in = iron.tensor(np.asarray(scores_in, dtype=np.int32), dtype=np.int32)
    out = iron.zeros(OUT_N, dtype=np.int32)
    DefaultNPURuntime.run(h, [s_in, out])
    return np.array(out.numpy())


def expected(scores_in):
    """Host mirror of the router's top-k argmax + experts pattern."""
    s = np.asarray(scores_in, dtype=np.int64).copy()
    idxs = []
    for _ in range(TOPK):
        best = int(np.argmax(s))  # first max on ties -> matches kernel
        idxs.append(best)
        s[best] = INT_MIN
    chunks = []
    for idx in idxs:
        chunks.append(np.arange(CHUNK, dtype=np.int32) + 1000 * idx)
    return np.array(idxs, dtype=np.int32), np.concatenate(chunks).astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scores", default="",
        help=f"comma list of {NUM_EXPERT} ints (default: random)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.scores:
        scores = np.array([int(x) for x in args.scores.split(",")], dtype=np.int32)
        assert scores.size == NUM_EXPERT, f"need {NUM_EXPERT} scores"
    else:
        rng = np.random.default_rng(args.seed)
        scores = rng.integers(0, 1000, size=NUM_EXPERT).astype(np.int32)

    out = build_and_run(scores, args.work_dir, verbose=args.verbose)
    got_idx = out[:TOPK]
    got_data = out[TOPK : TOPK + TOPK * CHUNK]
    exp_idx, exp_data = expected(scores)

    print(f"  scores        : {list(int(x) for x in scores)}")
    print(f"  selected (hw) : {list(int(x) for x in got_idx)}")
    print(f"  selected (exp): {list(int(x) for x in exp_idx)}")
    idx_ok = np.array_equal(got_idx, exp_idx)
    data_ok = np.array_equal(got_data, exp_data)
    for k in range(TOPK):
        seg = got_data[k * CHUNK : (k + 1) * CHUNK]
        print(f"  expert[{int(got_idx[k])}] first/last fetched: {int(seg[0])} / {int(seg[-1])}")
    ok = idx_ok and data_ok
    print(f"\n  top-k select : {'OK' if idx_ok else 'FAIL'}")
    print(f"  fetched data : {'OK' if data_ok else 'FAIL'}")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ── retargeting at a shim tile (GPT-OSS's actual configuration) ──────────────
# The control-packet path above is identical for a shim tile; only the addressed
# registers change. To drive a shim(col,0) MM2S DMA instead of the memtile, use
# the "shim" register module (verified to match packet_control_gen.hpp):
#   DMA_BD0_0           -> 0x1D000  (get_SHM_bd_x_0_address(bd) = 0x1D000+bd*0x20)
#   DMA_MM2S_0_Ctrl     -> 0x1D210
#   DMA_MM2S_0_Task_Queue -> 0x1D214 (Shimtile_MM2S_X_TASK_QUEUE_addr)
# and point the packetflow dest at {shim_tile, WireBundle.TileControl, 0}. The BD
# source address then becomes the DDR buffer-object virtual base + expert offset;
# GPT-OSS reads that virtual base back over a response packet flow first
# (get_SHM_bd_virtual_address -> control_packet_gen with operation=read).


if __name__ == "__main__":
    raise SystemExit(main())
