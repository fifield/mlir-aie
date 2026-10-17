# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Host for npu3_mmac_flat_aievec: run the flat-staging int8 GEMM and
compare C with a NumPy reference (int8 x int8 or uint8 -> int32, wrapping).

The MLIR design is constructed below with the Python bindings.
"""

# Compare flat-staging GEMM with NumPy for stream/L1 feeds and signed/unsigned
# int8 weights. Prime before A/C acquires; interleave prefetch with row MMACs.
# A nonzero initial accumulator makes first-tile zero_if observable.
#
# The schedule checker simulates two hardware-loop iterations. Stream feeds
# have a peeled tile and a final tile, giving four checked tiles. L1 feeds
# have one tile per loop iteration plus a final tile, giving three. Each
# tile has 32 inserts and 16 MMACs, plus 32 pops for stream feeds.
#
# REQUIRES: ryzen_ai_npu3, peano, xrt_python_bindings
#
# RUN: %python %s --emit --feed stream --rhs int8 --kt 3 -o stream-int8.mlir
# RUN: %aiecc --get-full-elf --full-elf-name=stream-int8.elf --tmpdir stream-int8.prj stream-int8.mlir
# RUN: llvm-objdump -d stream-int8.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf > stream-int8.dis
# RUN: %python %S/../../aiecc/aie4-mmac-schedule-checker.py stream-int8.dis --inserts-per-tile 32,32,32,32 --pops-per-tile 32,32,32,32 --mmacs-per-tile 16,16,16,16 --forbid-lda-s8 | FileCheck %s --check-prefix=STREAM-ORDER
# RUN: %run_on_npu3% %python %s stream-int8.elf --feed stream --rhs int8 --kt 3 | FileCheck %s --check-prefix=STREAM-INT8
# STREAM-ORDER: ORDER OK: 4 tiles, 128 inserts, 64 MMACs, 128 pops, no accumulator spills (loop body checked twice)
# STREAM-INT8: stream int8 KT=3: 1024/1024 exact
# STREAM-INT8: PASS
#
# RUN: %python %s --emit --feed stream --rhs uint8 --kt 3 -o stream-uint8.mlir
# RUN: %aiecc --get-full-elf --full-elf-name=stream-uint8.elf --tmpdir stream-uint8.prj stream-uint8.mlir
# RUN: llvm-objdump -d stream-uint8.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf > stream-uint8.dis
# RUN: %python %S/../../aiecc/aie4-mmac-schedule-checker.py stream-uint8.dis --inserts-per-tile 32,32,32,32 --pops-per-tile 32,32,32,32 --mmacs-per-tile 16,16,16,16 --forbid-lda-s8 | FileCheck %s --check-prefix=STREAM-ORDER
# RUN: %run_on_npu3% %python %s stream-uint8.elf --feed stream --rhs uint8 --kt 3 | FileCheck %s --check-prefix=STREAM-UINT8
# STREAM-UINT8: stream uint8 KT=3: 1024/1024 exact
# STREAM-UINT8: PASS
#
# L1 int8 uses five K tiles to exercise staging across real loop back edges.
# RUN: %python %s --emit --feed l1 --rhs int8 --kt 5 -o l1-int8.mlir
# RUN: %aiecc --get-full-elf --full-elf-name=l1-int8.elf --tmpdir l1-int8.prj l1-int8.mlir
# RUN: llvm-objdump -d l1-int8.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf > l1-int8.dis
# RUN: %python %S/../../aiecc/aie4-mmac-schedule-checker.py l1-int8.dis --inserts-per-tile 32,32,32 --mmacs-per-tile 16,16,16 --forbid-lda-s8 | FileCheck %s --check-prefix=L1-ORDER
# RUN: %run_on_npu3% %python %s l1-int8.elf --feed l1 --rhs int8 --kt 5 | FileCheck %s --check-prefix=L1-INT8
# L1-ORDER: ORDER OK: 3 tiles, 96 inserts, 48 MMACs, no accumulator spills (loop body checked twice)
# L1-INT8: l1 int8 KT=5: 1024/1024 exact
# L1-INT8: PASS
#
# RUN: %python %s --emit --feed l1 --rhs uint8 --kt 3 -o l1-uint8.mlir
# RUN: %aiecc --get-full-elf --full-elf-name=l1-uint8.elf --tmpdir l1-uint8.prj l1-uint8.mlir
# RUN: llvm-objdump -d l1-uint8.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf > l1-uint8.dis
# RUN: %python %S/../../aiecc/aie4-mmac-schedule-checker.py l1-uint8.dis --inserts-per-tile 32,32,32 --mmacs-per-tile 16,16,16 --forbid-lda-s8 | FileCheck %s --check-prefix=L1-ORDER
# RUN: %run_on_npu3% %python %s l1-uint8.elf --feed l1 --rhs uint8 --kt 3 | FileCheck %s --check-prefix=L1-UINT8
# L1-UINT8: l1 uint8 KT=3: 1024/1024 exact
# L1-UINT8: PASS
#
# Five K tiles: the zero-overhead loop body runs three times.
# RUN: %python %s --emit --feed stream --rhs int8 --kt 5 -o stream-int8-kt5.mlir
# RUN: %aiecc --get-full-elf --full-elf-name=stream-int8-kt5.elf --tmpdir stream-int8-kt5.prj stream-int8-kt5.mlir
# RUN: llvm-objdump -d stream-int8-kt5.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf > stream-int8-kt5.dis
# RUN: %python %S/../../aiecc/aie4-mmac-schedule-checker.py stream-int8-kt5.dis --inserts-per-tile 32,32,32,32 --pops-per-tile 32,32,32,32 --mmacs-per-tile 16,16,16,16 --forbid-lda-s8 | FileCheck %s --check-prefix=STREAM-ORDER
# RUN: %run_on_npu3% %python %s stream-int8-kt5.elf --feed stream --rhs int8 --kt 5 | FileCheck %s --check-prefix=STREAM-KT5
# STREAM-KT5: stream int8 KT=5: 1024/1024 exact
# STREAM-KT5: PASS
#
# Port 0 leaves the slot-1 constant shared with the loop increment, exercising
# the indexed-autoincrement workaround as the L1 feed does.
# RUN: %python %s --emit --feed stream --rhs int8 --kt 5 --port 0 -o stream-port0.mlir
# RUN: %aiecc --get-full-elf --full-elf-name=stream-port0.elf --tmpdir stream-port0.prj stream-port0.mlir
# RUN: %run_on_npu3% %python %s stream-port0.elf --feed stream --rhs int8 --kt 5 | FileCheck %s --check-prefix=STREAM-KT5

import argparse
import sys

import numpy as np

N = K = 64


def emit_design(feed, rhs, kt, m, port):
    from aie.dialects import arith, scf, vector
    from aie.dialects.aie import (
        AIEDevice,
        DMAChannelDir,
        ObjectFifoPort,
        WireBundle,
        core,
        device,
        dma_bd,
        end,
        flow,
        object_fifo,
        packetflow,
        shim_dma_allocation,
        tile,
    )
    from aie.dialects.aiex import (
        bds,
        dma_await_task,
        dma_configure_task_for,
        dma_free_task,
        dma_start_task,
        runtime_sequence,
    )
    from aie.helpers.dialects.aievec_aie4 import (
        INT8,
        UINT8,
        commit,
        mmac_active,
        stage,
        stage_stream,
    )
    from aie.ir import (
        Context,
        DenseElementsAttr,
        IntegerAttr,
        IndexType,
        InsertionPoint,
        IntegerType,
        Location,
        Module,
        MemRefType,
        BoolAttr,
        VectorType,
    )

    N = 64
    K = 64
    SLOTS = 32
    CHUNK_BYTES = 128
    TILE_BYTES = K * N
    ACC_INIT = 12345

    def build_device(feed, kt, m, port):
        i8 = IntegerType.get_signless(8)
        i32 = IntegerType.get_signless(32)
        a_ty = MemRefType.get([kt * m * K], i8)
        w_ty = MemRefType.get([kt * TILE_BYTES], i8)
        c_ty = MemRefType.get([m * N], i32)
        module = Module.create()
        acquires = []
        releases = []
        with InsertionPoint(module.body):

            @device(AIEDevice.npu3)
            def device_body():
                shim = tile(0, 0, packet_id=0)
                compute = tile(0, 2)
                control = packetflow(
                    0,
                    shim,
                    WireBundle.Control32,
                    0,
                    {"dest": shim, "port": WireBundle.Controller32, "channel": 0},
                    keep_pkt_header=True,
                )
                control.attributes["priority_route"] = BoolAttr.get(True)
                if feed == "stream":
                    flow(shim, WireBundle.DMA, 1, compute, WireBundle.Core, port)
                    in_w = shim_dma_allocation("inW", shim, DMAChannelDir.MM2S, 1)
                else:
                    in_w = object_fifo("inW", shim, [compute], 1, w_ty)
                in_a = object_fifo("inA", shim, [compute], 1, a_ty)
                out = object_fifo("out", compute, [shim], 1, c_ty)

                @core(compute)
                def core_body():
                    if feed == "l1":
                        acquires.append(in_w.acquire(ObjectFifoPort.Consume, 1))
                    acquires.append(in_a.acquire(ObjectFifoPort.Consume, 1))
                    acquires.append(out.acquire(ObjectFifoPort.Produce, 1))
                    releases.append(in_a.release(ObjectFifoPort.Consume, 1))
                    out.release(ObjectFifoPort.Produce, 1)
                    if feed == "l1":
                        in_w.release(ObjectFifoPort.Consume, 1)

                @runtime_sequence(a_ty, w_ty, c_ty)
                def sequence(a, w, c):
                    def task(alloc, mem, length, issue_token=False):
                        configured = dma_configure_task_for(
                            alloc, issue_token=issue_token
                        )
                        with bds(configured) as bd:
                            with bd[0]:
                                dma_bd(mem, offset=0, transfer_len=length)
                                end()
                        return configured

                    tout = task(out, c, m * N, issue_token=True)
                    ta = task(in_a, a, kt * m * K)
                    tw = task(in_w, w, kt * TILE_BYTES)
                    dma_start_task(tout, ta, tw)
                    dma_await_task(tout)
                    dma_free_task(ta, tw)

        return module, acquires, releases[0]

    def emit(feed, rhs, kt, m, port=1):
        assert SLOTS % m == 0, "pops per row must be a whole number"
        assert kt >= 2, "the K loop needs at least one iteration"
        fmt = INT8 if rhs == "int8" else UINT8
        rhs_signed = True if rhs == "int8" else None  # UINT8 implies False
        pops_per_row = SLOTS // m
        with Context(), Location.unknown():
            module, acquires, first_release = build_device(feed, kt, m, port)
            a_mem = acquires[-2]
            c_mem = acquires[-1]
            w_mem = acquires[0] if feed == "l1" else None

            i8 = IntegerType.get_signless(8)
            i32 = IntegerType.get_signless(32)
            idx = IndexType.get()
            acc_ty = VectorType.get([N], i32)

            def cidx(v):
                return arith.constant(idx, v)

            def stage_tile_chunk(tile, slot, ktile=None):
                """Write staging slot `slot` of B tile `tile` (an int, or the
                index Value `ktile` for a run-time tile)."""
                if feed == "stream":
                    stage_stream(fmt, port, slot, rhs_signed=rhs_signed)
                    return
                if ktile is None:
                    off = cidx(tile * TILE_BYTES + slot * CHUNK_BYTES)
                else:
                    base = arith.muli(ktile, cidx(TILE_BYTES))
                    off = arith.addi(base, cidx(slot * CHUNK_BYTES))
                chunk = vector.load(VectorType.get([CHUNK_BYTES], i8), w_mem, [off])
                stage(fmt, chunk, slot, rhs_signed=rhs_signed)

            # Prime: tile 0, before the A and C acquires.
            with InsertionPoint(acquires[-2].owner):
                for s in range(SLOTS):
                    stage_tile_chunk(0, s)

            def rows(ktile, accs, zero_if, prefetch_tile):
                out = []
                for r in range(m):
                    row = arith.addi(arith.muli(ktile, cidx(m)), cidx(r))
                    off = arith.muli(row, cidx(K))
                    a = vector.load(VectorType.get([K], i8), a_mem, [off])
                    out.append(
                        mmac_active(
                            fmt,
                            a,
                            accs[r],
                            zero_if=zero_if,
                            lhs_signed=True,
                            rhs_signed=rhs_signed,
                        )
                    )
                    if prefetch_tile is not None:
                        for p in range(pops_per_row):
                            stage_tile_chunk(
                                None, r * pops_per_row + p, ktile=prefetch_tile
                            )
                return out

            with InsertionPoint(first_release):
                init = arith.constant(
                    acc_ty,
                    DenseElementsAttr.get_splat(acc_ty, IntegerAttr.get(i32, ACC_INIT)),
                )
                trip = cidx(kt - 1)
                loop = scf.ForOp(cidx(0), trip, cidx(1), [init] * m)
                with InsertionPoint(loop.body):
                    ktile = loop.induction_variable
                    commit(fmt, rhs_signed=rhs_signed)
                    first = arith.cmpi(arith.CmpIPredicate.eq, ktile, cidx(0))
                    nxt = arith.addi(ktile, cidx(1))
                    scf.YieldOp(rows(ktile, list(loop.inner_iter_args), first, nxt))
                # Peeled last tile: no prefetch.
                commit(fmt, rhs_signed=rhs_signed)
                last = rows(trip, list(loop.results), None, None)
                for r in range(m):
                    vector.store(last[r], c_mem, [cidx(r * N)])
            assert module.operation.verify()
            return str(module)

    return emit(feed, rhs, kt, m, port)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("elf", nargs="?")
    p.add_argument("--emit", action="store_true", help="construct the MLIR design")
    p.add_argument("--port", type=int, choices=[0, 1], default=1)
    p.add_argument("-o", "--output")
    p.add_argument("--feed", choices=["stream", "l1"], required=True)
    p.add_argument("--rhs", choices=["int8", "uint8"], required=True)
    p.add_argument("--kt", type=int, default=3)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.emit:
        if not args.output:
            p.error("--emit requires --output")
        with open(args.output, "w") as out:
            out.write(emit_design(args.feed, args.rhs, args.kt, args.m, args.port))
        return 0
    if not args.elf:
        p.error("an ELF is required for the hardware run")
    import pyxrt as xrt

    kt, m = args.kt, args.m

    rng = np.random.default_rng(args.seed)
    a = rng.integers(-128, 128, size=(kt, m, K), dtype=np.int8)
    b_dtype = np.int8 if args.rhs == "int8" else np.uint8
    info = np.iinfo(b_dtype)
    b = rng.integers(info.min, info.max + 1, size=(kt, K, N), dtype=b_dtype)
    # W: per tile, B^T row-major (B stored column-major).
    w = np.ascontiguousarray(b.transpose(0, 2, 1)).view(np.uint8)

    ref = np.zeros((m, N), dtype=np.int64)
    for k in range(kt):
        ref += a[k].astype(np.int64) @ b[k].astype(np.int64)
    ref = ((ref + 2**31) % 2**32 - 2**31).astype(np.int32)

    device = xrt.device(0)
    context = xrt.hw_context(device, xrt.elf(args.elf))
    kernel = xrt.ext.kernel(context, "main:sequence")
    bos = []
    for data in (a.view(np.uint8).reshape(-1), w.reshape(-1)):
        bo = xrt.ext.bo(device, data.nbytes)
        np.frombuffer(bo.map(), dtype=np.uint8, count=data.nbytes)[:] = data
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bos.append(bo)
    out_bytes = m * N * 4
    bo_out = xrt.ext.bo(device, out_bytes)
    np.frombuffer(bo_out.map(), dtype=np.uint8, count=out_bytes)[:] = 0xA5
    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = kernel(bos[0], bos[1], bo_out)
    state = run.wait()
    if hasattr(xrt, "ert_cmd_state") and (
        state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    ):
        print(f"kernel returned {state}")
        return 1
    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    got = np.frombuffer(bo_out.map(), dtype=np.int32, count=m * N).reshape(m, N)

    exact = int(np.count_nonzero(got == ref))
    print(f"{args.feed} {args.rhs} KT={kt}: {exact}/{m * N} exact")
    if exact != m * N:
        bad = np.argwhere(got != ref)[:4]
        for r, c in bad:
            print(f"  [{r}][{c}] got {got[r, c]}, expected {ref[r, c]}")
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
