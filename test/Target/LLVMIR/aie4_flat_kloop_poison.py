#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Regression for the gemm_int8 KERNEL=aievec core: a runtime K-loop
# whose 64 accumulator iter_args start as ub.poison and whose MMACs use
# a dynamic zero_if (kt == 0).
#  - the poison reaches Peano as phi inputs rather than zero initializers;
#  - the conf register of mmac.i is an `or` of the kt == 0 flag, so the
#    first iteration is not peeled;
#  - the loop holds exactly one K-tile (one commit, 64 MMACs, 32 pops): not
#    peeled, unrolled or software-pipelined. The timing checker follows the
#    whole core: prime, two loop iterations, peeled last K-tile.
#  - no staging insert uses a post-incremented index register (Peano's
#    combine_indexed_to_autoinc can steal the loop counter's +1).
#
# REQUIRES: peano
# RUN: %python %s > %t.mlir
# RUN: %aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %t.mlir
# RUN: FileCheck %s --check-prefix=IR --input-file %t.prj/opted_main_core_0_2.ll
# RUN: llvm-objdump -d --no-show-raw-insn %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.s
# RUN: FileCheck %s --check-prefix=ASM --input-file %t.s
# RUN: %python %S/../../aiecc/aie4-mmac-schedule-checker.py %t.s --mmacs-per-tile 64,64,64 --inserts-per-tile 32,32,32 --pops-per-tile 32,32,32 --forbid-lda-s8 | FileCheck %s --check-prefix=ORDER
# RUN: not grep -E "za\[i[0-9]+\+\+" %t.s

# IR-COUNT-64: phi <64 x i32> [ poison
# IR-NOT:      phi <64 x i32> [ poison
# IR-NOT:      phi <64 x i32> [ zeroinitializer

# ASM:      movxm{{[[:space:]]+}}le, #
# ASM:      {{eqz?}}{{[[:space:]]+}}[[Z:r[0-9]+]],
# ASM:      or{{[[:space:]]+}}[[CONF:r[0-9]+]], [[Z]],
# ASM:      mmac.i{{.*}}, zb, [[CONF]]
# ASM:      <.L_LEnd

# ORDER: ORDER OK: 3 tiles, 96 inserts, 192 MMACs, 96 pops, no accumulator spills (loop body checked twice)

"""Emit the flat-AIEVec K-loop of the gemm_int8 KERNEL=aievec core, alone.

One core primes 32 staging slots from stream 1, then runs a runtime
K-loop over K-tiles 0..KT-2 whose 64 accumulator iter_args start as
ub.poison: every MMAC has a dynamic zero_if (kt == 0), so the poison is
never read. The loop body commits,
issues 64 row MMACs and pops the next K-tile after every odd row; the last
K-tile is peeled (no zero_if, no pops) and stored. A rows are read through
a ptr-dialect pointer carried by the K-loop, as the generator does.

64 rows as in gemm_int8: with 32 rows or fewer llc's MachinePipeliner
pipelines this loop today.
"""

import argparse

from aie.dialects import arith, builtin, scf, ub, vector
from aie.dialects.aie import AIEDevice, Core, Device, buffer, end, tile
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.aievec_aie4 import INT8, commit, mmac_active, stage_stream
from aie.ir import (
    Attribute,
    Block,
    InsertionPoint,
    MemRefType,
    Operation,
    Type,
)

ROWS = 64


def build(kt):
    words_per_tile = ROWS * 16
    acc_type = T.vector(64, T.i32())
    word_type = T.vector(16, T.i32())
    row_type = T.vector(64, T.i8())
    generic_type = MemRefType.get(
        [kt * words_per_tile],
        T.i32(),
        memory_space=Attribute.parse("#ptr.generic_space"),
    )
    ptr_type = Type.parse("!ptr.ptr<#ptr.generic_space>")
    metadata_type = Type.parse(f"!ptr.ptr_metadata<{generic_type}>")

    # The ptr dialect has no dedicated Python bindings in this MLIR build.
    # Construct its operations with the generic Python operation builder.
    def ptr_add(ptr, offset):
        # All offsets stay within A. Leave the optional overflow flags unset;
        # the generic Python builder cannot set ptr's enum property.
        return Operation.create(
            "ptr.ptr_add", results=[ptr_type], operands=[ptr, offset]
        ).result

    dev = Device(AIEDevice.npu3)
    with InsertionPoint(Block.create_at_start(dev.body_region)):
        core_tile = tile(0, 2)
        a = buffer(core_tile, T.memref(kt * words_per_tile, T.i32()), name="a")
        c = buffer(core_tile, T.memref(ROWS * 64, T.i32()), name="c")
        core = Core(core_tile)
        with InsertionPoint(Block.create_at_start(core.body)):
            zero = arith.constant(T.index(), 0)
            one = arith.constant(T.index(), 1)
            last_kt = arith.constant(T.index(), kt - 1)
            tile_bytes = arith.constant(T.index(), words_per_tile * 4)
            port = arith.constant(T.i32(), 1)
            a_offsets = [arith.constant(T.index(), r * 64) for r in range(ROWS)]
            c_offsets = [arith.constant(T.index(), r * 64) for r in range(ROWS)]
            slots = [arith.constant(T.i32(), s) for s in range(32)]
            for slot in slots:
                stage_stream(INT8, port, slot, rhs_signed=True)

            poison = ub.poison(acc_type)
            generic_a = builtin.unrealized_conversion_cast([generic_type], [a])
            a_ptr = Operation.create(
                "ptr.to_ptr", results=[ptr_type], operands=[generic_a]
            ).result
            metadata = Operation.create(
                "ptr.get_metadata", results=[metadata_type], operands=[generic_a]
            ).result

            def ktile(ptr, accs, *, zero_if=None, prefetch=False):
                commit(INT8, rhs_signed=True)
                results = []
                for r in range(ROWS):
                    row_ptr = ptr_add(ptr, a_offsets[r])
                    row_memref = Operation.create(
                        "ptr.from_ptr",
                        results=[generic_type],
                        operands=[row_ptr, metadata],
                    ).result
                    words = vector.load(word_type, row_memref, [zero], alignment=64)
                    row = vector.bitcast(row_type, words)
                    results.append(
                        mmac_active(
                            INT8,
                            row,
                            accs[r],
                            zero_if=zero_if,
                            lhs_signed=True,
                            rhs_signed=True,
                        )
                    )
                    if prefetch and r & 1:
                        stage_stream(INT8, port, slots[r // 2], rhs_signed=True)
                return results

            loop = scf.ForOp(zero, last_kt, one, iter_args=[a_ptr] + [poison] * ROWS)
            with InsertionPoint(loop.body):
                ptr, *accs = loop.inner_iter_args
                first = arith.cmpi(
                    arith.CmpIPredicate.eq, loop.induction_variable, zero
                )
                updated = ktile(ptr, accs, zero_if=first, prefetch=True)
                scf.yield_([ptr_add(ptr, tile_bytes)] + updated)

            ptr, *accs = loop.results
            last = ktile(ptr, accs)
            for r, value in enumerate(last):
                vector.store(value, c, [c_offsets[r]], alignment=64)
            end()
        end()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kt", type=int, default=4, help="K-tiles (>= 2)")
    kt = parser.parse_args().kt
    assert kt >= 2
    with mlir_mod_ctx() as ctx:
        build(kt)
        ctx.module.operation.verify()
        print(ctx.module)


if __name__ == "__main__":
    main()
