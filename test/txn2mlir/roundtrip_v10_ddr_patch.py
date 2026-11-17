# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A v1.0 transaction encodes a DDR patch as an 8-byte custom operation header
# followed by aie-codegen's 16-byte patch_op_opt_t: a 32-bit register address,
# an 8-bit argument index and a 64-bit argument offset. That is not the layout
# v0.1 uses, so each field must be read from its own offset and the next
# operation must start 24 bytes after the patch. An operation whose size is
# below its header is rejected, not misparsed.
#
# The transactions are built by hand below and fed to txn2mlir.py on stdin; the
# tool path comes from lit's txn2mlir.py substitution.

# RUN: %python %s txn2mlir.py patch | FileCheck %s
# RUN: %python %s txn2mlir.py patch_arg_plus_64bit | FileCheck %s --check-prefix=BIG
# RUN: %python %s txn2mlir.py patch_short | FileCheck %s --check-prefix=SHORT

# The write after the patch shows the patch advanced the parser by its size.
# CHECK: aie.device(npu1_1col)
# CHECK: %[[OFFSET:.*]] = arith.constant 256 : i32
# CHECK: aiex.npu.address_patch(%[[OFFSET]] : i32) {addr = 2211840 : ui32, arg_idx = 3 : i32}
# CHECK: %[[ADDR:.*]] = arith.constant 2211844 : i32
# CHECK: %[[VAL:.*]] = arith.constant 2 : i32
# CHECK: aiex.npu.write32(%[[ADDR]], %[[VAL]]) : i32, i32
# CHECK-NOT: aiex.npu

# BIG: %[[OFFSET:.*]] = arith.constant 8589934848 : i64
# BIG: aiex.npu.address_patch(%[[OFFSET]] : i64) {addr = 2211840 : ui32, arg_idx = 3 : i32}

# SHORT: Transaction op size 16 is smaller than its 24-byte header
# SHORT-NEXT: Failed to parse binary

import struct
import subprocess
import sys

DDR_PATCH, WRITE = 0x81, 0
ADDRESS, ARG_IDX, VALUE = 0x21C000, 3, 2


def transaction(arg_plus, op_size=24):
    """Build a two-operation npu1_1col v1.0 transaction.

    The first operation patches ADDRESS with argument ARG_IDX plus `arg_plus`;
    the second writes VALUE to the register after it.
    """
    # The opcode in a 4-byte word and the size of the whole operation, then
    # patch_op_opt_t: the register address, the argument index padded to a
    # word, and the 64-bit argument offset. Fields are naturally aligned.
    patch = struct.pack("<B3xIIB3xQ", DDR_PATCH, op_size, ADDRESS, ARG_IDX, arg_plus)
    patch = patch.ljust(op_size, b"\0")
    # A v1.0 write is the opcode, the 32-bit address and the value.
    write = struct.pack("<B3xII", WRITE, ADDRESS + 4, VALUE)
    body = patch + write
    # major, minor, device generation, rows, columns, memtile rows, padding,
    # then the number of operations and the size of the whole transaction.
    header = struct.pack("<8BII", 1, 0, 3, 6, 1, 1, 0, 0, 2, 16 + len(body))
    return header + body


CASES = {
    "patch": transaction(256),
    "patch_arg_plus_64bit": transaction((2 << 32) + 256),
    "patch_short": transaction(256, 16),
}

if __name__ == "__main__":
    tool, case = sys.argv[1:]
    result = subprocess.run(
        [sys.executable, tool, "-f", "-"],
        input=CASES[case],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stdout.buffer.write(result.stdout)
    sys.exit(result.returncode)
