# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A v0.1 operation of its full size parses. An operation whose size is below its
# header is rejected, not misparsed. (A size of zero would never advance.) A
# DDR patch reads its 64-bit argument offset from bytes 40 to 47.
#
# The transactions are built by hand below and fed to txn2mlir.py on stdin; the
# tool path comes from lit's txn2mlir.py substitution.

# RUN: %python %s txn2mlir.py write | FileCheck %s
# RUN: %python %s txn2mlir.py write_short | FileCheck %s --check-prefix=WRITE
# RUN: %python %s txn2mlir.py maskwrite_short | FileCheck %s --check-prefix=MASKWRITE
# RUN: %python %s txn2mlir.py blockwrite_short | FileCheck %s --check-prefix=BLOCKWRITE
# RUN: %python %s txn2mlir.py sync_short | FileCheck %s --check-prefix=SYNC
# RUN: %python %s txn2mlir.py ddr_patch_short | FileCheck %s --check-prefix=DDR-PATCH

# CHECK: %[[ADDR:.*]] = arith.constant 2211840 : i32
# CHECK: %[[VAL:.*]] = arith.constant 2 : i32
# CHECK: aiex.npu.write32(%[[ADDR]], %[[VAL]]) : i32, i32

# WRITE: Transaction op size 20 is smaller than its 24-byte header
# WRITE-NEXT: Failed to parse binary
# MASKWRITE: Transaction op size 20 is smaller than its 28-byte header
# MASKWRITE-NEXT: Failed to parse binary
# BLOCKWRITE: Transaction op size 12 is smaller than its 16-byte header
# BLOCKWRITE-NEXT: Failed to parse binary
# SYNC: Transaction op size 12 is smaller than its 16-byte header
# SYNC-NEXT: Failed to parse binary
# DDR-PATCH: Transaction op size 44 is smaller than its 48-byte header
# DDR-PATCH-NEXT: Failed to parse binary

import struct
import subprocess
import sys

ADDRESS, VALUE, MASK = 0x21C000, 2, 6


def transaction(op, size):
    """Build a one-operation npu1_1col v0.1 transaction.

    The operation is laid out as TxnEncoding.h writes it, with its size field
    set to `size` and the operation padded with zeros up to that size.
    """
    if op == "write":
        # Opcode 0, the 64-bit address, the value and the operation size.
        body = struct.pack("<B7xQII", 0, ADDRESS, VALUE, size)
    elif op == "maskwrite":
        # Opcode 3, the 64-bit address, value, mask and the operation size.
        body = struct.pack("<B7xQIII", 3, ADDRESS, VALUE, MASK, size)
    elif op == "blockwrite":
        # Opcode 1, the column and row, the address, the operation size, then
        # the payload.
        body = struct.pack("<B7xII", 1, ADDRESS, size) + struct.pack("<I", VALUE)
    elif op == "sync":
        # Opcode 0x80 and the operation size, then the descriptor and config
        # words of a sync on column 0, row 0, direction 1, channel 2, one
        # column and row.
        config = (2 << 24) | (1 << 16) | (1 << 8)
        body = struct.pack("<B3xIII", 0x80, size, 1, config)
    else:
        # Opcode 0x81 and the operation size, op_base, the action padded to
        # 8 bytes, then the 64-bit register address, argument index and
        # argument offset.
        body = struct.pack("<B3xI8xI4xQQQ", 0x81, size, 0, ADDRESS, 1, 8)
    body = body.ljust(size, b"\0")
    # major, minor, device generation, rows, columns, memtile rows, padding,
    # then the number of operations and the size of the whole transaction.
    header = struct.pack("<8BII", 0, 1, 3, 6, 1, 1, 0, 0, 1, 16 + len(body))
    return header + body


CASES = {
    "write": transaction("write", 24),
    "write_short": transaction("write", 20),
    "maskwrite_short": transaction("maskwrite", 20),
    "blockwrite_short": transaction("blockwrite", 12),
    "sync_short": transaction("sync", 12),
    "ddr_patch_short": transaction("ddr_patch", 44),
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
