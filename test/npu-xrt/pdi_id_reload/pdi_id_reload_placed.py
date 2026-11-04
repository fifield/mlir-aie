# pdi_id_reload/pdi_id_reload_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

# Generates one ELF's design for the PDI id regression test.
#
#   python pdi_id_reload_placed.py --entry main_a --cfg cfga --k 1 > a.mlir
#
# The entry device @<entry> has one runtime sequence @seq that runs
#   aiex.configure @<cfg> { aiex.run @sequence }
# so the lowering emits one cert section @<cfg> and a LOAD_PDI for it.
# @<cfg> is shim -> core(0,2) -> shim with depth-1 objectFifos; the core loops
# forever writing out[i] = in[i] + K. Only K and the names vary between ELFs.
import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith as arith_d
from aie.dialects import memref as memref_d
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.ir import IntegerType

N = 64


def pdi_id_reload(entry, cfg, k):
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu3)
        def device_body():
            ty = np.ndarray[(N,), np.dtype[np.int32]]
            shim = tile(0, 0)
            core_tile = tile(0, 2)
            in_fifo = object_fifo("in", shim, core_tile, 1, ty)
            out_fifo = object_fifo("out", core_tile, shim, 1, ty)

            @core(core_tile)
            def core_body():
                i32 = IntegerType.get_signless(32)
                for _ in range_(sys.maxsize):
                    e_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                    e_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(N):
                        x = memref_d.load(e_in, [i])
                        y = arith_d.addi(x, arith_d.constant(i32, k))
                        memref_d.store(y, e_out, [i])
                    in_fifo.release(ObjectFifoPort.Consume, 1)
                    out_fifo.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(ty, ty)
            def sequence(A, C):
                a = shim_dma_single_bd_task(
                    in_fifo, A, sizes=[1, 1, 1, N], issue_token=True
                )
                c = shim_dma_single_bd_task(
                    out_fifo, C, sizes=[1, 1, 1, N], issue_token=True
                )
                dma_start_task(a, c)
                dma_await_task(a, c)

        assert ctx.module.operation.verify()
        text = str(ctx.module)

    # Name the configuration device and its sequence, then add the entry device.
    # Section names must be lower case (aiebu-asm lowercases label references).
    marker = "aie.device(npu3) {"
    assert text.count(marker) == 1, text
    text = text.replace(marker, "aie.device(npu3) @%s {" % cfg)
    assert text.count("aie.runtime_sequence(") == 1
    text = text.replace("aie.runtime_sequence(", "aie.runtime_sequence @sequence(")
    entry_dev = """  aie.device(npu3) @{entry} {{
    aie.runtime_sequence @seq(%a: memref<{n}xi32>, %c: memref<{n}xi32>) {{
      aiex.configure @{cfg} {{
        aiex.run @sequence(%a, %c) : (memref<{n}xi32>, memref<{n}xi32>)
      }}
    }}
  }}
""".format(entry=entry, cfg=cfg, n=N)
    last = text.rstrip().rfind("}")
    return text[:last] + entry_dev + "}\n"


p = argparse.ArgumentParser()
p.add_argument("--entry", required=True, help="entry device (kernel name) of the ELF")
p.add_argument("--cfg", required=True, help="lower-case name of the configuration")
p.add_argument("--k", type=int, required=True, help="the core computes in + k")
args = p.parse_args()
print(pdi_id_reload(args.entry, args.cfg, args.k))
