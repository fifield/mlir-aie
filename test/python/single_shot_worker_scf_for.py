# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.iron import Program, Runtime, Worker
from aie.iron.device import NPU1


def core_body():
    pass


worker = Worker(core_body, while_true=False)

rt = Runtime()
with rt.sequence():
    rt.start(worker)

module = Program(NPU1(), rt).resolve_program()
print(module)

# CHECK: aie.core
# CHECK: scf.for
