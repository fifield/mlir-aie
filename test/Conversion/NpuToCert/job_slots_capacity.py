# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s 64 | aie-opt -cert-legalize-pages -aie-cert-verify | FileCheck %s --check-prefix=FULL
# RUN: %PYTHON %s 65 | not aie-opt -cert-legalize-pages 2>&1 | FileCheck %s --check-prefix=OVER

# All input IDs alias slot 0. Allocation must use all 64 slots successfully,
# but report capacity exhaustion when there is no free slot.
# FULL: aiex.cert.job(0)
# FULL: aiex.cert.job(63)
# FULL-NOT: aiex.cert.job
# OVER: cert.page has 65 jobs, exceeding the 64 firmware job slots

import sys

print("aie.device(npu3) { aiex.cert.page {")
for i in range(int(sys.argv[1])):
    print(f"aiex.cert.job({i * 64}) {{}}")
print("} }")
