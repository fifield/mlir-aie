# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The srs helper builds aievec.aie4.srs with checked arguments; an omitted
# mode means ambient.

from aie.extras import types as T
from aie.helpers.dialects.func import func
from aie.helpers.dialects.aievec_aie4 import (
    CONV_EVEN,
    POS_INF,
    SAT,
    SYM_SAT,
    TRUNCATE,
    srs,
)
from aie.dialects.aievec import AIE4RoundingMode, AIE4SaturationMode
from aie.ir import Module
from util import construct_and_print_module

# CHECK-LABEL: TEST: test_enum_strings
# CHECK: pos_inf 9
# CHECK: truncate 0
# CHECK: symmetric 3
print("\nTEST: test_enum_strings")
assert str(AIE4RoundingMode.PosInf) == "pos_inf"
print(POS_INF, int(POS_INF))
print(TRUNCATE, int(TRUNCATE))
print(SYM_SAT, int(SYM_SAT))
assert isinstance(SAT, AIE4SaturationMode)


# CHECK-LABEL: TEST: test_srs
# CHECK: aievec.aie4.srs %{{.*}} {isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>, shift = 12 : i32} : vector<64xi32> -> vector<64xi8>
# CHECK: aievec.aie4.srs %{{.*}} {isSigned = false, rounding = #aievec.aie4_rnd<conv_even>, shift = 0 : i32} : vector<64xi32> -> vector<64xi8>
# CHECK: aievec.aie4.srs %{{.*}} {isSigned = true, shift = 59 : i32} : vector<32xi32> -> vector<32xi16>
# CHECK: ROUND TRIP OK
@construct_and_print_module
def test_srs(module):
    i32 = T.i32()

    @func(emit=True)
    def narrow(a: T.vector(64, i32), b: T.vector(32, i32)):
        srs(a, shift=12, signed=True, rounding=POS_INF, saturation=SAT)
        srs(a, shift=0, signed=False, rounding=CONV_EVEN)
        srs(b, shift=59, signed=True)

    assert module.operation.verify()
    print(module)
    assert str(Module.parse(str(module))) == str(module)
    print("ROUND TRIP OK")
    return None


# CHECK-LABEL: TEST: test_srs_errors
# CHECK: ValueError: srs: shift must be in [0, 59], got 60
# CHECK: ValueError: srs: acc must be vector<64xi32> or vector<32xi32>, got vector<16xi32>
# CHECK: TypeError: srs: signed must be a bool, got 1
# CHECK: TypeError: srs: rounding must be an AIE4RoundingMode or None, got 9
# CHECK: TypeError: srs: saturation must be an AIE4SaturationMode or None, got 1
@construct_and_print_module
def test_srs_errors(module):
    i32 = T.i32()

    @func(emit=True)
    def bad(a: T.vector(64, i32), c: T.vector(16, i32)):
        for kwargs in (
            dict(acc=a, shift=60, signed=True),
            dict(acc=c, shift=4, signed=True),
            dict(acc=a, shift=4, signed=1),
            dict(acc=a, shift=4, signed=True, rounding=9),
            dict(acc=a, shift=4, signed=True, saturation=1),
        ):
            acc = kwargs.pop("acc")
            try:
                srs(acc, **kwargs)
                print("no error")
            except (ValueError, TypeError) as e:
                print(f"{type(e).__name__}: {e}")

    return None
