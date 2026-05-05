# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %python %s | FileCheck %s

# Verifies that aie.iron._loc.capture_user_loc:
#   1. captures frame info eagerly (no MLIR Context required)
#   2. returns a CapturedLoc whose filename matches the caller
#   3. wraps the file/line in a NameLoc when materialized with a name
#   4. plays nicely as a `with` block via loc_or_unknown

import os
from aie.iron._loc import capture_user_loc, loc_or_unknown, CapturedLoc
from aie.ir import Context, Location


def main():
    # 1. Outside a Context, capture still works (eager).
    assert Context.current is None
    captured = capture_user_loc()
    assert isinstance(captured, CapturedLoc), f"got {type(captured).__name__}"
    assert os.path.basename(captured.filename) == "loc_capture.py"
    assert captured.line > 0
    assert captured.name is None
    print(f"CHECK-1: eager capture: {os.path.basename(captured.filename)}:{captured.line}")

    # 2. With a name, the name field is set; only materialized later.
    named_capture = capture_user_loc(name="my_name")
    assert named_capture.name == "my_name"
    print(f"CHECK-2: named capture: name={named_capture.name}")

    # 3. Inside a Context, materialize() builds a real Location and
    #    `loc_or_unknown(captured)` enters it as a context manager.
    with Context():
        loc = captured.materialize()
        s = str(loc)
        assert "loc_capture.py" in s, f"unexpected: {s}"
        print(f"CHECK-3: materialized = {s}")

        # NameLoc wrapping
        named_loc = named_capture.materialize()
        s2 = str(named_loc)
        assert s2.startswith('loc("my_name"'), s2
        print(f"CHECK-4: named materialized = {s2}")

        # `with loc_or_unknown(None):` falls back to unknown; the with
        # block's value is a real Location either way.
        with loc_or_unknown(None) as fallback:
            assert "unknown" in str(fallback)
        with loc_or_unknown(captured) as active:
            assert "loc_capture.py" in str(active)
        print("CHECK-5: loc_or_unknown context manager works")

    print("ALL OK")


# CHECK: CHECK-1: eager capture: loc_capture.py:{{[0-9]+}}
# CHECK: CHECK-2: named capture: name=my_name
# CHECK: CHECK-3: materialized = loc("{{.*}}loc_capture.py":{{[0-9]+}}:{{[0-9]+}})
# CHECK: CHECK-4: named materialized = loc("my_name"("{{.*}}loc_capture.py":{{[0-9]+}}:{{[0-9]+}}))
# CHECK: CHECK-5: loc_or_unknown context manager works
# CHECK: ALL OK

if __name__ == "__main__":
    main()
