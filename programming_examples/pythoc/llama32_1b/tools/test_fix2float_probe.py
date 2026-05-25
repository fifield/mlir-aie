#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Probe each step of the Fix2Float u8->bf16 chain on AIE2P.

The AIE-API recipe (aie_api/detail/aie2p/elementary.hpp:51-58) is:
  1. Zero-extend <N x u8> to <N x i32> (acc32 representation)
  2. Add magic_l constant (0x4b010000 broadcast)
  3. Bitcast <N x i32> to <N x float>     (= accfloat representation)
  4. Subtract magic_l as accfloat float    (= 8454144.0)
  5. Convert <N x float> accfloat to <N x bf16>

Step 1 needs `unpack_unsigned` (UPS intrinsic) because G_ZEXT on vectors
won't legalize on AIE2P. Steps 2/3/4/5 may or may not legalize at llc.

This script tries each step incrementally. Build up until we hit a wall.
"""

import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

KERNEL_STEP_2 = '''
"""Step 2: vector i32 add + magic constant (plain LLVM `add` on `<32 x i32>`)."""
from aie.iron.pythoc import aie_kernel
from pythoc import bf16, i32, ptr, u8, u32, void
from pythoc.aie import aie_vector, load_v, store_v, unpack_unsigned

@aie_kernel
def step2(src: ptr[u8, True], dst: ptr[i32, True]) -> void:
    u8_v: aie_vector[u8, 32] = load_v(src, 32)
    # Step 1: u8 -> i16 -> i32 via UPS
    i16_v: aie_vector[i32, 32] = unpack_unsigned(u8_v, i32)  # may fail if i8->i32 not direct
    # Step 2: vector add with constant
    magic: aie_vector[i32, 32] = aie_vector[i32, 32]()  # uninit -- replace
    # Replace with broadcast
'''

KERNEL_STEP_4_BITCAST_AND_FSUB = '''
"""Step 3-5: zext -> bitcast int->float -> fsub -> accfloat_to_bf16."""
from aie.iron.pythoc import aie_kernel
from pythoc import bf16, f32, i32, ptr, u8, void
from pythoc.aie import (
    aie_vector,
    accfloat_to_bf16,
    broadcast,
    load_v,
    store_v,
    unpack_unsigned,
    vector_add,
    vector_cast,
    vector_sub,
)

# Magic constant: bf16(0x4b01) interpreted as accfloat float is
# 8454144.0; equivalently the i32 bit pattern 0x4b010000.
MAGIC_L_F32: f32

from pythoc import i16
from pythoc.aie import v32accfloat_to_v32bf16, I512_I512_ACC1024_bf_msc_conf

# Magic constant as bf16: 0x4b01 reinterpreted as bf16 == 8454144.0
MAGIC_L_BF: bf16

@aie_kernel
def fix2float(src: ptr[u8, True], dst: ptr[bf16, True]) -> void:
    u8_v: aie_vector[u8, 32] = load_v(src, 32)
    # Step 1: u8 -> i16 -> i32 (UPS chain avoids G_ZEXT)
    i16_v: aie_vector[i16, 32] = unpack_unsigned(u8_v, i16)
    i32_v: aie_vector[i32, 32] = unpack_unsigned(i16_v, i32)
    # Step 2: vector add magic_l_as_i32 (= 0x4b010000)
    magic_i32: aie_vector[i32, 32] = broadcast(i32, 32, i32(0x4b010000))
    sum_i32: aie_vector[i32, 32] = vector_add(i32_v, magic_i32)
    # Step 3: bitcast <32 x i32> -> <32 x f32> (= accfloat representation)
    sum_acc: aie_vector[f32, 32] = vector_cast(sum_i32, f32, 32)
    # Step 4: subtract magic via MSC (acc = sum_acc - magic_l_bf16 * 1.0)
    #   MSC computes acc - a*b. Setting a = magic_l_bf16, b = 1.0.
    magic_bf: aie_vector[bf16, 32] = broadcast(bf16, 32, MAGIC_L_BF)
    ones_bf: aie_vector[bf16, 32] = broadcast(bf16, 32, bf16(1.0))
    conf: i32 = i32(60)  # per-lane bf16 MAC mode (same as kernels/matvec.py)
    diff_acc: aie_vector[f32, 32] = I512_I512_ACC1024_bf_msc_conf(
        magic_bf, ones_bf, sum_acc, conf
    )
    # Step 5: accfloat -> bf16
    bf_v: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(diff_acc)
    store_v(dst, bf_v)
'''


def compile_and_inspect(src: str, fn: str, label: str, out_dir: Path,
                        extras: dict) -> bool:
    from aie.iron.pythoc.compiler import compile_pythoc_source
    print(f"\n--- {label}: compile {fn} ---")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        compile_pythoc_source(
            source_code=src,
            function_name=fn,
            target_arch="aie2p",
            output_dir=str(out_dir),
            extra_globals=extras,
        )
    except Exception as e:
        # Surface full error; opt.ll may or may not have been written.
        print(f"  FAIL: {type(e).__name__}")
        for line in str(e).splitlines():
            print(f"    | {line}")
        opt_ll = out_dir / f"{fn}.opt.ll"
        if opt_ll.exists():
            print(f"  (opt.ll still landed at {opt_ll})")
        return False
    print(f"  OK: {fn}.o compiled")
    return True


def main() -> int:
    from pythoc.aie import unpack_unsigned, broadcast, vector_cast
    from pythoc.aie import v32accfloat_to_v32bf16

    out_dir = Path(os.environ.get("PYTHOC_DEBUG_DIR", tempfile.mkdtemp(prefix="fix2f_")))

    from pythoc.aie import vector_add, vector_sub, I512_I512_ACC1024_bf_msc_conf
    import ml_dtypes
    extras = dict(
        unpack_unsigned=unpack_unsigned,
        broadcast=broadcast,
        vector_cast=vector_cast,
        vector_add=vector_add,
        vector_sub=vector_sub,
        v32accfloat_to_v32bf16=v32accfloat_to_v32bf16,
        I512_I512_ACC1024_bf_msc_conf=I512_I512_ACC1024_bf_msc_conf,
        MAGIC_L_F32=8454144.0,
        MAGIC_L_BF=ml_dtypes.bfloat16(8454144.0),
    )

    compile_and_inspect(KERNEL_STEP_4_BITCAST_AND_FSUB, "fix2float",
                        "Steps 1-5 (full chain)", out_dir, extras)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
