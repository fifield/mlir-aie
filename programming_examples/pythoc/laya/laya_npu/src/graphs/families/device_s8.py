"""device_s8 (M37): the B1-split dense engine on all 8 columns (engine_s8.py), in place of device_02 for the B1/L64
graphs that keep the engine on its own device: exact/b1_l64, pair/exact/b1_l64 and pair/fast/b1_l64.

At B = 1 device_02's two 64-row halves both compute batch row 0. device_s8 gives them different weight tiles, so
every core runs half of device_02's tiles; the outputs are bit-identical. Runtime sequences s8_qkvR, s8_out,
s8_ln, s8_gateup, s8_down replace sequence_001..005 (same operand types except the 96-tile gate/up weight).

Cores (32, built by build_dense.py from core_template(), with the RoPE rule 3 j < 2 ntile):
- dense_native_s8: device_02's dense_native (exact)
- dense_fast_s8: device_02's dense_fast (the fast pair's n2 leaf)
"""

from . import engine_s8 as E

DEVICES = {"device_s8": E.build}
SEQUENCES = [(f"s8_{op}", "exact", lambda op=op: E.build(op, 64, 1, op != "ln"))
             for op in ("qkvR", "out", "ln", "gateup", "down")]


def core_template():
    """dense_s8: the device with build_dense.py's object placeholders."""
    return E.build().replace('"laya_gemm_native.o"', '"@LAYA_GEMM_NATIVE@"').replace(
        '"bf16_zero_kernel.o"', '"@BF16_ZERO_KERNEL@"').replace('"engine.o"', '"@ENGINE@"')
