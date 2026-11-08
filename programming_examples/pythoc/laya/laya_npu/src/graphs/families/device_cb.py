"""device_cb (M36): fast B1/L64's backbone device. The B1-split dense engine (engine_cb.py, columns 0-3) and
native attention (attention_cb.py, columns 4-7) in one aie.device, loaded once for all 28 layers.

Before M36 the backbone alternated device_02 and device_05 in 57 configuration regions.

The two halves share no shim, memtile or route. Runtime sequences are lowered from the combined Program,
and each sequence drives only its own half:
- engine: cb_qkvR, cb_out, cb_ln, cb_gateup, cb_down
- attention: cb_attn

Every Runtime carries all 24 shim endpoints. The cores are built from each half alone, with the same
buffers and locks per core tile (core_templates()):
- dense_m2n2_cb: the engine half (build_dense.py)
- attention_special/l64_native_cb: the attention half (build_special.py)
"""

import json, re
from pathlib import Path

from aie.iron import Program, Runtime
from aie.iron.device import NPU2

from . import attention_cb as A
from . import engine_cb as E

ROOT = Path(__file__).resolve().parents[3]
A.COLUMNS, A.X0 = 4, 4


def variant():
    v = A.VARIANTS["device_05"]
    return A.Variant(v.L, v.family, v.skip, v.keys)


def program(kind=None, op=None):
    """device_cb as IRON MLIR with one engine op's sequence, the attention sequence, or (kind None) none."""
    ew, Ap, Bp, Yc = E.parts()
    v = variant()
    aw, inps, kvs, outs = A.parts(v)
    if kind == "attn":
        arg_ty, seq = A.seq(v, 1, inps, kvs, outs)
    else:
        arg_ty, seq = E.seq(op or "out", 64, 1, op != "ln", Ap, Bp, Yc, empty=kind is None)
    rt = Runtime(seq, arg_ty + Yc + Bp + Ap + inps + kvs + outs)
    return str(Program(NPU2(), rt, workers=ew + aw).resolve_program())


DEVICES = {"device_cb": program}
SEQUENCES = [(f"cb_{op}", "exact", lambda op=op: program("eng", op)) for op in ("qkvR", "out", "ln", "gateup", "down")]
SEQUENCES += [("cb_attn", "exact", lambda op=None: program("attn"))]


def engine_core_template():
    """dense_cb: the engine half alone, with build_dense.py's object placeholders."""
    return E.build().replace('"laya_gemm_native.o"', '"@LAYA_GEMM_NATIVE@"').replace(
        '"bf16_zero_kernel.o"', '"@BF16_ZERO_KERNEL@"').replace('"engine.o"', '"@ENGINE@"')


def attention_core_template():
    """l64_native_cb: the attention half alone with l64_native's link placeholders (src/special/recipes.json)."""
    bindings = json.loads((ROOT / "src/special/recipes.json").read_text())["devices"]["l64_native"]["bindings"]
    by_name = {key.strip("@").rsplit("_", 1)[0]: key for key in bindings}
    text = A.program(variant())  # COLUMNS = 4, X0 = 4 (set above)

    def link(m):
        return f'func.func private @{m[1]}({m[2]}) attributes {{link_with = "{by_name[m[1]]}"}}'

    text = re.sub(r'func\.func private @(\w+)\((.*)\) attributes \{link_with = "attention\.o"\}', link, text)
    assert "attention.o" not in text
    return text
