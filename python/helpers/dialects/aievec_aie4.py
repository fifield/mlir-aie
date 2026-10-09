# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checked builders for AIE4 staging, MMACs and matrix regions.

Formats are INT8, UINT8, INT4, BF16 and FP8. UINT8 means INT8 with
rhs_signed=False. Plain integer slots and ports become i32 constants.

`srs` narrows i32 accumulators with optional rounding and saturation modes.
"""

from ...dialects.aievec import (  # pyright: ignore[reportMissingImports]
    AIE4MatrixFormat,
    AIE4MatrixRegionOp,
    AIE4RoundingMode,
    AIE4SaturationMode,
    aie4_commit,
    aie4_matrix_yield,
    aie4_mmac,
    aie4_mmac_active,
    aie4_srs,
    aie4_stage,
    aie4_stage_stream,
)
from ...dialects.arith import (  # pyright: ignore[reportMissingImports]
    constant as _arith_constant,
)
from ...ir import (  # pyright: ignore[reportMissingImports]
    F32Type,
    InsertionPoint,
    IntegerType,
    Value,
    VectorType,
)

# ===----------------------------------------------------------------------=== #
# Matrix staging and MMAC
# ===----------------------------------------------------------------------=== #


class _Uint8:
    """INT8 with rhs_signed=False. A convenience, not a format enum."""

    def __repr__(self):
        return "UINT8"


INT8 = AIE4MatrixFormat.Int8
BF16 = AIE4MatrixFormat.Bf16
FP8 = AIE4MatrixFormat.Fp8
INT4 = AIE4MatrixFormat.Int4
UINT8 = _Uint8()

_INTEGER_FORMATS = (AIE4MatrixFormat.Int8, AIE4MatrixFormat.Int4)
_MMAC_COLUMNS = 64


def _format(fmt, rhs_signed, op):
    """Resolve UINT8 and check `fmt`. Returns (format enum, rhs_signed)."""
    if fmt is UINT8:
        if rhs_signed is True:
            raise ValueError(f"{op}: UINT8 implies rhs_signed=False")
        return AIE4MatrixFormat.Int8, False
    if not isinstance(fmt, AIE4MatrixFormat):
        raise TypeError(
            f"{op}: fmt must be INT8, UINT8, INT4, BF16 or FP8, got {fmt!r}"
        )
    return fmt, rhs_signed


def _signedness(fmt, name, value, op):
    """Integer formats require a bool; float formats reject the attribute."""
    if fmt in _INTEGER_FORMATS:
        if not isinstance(value, bool):
            raise ValueError(f"{op}: {fmt} requires {name}=True or False")
    elif value is not None:
        raise ValueError(f"{op}: {name} applies only to integer formats")
    return value


_SLOTS = range(32)
_PORTS = (0, 1)


def _i32(v, name, op, valid, *, loc=None, ip=None):
    """A plain int in `valid` becomes an i32 constant; a Value passes
    through unchecked."""
    if isinstance(v, Value):
        return v
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{op}: {name} must be an int or a Value, got {v!r}")
    if v not in valid:
        if isinstance(valid, range):
            expected = f"in [{valid.start}, {valid.stop})"
        else:
            expected = "one of " + ", ".join(str(x) for x in valid)
        raise ValueError(f"{op}: {name} must be {expected}, got {v}")
    return _arith_constant(IntegerType.get_signless(32), v, loc=loc, ip=ip)


def _acc_type(fmt):
    """vector<64xi32> for integer formats, vector<64xf32> for float ones."""
    if fmt in _INTEGER_FORMATS:
        element = IntegerType.get_signless(32)
    else:
        element = F32Type.get()
    return VectorType.get([_MMAC_COLUMNS], element)


def stage(fmt, chunk, slot, *, rhs_signed=None, loc=None, ip=None):
    """aievec.aie4.stage: write a 1024-bit chunk to staging slot `slot`."""
    fmt, rhs_signed = _format(fmt, rhs_signed, "stage")
    _signedness(fmt, "rhs_signed", rhs_signed, "stage")
    return aie4_stage(
        chunk,
        _i32(slot, "slot", "stage", _SLOTS, loc=loc, ip=ip),
        fmt,
        rhs_signed=rhs_signed,
        loc=loc,
        ip=ip,
    )


def stage_stream(fmt, port, slot, *, rhs_signed=None, loc=None, ip=None):
    """aievec.aie4.stage_stream: pop 1024 bits from input stream `port` into
    staging slot `slot`."""
    fmt, rhs_signed = _format(fmt, rhs_signed, "stage_stream")
    _signedness(fmt, "rhs_signed", rhs_signed, "stage_stream")
    return aie4_stage_stream(
        _i32(port, "port", "stage_stream", _PORTS, loc=loc, ip=ip),
        _i32(slot, "slot", "stage_stream", _SLOTS, loc=loc, ip=ip),
        fmt,
        rhs_signed=rhs_signed,
        loc=loc,
        ip=ip,
    )


def commit(fmt, *, rhs_signed=None, loc=None, ip=None):
    """aievec.aie4.commit: copy the staging image to the active matrix."""
    fmt, rhs_signed = _format(fmt, rhs_signed, "commit")
    _signedness(fmt, "rhs_signed", rhs_signed, "commit")
    return aie4_commit(fmt, rhs_signed=rhs_signed, loc=loc, ip=ip)


def mmac(fmt, x, acc=None, *, lhs_signed=None, rhs_signed=None, loc=None, ip=None):
    """aievec.aie4.mmac: one row times the enclosing matrix_region's matrix.
    Without `acc` the accumulator is zero."""
    fmt, rhs_signed = _format(fmt, rhs_signed, "mmac")
    _signedness(fmt, "lhs_signed", lhs_signed, "mmac")
    _signedness(fmt, "rhs_signed", rhs_signed, "mmac")
    return aie4_mmac(
        _acc_type(fmt),
        x,
        fmt,
        acc=acc,
        lhs_signed=lhs_signed,
        rhs_signed=rhs_signed,
        loc=loc,
        ip=ip,
    )


def mmac_active(
    fmt,
    x,
    acc=None,
    *,
    zero_if=None,
    lhs_signed=None,
    rhs_signed=None,
    loc=None,
    ip=None,
):
    """aievec.aie4.mmac_active: one row times the committed matrix.

    Without `acc` the accumulator is zero. `zero_if` (integer formats only,
    with `acc`) is an i1 Value or a bool; when true the accumulator is
    replaced by zeros at run time."""
    fmt, rhs_signed = _format(fmt, rhs_signed, "mmac_active")
    _signedness(fmt, "lhs_signed", lhs_signed, "mmac_active")
    _signedness(fmt, "rhs_signed", rhs_signed, "mmac_active")
    if zero_if is not None:
        if fmt not in _INTEGER_FORMATS:
            raise ValueError(
                "mmac_active: zero_if is integer-only; peel the first K tile"
            )
        if acc is None:
            raise ValueError(
                "mmac_active: zero_if requires acc; without acc the "
                "accumulator is always zero"
            )
        if isinstance(zero_if, bool):
            zero_if = _arith_constant(
                IntegerType.get_signless(1), int(zero_if), loc=loc, ip=ip
            )
        elif not isinstance(zero_if, Value):
            raise TypeError(
                f"mmac_active: zero_if must be a bool or an i1 Value, got {zero_if!r}"
            )
    return aie4_mmac_active(
        _acc_type(fmt),
        x,
        fmt,
        acc=acc,
        zero_if=zero_if,
        lhs_signed=lhs_signed,
        rhs_signed=rhs_signed,
        loc=loc,
        ip=ip,
    )


def matrix_region(b, inits, fmt, *, rhs_signed=None, results=None, loc=None, ip=None):
    """Decorator that builds an aievec.aie4.matrix_region around a body.

    The entry block takes the types of `inits`. The decorated function is
    called with the block arguments and returns the values to yield (a
    Value, a sequence of Values, or None). The result types are those of
    `inits` unless `results` lists them (e.g. a region without inits that
    yields a value). The decorator evaluates to the region's results: a
    Value for one result, a list for several, or the op for none.

        @matrix_region(b_tile, accs, INT8, rhs_signed=True)
        def out(*accs):
            return [mmac(INT8, rows[r], accs[r], lhs_signed=True,
                         rhs_signed=True) for r in range(len(accs))]
    """
    fmt, rhs_signed = _format(fmt, rhs_signed, "matrix_region")
    _signedness(fmt, "rhs_signed", rhs_signed, "matrix_region")
    inits = list(inits)

    def decorator(body):
        types = [v.type for v in inits]
        result_types = types if results is None else list(results)
        op = AIE4MatrixRegionOp(
            results_=result_types,
            rhs=b,
            inits=inits,
            format=fmt,
            rhs_signed=rhs_signed,
            loc=loc,
            ip=ip,
        )
        block = op.body.blocks.append(*types)
        with InsertionPoint(block):
            yielded = body(*block.arguments)
            if yielded is None:
                yielded = []
            elif isinstance(yielded, Value):
                yielded = [yielded]
            aie4_matrix_yield(list(yielded))
        values = op.results
        if len(values) == 1:
            return values[0]
        return list(values) if len(values) > 1 else op

    return decorator


# ===----------------------------------------------------------------------=== #
# SRS rounding and saturation
# ===----------------------------------------------------------------------=== #

# crRnd encodings.
FLOOR = AIE4RoundingMode.Floor
CEIL = AIE4RoundingMode.Ceil
SYM_FLOOR = AIE4RoundingMode.SymFloor
SYM_CEIL = AIE4RoundingMode.SymCeil
NEG_INF = AIE4RoundingMode.NegInf
POS_INF = AIE4RoundingMode.PosInf
SYM_ZERO = AIE4RoundingMode.SymZero
SYM_INF = AIE4RoundingMode.SymInf
CONV_EVEN = AIE4RoundingMode.ConvEven
CONV_ODD = AIE4RoundingMode.ConvOdd

# crSat encodings.
TRUNCATE = AIE4SaturationMode.Truncate
SAT = AIE4SaturationMode.Saturate
SYM_SAT = AIE4SaturationMode.Symmetric

# Accumulator lane count -> result element width.
_SRS_SHAPES = {64: 8, 32: 16}
_SRS_MAX_SHIFT = 59


def srs(
    acc,
    *,
    shift,
    signed,
    rounding=None,
    saturation=None,
    loc=None,
    ip=None,
) -> Value:
    """aievec.aie4.srs: vector<64xi32> -> vector<64xi8> or
    vector<32xi32> -> vector<32xi16>.

    `shift` is a right shift in 0..59. `signed` selects the signed or
    unsigned output range. `rounding` / `saturation` are an
    `AIE4RoundingMode` / `AIE4SaturationMode` (e.g. POS_INF, SAT) or None,
    which means ambient: the op reads the core's current mode.
    """
    acc_ty = acc.type
    lanes = None
    if isinstance(acc_ty, VectorType):
        elt = acc_ty.element_type
        if (
            acc_ty.rank == 1
            and not acc_ty.scalable
            and isinstance(elt, IntegerType)
            and elt.is_signless
            and elt.width == 32
            and acc_ty.shape[0] in _SRS_SHAPES
        ):
            lanes = acc_ty.shape[0]
    if lanes is None:
        raise ValueError(
            f"srs: acc must be vector<64xi32> or vector<32xi32>, got {acc_ty}"
        )
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ValueError(f"srs: shift must be an int in [0, 59], got {shift!r}")
    if not 0 <= shift <= _SRS_MAX_SHIFT:
        raise ValueError(f"srs: shift must be in [0, 59], got {shift}")
    if not isinstance(signed, bool):
        raise TypeError(f"srs: signed must be a bool, got {signed!r}")
    if rounding is not None and not isinstance(rounding, AIE4RoundingMode):
        raise TypeError(
            f"srs: rounding must be an AIE4RoundingMode or None, got {rounding!r}"
        )
    if saturation is not None and not isinstance(saturation, AIE4SaturationMode):
        raise TypeError(
            "srs: saturation must be an AIE4SaturationMode or None, got "
            f"{saturation!r}"
        )
    result_ty = VectorType.get([lanes], IntegerType.get_signless(_SRS_SHAPES[lanes]))
    return aie4_srs(
        result_ty,
        acc,
        shift,
        signed,
        rounding=rounding,
        saturation=saturation,
        loc=loc,
        ip=ip,
    )
