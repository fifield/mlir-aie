# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Peano tile-mapped intrinsics and helpers using tile register addresses.

The raw generated operations take core-visible LLVM pointers. read_tm/write_tm
instead accept a tile register offset, matching Peano's C++ read_tm/write_tm
wrappers with their default TM aperture. Barriers and processor-bus enablement
remain explicit. See docs/TileMappedAccess.md.
"""

from ._xllvm_ops_gen import *
from . import aie, arith, llvm
from ._ods_common import get_op_result_or_value
from ..ir import InsertionPoint, IntegerType, Operation, OpView, Value
from .._mlir_libs._aie import register_dialect
from .._mlir_libs import get_dialect_registry

register_dialect(get_dialect_registry())

_TM_OPS = {
    "aie2": (intr_aie2_read_tm, intr_aie2_write_tm, intr_aie2_sched_barrier),
    "aie2p": (intr_aie2p_read_tm, intr_aie2p_write_tm, intr_aie2p_sched_barrier),
    "aie4": (intr_aie4_read_tm, intr_aie4_write_tm, intr_aie4_sched_barrier),
}

# Tile register offsets reachable through the core processor bus, as
# [start, end), and the offset Peano's default TM aperture adds to them.
# AIE2/AIE2P: registers lie between the 64 KiB data memory and 0x80000, where
# the aperture starts (regdb; core-visible R + 0x80000 is a 20-bit address).
# AIE4: processor-bus addresses 0x48000-0x5CFFF cover the core module,
# memory module, and stream switch. Adding -0x58000 to tile-register offsets
# 0xA0000-0xB4FFF produces this processor-bus window.
_TM_WINDOWS = {
    "aie2": (0x10000, 0x80000, 0x80000),
    "aie2p": (0x10000, 0x80000, 0x80000),
    "aie4": (0xA0000, 0xB5000, -0x58000),
}

_DEVICE_ARCHS = {
    aie.AIEArch.AIE2: "aie2",
    aie.AIEArch.AIE2p: "aie2p",
    aie.AIEArch.AIE4: "aie4",
}


def _device_arch(ip):
    """Return the target arch of the enclosing aie.device, if any."""
    try:
        op = (ip or InsertionPoint.current).block.owner
    except ValueError:  # No insertion point.
        return None
    while op is not None:
        if isinstance(op.opview, aie.DeviceOp):
            model = aie.get_target_model(op.opview.device.value)
            return aie.AIEArch(model.get_target_arch())
        op = op.parent
    return None


def _arch(arch, ip):
    """Default arch to the enclosing device's target; reject a mismatch."""
    device = _device_arch(ip)
    device_name = None if device is None else _DEVICE_ARCHS.get(device)
    if device is not None and device_name is None:
        raise ValueError(f"tile-mapped access is unsupported on {device}")
    if arch is None:
        if device is None:
            raise ValueError("arch is required outside an aie.device")
        return device_name
    if arch not in _TM_OPS:
        raise ValueError("arch must be 'aie2', 'aie2p', or 'aie4'")
    if device_name is not None and arch != device_name:
        raise ValueError(
            f"arch '{arch}' does not match the enclosing device's '{device_name}'"
        )
    return arch


def _i32(value, *, loc=None, ip=None):
    if isinstance(value, int):
        if not -(1 << 31) <= value < (1 << 32):
            raise ValueError("value must fit in 32 bits")
        return arith.constant(IntegerType.get_signless(32), value, loc=loc, ip=ip)
    if isinstance(value, (OpView, Operation)) and len(value.results) == 1:
        value = get_op_result_or_value(value)
    if not isinstance(value, Value) or value.type != IntegerType.get_signless(32):
        raise TypeError("expected a Python integer or an MLIR i32 value")
    return value


def _tm_pointer(address, arch, *, loc=None, ip=None):
    start, end, offset = _TM_WINDOWS[arch]
    if isinstance(address, int) and (not start <= address < end or address % 4):
        raise ValueError(
            f"{arch} register address must be an aligned offset in "
            f"[{start:#x}, {end:#x})"
        )
    address = _i32(address, loc=loc, ip=ip)
    offset = _i32(offset, loc=loc, ip=ip)
    mapped = arith.addi(address, offset, loc=loc, ip=ip)
    return llvm.inttoptr(llvm.PointerType.get(), mapped, loc=loc, ip=ip)


def read_tm(address, *, arch=None, loc=None, ip=None):
    """Read a tile register through the core processor bus.

    Args:
        address (int | Value): Aligned tile register offset in the target's
            processor-bus window, or an i32 SSA value or single-result op.
        arch (str, optional): aie2, aie2p, or aie4. Defaults to the enclosing
            aie.device's target, which it must match.
        loc (Location, optional): Location of the emitted operations.
        ip (InsertionPoint, optional): Insertion point for the emitted operations.

    Returns:
        Value: The register's i32 value.
    """
    arch = _arch(arch, ip)
    ptr = _tm_pointer(address, arch, loc=loc, ip=ip)
    return _TM_OPS[arch][0](IntegerType.get_signless(32), ptr, loc=loc, ip=ip)


def write_tm(value, address, *, arch=None, loc=None, ip=None):
    """Write a tile register; operand order matches Peano's C++ intrinsic.

    Args:
        value (int | Value): Full 32-bit register value, or an i32 SSA value
            or single-result op.
        address (int | Value): Aligned tile register offset in the target's
            processor-bus window, or an i32 SSA value or single-result op.
        arch (str, optional): aie2, aie2p, or aie4. Defaults to the enclosing
            aie.device's target, which it must match.
        loc (Location, optional): Location of the emitted operations.
        ip (InsertionPoint, optional): Insertion point for the emitted operations.

    Returns:
        OpView: The register-write operation.
    """
    arch = _arch(arch, ip)
    value = _i32(value, loc=loc, ip=ip)
    ptr = _tm_pointer(address, arch, loc=loc, ip=ip)
    return _TM_OPS[arch][1](value, ptr, loc=loc, ip=ip)


def sched_barrier(*, arch=None, loc=None, ip=None):
    """Emit a compiler barrier; it does not synchronize cores or drain traffic.

    Args:
        arch (str, optional): aie2, aie2p, or aie4. Defaults to the enclosing
            aie.device's target, which it must match.
        loc (Location, optional): Location of the emitted operation.
        ip (InsertionPoint, optional): Insertion point for the emitted operation.

    Returns:
        OpView: The scheduling-barrier operation.
    """
    return _TM_OPS[_arch(arch, ip)][2](loc=loc, ip=ip)
