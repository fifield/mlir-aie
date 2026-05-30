# inline.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

"""Write a PythoC kernel body *literally inside* an ``aie.core`` (Stage 2).

``@pythoc_inline`` lets you write the compute as a nested, type-annotated PythoC
function right in the traced core body, interleaved with the objectfifo
acquire/release dataflow. Under the hood it lifts that function into a synthetic
kernel, compiles it through the Stage-1 inline path (``alwaysinline`` ``.ll``),
declares the ``func.func`` at device scope, and emits an inlined call at the
current point in the trace. aiecc then llvm-links the IR into the core and
inlines it -- so the body is folded directly into the ``aie.core`` with no
``func.call`` boundary and no separate object file.

Usage::

    def core_fn(of_in, of_out, t1, t2):
        a = of_in.acquire(1)
        o = of_out.acquire(1)

        @pythoc_inline(a, o, t1, t2, height=H, width=W, out_channels=OC)
        def _compute(inp: ptr[bf16, True], out: ptr[bf16, True],
                     tmp1: ptr[bf16, True], tmp2: ptr[bf16, True]):
            event0()
            oc: i32 = 0
            while oc < out_channels:      # baked as a compile-time constant
                ...
            event1()

        of_in.release(1)
        of_out.release(1)

- Positional args are the buffers to pass. Their MLIR memref types drive the
  kernel's MLIR signature; the function's own ``ptr[...]`` annotations drive the
  compute semantics -- the two meet at opaque pointers, so e.g. a ``uint16``
  objectfifo can be processed as ``bf16``.
- Keyword args are exposed to the body as compile-time constants (ints/floats)
  or symbols (extern helpers), matching ``PythocKernel(extra_globals=...)``.
  Symbols visible in the defining module (e.g. ``invsqrt``, ``getTanhBf16``) are
  picked up automatically; pass runtime-derived constants explicitly.
"""

def _detect_target_arch(device_op) -> str:
    """Peano target arch from the aie.device op's device enum (npu2* -> aie2p).

    The op prints in generic form (``device = 9 : i32``), so we map the
    AIEDevice enum integer rather than parsing a device name.
    """
    try:
        from ...dialects.aie import AIEDevice

        val = int(str(device_op.attributes["device"]).split(":")[0].strip())
        aie2p_vals = {
            int(getattr(AIEDevice, n)) for n in dir(AIEDevice) if n.startswith("npu2")
        }
        return "aie2p" if val in aie2p_vals else "aie2"
    except Exception:
        return "aie2p"


def pythoc_inline(*buffers, **kwargs):
    """Decorator: inline a nested PythoC function into the current ``aie.core``.

    Args:
        *buffers: MLIR buffer values (e.g. from ``of.acquire(1)``) passed to the
            kernel, in the order of the decorated function's parameters.
        target_arch: Optional Peano arch override ("aie2"/"aie2p"); auto-detected
            from the device when omitted.
        **kwargs: Compile-time constants / extern symbols made available to the
            kernel body (in addition to the defining module's globals).
    """
    target_arch = kwargs.pop("target_arch", None)
    const_globals = kwargs

    def deco(fn):
        # Imports are deferred to call time (trace time) to avoid a circular
        # import: this module is imported during aie.iron package init, before
        # the sibling modules below are fully available.
        from ... import ir  # type: ignore  (the MLIR `aie.ir` module)
        from .decorators import aie_kernel
        from .kernel import PythocKernel

        # 1. Capture source/name as an @aie_kernel so PythocKernel can compile it.
        kfn = aie_kernel(fn)

        # 2. Exact MLIR signature taken from the buffers' memref types, so the
        #    declaration matches the call operands precisely (opaque pointers).
        arg_types = [b.type for b in buffers]

        # 3. Symbols available to the body: the defining module's globals
        #    (PythoC imports + extern helpers) overlaid with explicit constants.
        extra = {k: v for k, v in fn.__globals__.items() if not k.startswith("__")}
        extra.update(const_globals)

        # 4. Walk up from the current core insertion point to the enclosing
        #    aie.device op (by op name, robust to OpView class identity).
        owner = ir.InsertionPoint.current.block.owner
        op = owner.operation if hasattr(owner, "operation") else owner
        while op is not None and op.name != "aie.device":
            op = op.parent
        if op is None:
            raise RuntimeError("pythoc_inline must be used inside an aie.core body")
        device_op = op
        arch = target_arch or _detect_target_arch(device_op)

        # 4. Stage-1 inline compile -> alwaysinline/linkonce_odr .ll kernel.
        kernel = PythocKernel(
            kfn, arg_types, target_arch=arch, extra_globals=extra, inline=True
        )

        # 5. Declare func.func at device scope, then emit the call here in the
        #    core. aiecc llvm-links the .ll and the inliner folds it in (Stage 1).
        device_block = device_op.regions[0].blocks[0]
        with ir.InsertionPoint.at_block_begin(device_block):
            kernel.resolve()
        kernel(*buffers)
        return kernel

    return deco
