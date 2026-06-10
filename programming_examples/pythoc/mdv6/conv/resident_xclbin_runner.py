#!/usr/bin/env python3
"""Llama-style cached XRT runner for MDV6 xclbin+insts kernels.

The stock `DefaultNPURuntime.run()` already supports cached XRT contexts via
`CachedXRTRuntime`, but it is tensor-call oriented: each call normally receives
fresh tensors, so argument BO data is rewritten every launch.  For MDV6 fused
block work we want the llama32_1b pattern:

- load one xclbin/insts once;
- keep stable BOs per layer/block key;
- write static weight BOs once;
- reuse intermediate/output BOs as scratch;
- read back only requested outputs.

This helper is intentionally small and xclbin-based so existing IRON/aiecc
artifacts can use it before we have an ELF-based KernelCache like llama32_1b.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyxrt

from aie.utils.hostruntime.xrtruntime.hostruntime import CachedXRTRuntime
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel


@dataclass
class ResidentRunStats:
    write_ms: float
    kernel_ms: float
    read_ms: float
    total_ms: float
    n_written: int
    bytes_written: int
    n_read: int
    first_call: bool


class ResidentXCLBinRunner:
    """Cached-BO runner around a compiled xclbin/insts pair.

    Args are numpy arrays.  The first call for a `bo_key` allocates BOs and writes
    every non-output argument.  Later calls skip indices in `static_indices` and
    `intermediate_indices`, matching the llama cache semantics.  Mutable input
    indices are rewritten each call; output indices are only read back.
    """

    def __init__(self, xclbin: str | Path, insts: str | Path, *, kernel_name: str = "MLIR_AIE"):
        self.xclbin = str(Path(xclbin).resolve())
        self.insts = str(Path(insts).resolve())
        self.kernel_name = kernel_name
        self.runtime = CachedXRTRuntime()
        self.handle = self.runtime.load(NPUKernel(self.xclbin, self.insts, kernel_name=kernel_name))
        self._bo_cache: dict[str, list[XRTTensor]] = {}
        self.last_stats: ResidentRunStats | None = None

    def close(self):
        self._bo_cache.clear()
        self.runtime.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @staticmethod
    def _as_array(a) -> np.ndarray:
        arr = np.asarray(a)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    def _allocate(self, key: str, arrays: list[np.ndarray]) -> list[XRTTensor]:
        tensors = [XRTTensor(a.shape, dtype=a.dtype, device="npu") for a in arrays]
        self._bo_cache[key] = tensors
        return tensors

    def run(
        self,
        *args,
        bo_key: str = "default",
        output_indices: Iterable[int] | None = None,
        static_indices: Iterable[int] | None = None,
        intermediate_indices: Iterable[int] | None = None,
        inout_indices: Iterable[int] | None = None,
    ):
        arrays = [self._as_array(a) for a in args]
        if not arrays:
            raise ValueError("at least one argument is required")
        out_set = {len(arrays) - 1} if output_indices is None else set(output_indices)
        static_set = set(static_indices or [])
        interm_set = set(intermediate_indices or [])
        # inout BOs are synced to device before launch (unlike pure outputs)
        # AND read back after — e.g. a DDR bounce image the design fills from
        # and drains into.
        inout_set = set(inout_indices or [])
        out_set |= inout_set

        first_call = bo_key not in self._bo_cache
        if first_call:
            tensors = self._allocate(bo_key, arrays)
        else:
            tensors = self._bo_cache[bo_key]
            if len(tensors) != len(arrays):
                raise ValueError(f"bo_key {bo_key!r} has {len(tensors)} args, got {len(arrays)}")
            for i, (t, a) in enumerate(zip(tensors, arrays)):
                if t.shape != a.shape or t.dtype != a.dtype:
                    raise ValueError(
                        f"bo_key {bo_key!r} arg {i} shape/dtype mismatch: "
                        f"cached {t.shape}/{t.dtype}, got {a.shape}/{a.dtype}"
                    )

        t_write0 = time.perf_counter()
        n_written = 0
        bytes_written = 0
        for i, (tensor, arr) in enumerate(zip(tensors, arrays)):
            if i in out_set and i not in inout_set:
                continue
            if not first_call and (i in static_set or i in interm_set):
                continue
            np.copyto(tensor.data.reshape(arr.shape), arr, casting="no")
            tensor._sync_to_device()
            n_written += 1
            bytes_written += arr.nbytes
        write_ms = (time.perf_counter() - t_write0) * 1000.0

        bos = [t.buffer_object() for t in tensors]
        insts = self.handle.insts
        is_module = hasattr(pyxrt, "module") and isinstance(insts, pyxrt.module)
        if is_module:
            insts_bo = None
            insts_bytes = 0
        else:
            insts_bo = self.handle.insts_bo
            insts_bytes = int(getattr(insts, "nbytes"))

        t_kernel0 = time.perf_counter()
        h = self.handle.kernel(3, insts_bo, insts_bytes, *bos)
        ret = h.wait()
        kernel_ms = (time.perf_counter() - t_kernel0) * 1000.0
        if ret != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"Kernel returned {ret}")

        t_read0 = time.perf_counter()
        results = []
        for i, tensor in enumerate(tensors):
            if i in out_set:
                tensor._sync_from_device()
                results.append(np.array(tensor.data, copy=True))
            else:
                results.append(np.empty(0, dtype=arrays[i].dtype))
        read_ms = (time.perf_counter() - t_read0) * 1000.0

        self.last_stats = ResidentRunStats(
            write_ms=write_ms,
            kernel_ms=kernel_ms,
            read_ms=read_ms,
            total_ms=write_ms + kernel_ms + read_ms,
            n_written=n_written,
            bytes_written=bytes_written,
            n_read=len(out_set),
            first_call=first_call,
        )
        return tuple(results)
