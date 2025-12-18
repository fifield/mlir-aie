# kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""PythoC kernel wrapper for IRON.

Provides PythocKernel class that integrates PythoC-compiled kernels
into IRON programs.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from ..kernel import Kernel


class PythocKernel(Kernel):
    """IRON Kernel wrapper for PythoC-compiled functions.

    This class extends IRON's Kernel to support PythoC-compiled kernels.
    In Phase 1, it accepts LLVM IR files (.ll) from PythoC kernels.
    The IRON build system will compile .ll to .o using Peano toolchain.
    In Phase 2, it will also support inline decorated functions.

    Example (Phase 1 - External):
        from aie.iron.pythoc import compile_pythoc_kernel, PythocKernel

        # Compile PythoC kernel to LLVM IR
        ll_file = compile_pythoc_kernel(
            "pythoc_kernels/mul.py",
            "eltwise_mul_vectorized_i32"
        )

        # Create IRON kernel (IRON will compile .ll to .o)
        kernel = PythocKernel(
            "eltwise_mul_vectorized_i32",
            str(ll_file),
            [tile_ty, tile_ty, tile_ty]
        )
    """

    def __init__(
        self,
        kernel_fn_or_name: str,
        llvm_ir_path_or_types: str | list,
        types: Optional[list[type[np.ndarray] | np.dtype]] = None,
        target_arch: str = "aie2p",
    ):
        """Initialize PythocKernel.

        Args:
            kernel_fn_or_name: Function name (Phase 1) or decorated function (Phase 2)
            llvm_ir_path_or_types: Path to LLVM IR file (.ll) (Phase 1) or type list (Phase 2)
            types: Type signature (Phase 1 only, when llvm_ir_path is string)
            target_arch: Target architecture (aie2, aie2p)

        Raises:
            NotImplementedError: If trying to use Phase 2 features
            TypeError: If arguments are invalid
        """
        # Phase 1: External LLVM IR file
        if isinstance(kernel_fn_or_name, str):
            if not isinstance(llvm_ir_path_or_types, str):
                raise TypeError(
                    "For external kernels, llvm_ir_path_or_types must be a string path"
                )
            if types is None:
                raise TypeError("For external kernels, types parameter is required")

            # Validate LLVM IR file exists
            ll_path = Path(llvm_ir_path_or_types)
            if not ll_path.exists():
                raise FileNotFoundError(f"LLVM IR file not found: {ll_path}")

            # Call parent Kernel constructor
            # Note: bin_name should be the .ll file - IRON's Peano compilation
            # will handle converting .ll to .o during the build process
            super().__init__(
                name=kernel_fn_or_name, bin_name=str(ll_path), arg_types=types
            )

            self._target_arch = target_arch
            self._is_pythoc = True
            self._is_llvm_ir = True

        else:
            # Phase 2: Inline decorated function
            if not hasattr(kernel_fn_or_name, "__aie_kernel__"):
                raise TypeError(
                    "kernel_fn_or_name must be either a string (function name) "
                    "or a function decorated with @aie_kernel"
                )

            # Import compile_pythoc_source here to avoid circular imports
            from .compiler import compile_pythoc_source

            # Compile the inline function to LLVM IR
            ll_file = compile_pythoc_source(
                source_code=kernel_fn_or_name.__aie_source__,
                function_name=kernel_fn_or_name.__aie_name__,
                target_arch=target_arch,
                output_dir=None,
                optimization_level=2,
                verbose=False,
            )

            # Validate types parameter
            if not isinstance(llvm_ir_path_or_types, list):
                raise TypeError(
                    "For inline kernels, llvm_ir_path_or_types must be a list of types"
                )

            # Call parent Kernel constructor with compiled LLVM IR
            super().__init__(
                name=kernel_fn_or_name.__aie_name__,
                bin_name=str(ll_file),
                arg_types=llvm_ir_path_or_types,
            )

            self._target_arch = target_arch
            self._is_pythoc = True
            self._is_llvm_ir = True
            self._is_inline = True
