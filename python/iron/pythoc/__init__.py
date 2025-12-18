# pythoc/__init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""IRON integration for PythoC kernels.

This module enables using PythoC-compiled kernels in IRON programs.
PythoC is a Python-to-LLVM compiler that generates AIE-compatible object files.
"""

from .compiler import compile_pythoc_kernel, compile_pythoc_source
from .decorators import aie_kernel
from .kernel import PythocKernel
from .types import pythoc_to_numpy_type, infer_kernel_signature

__all__ = [
    'compile_pythoc_kernel',
    'compile_pythoc_source',
    'aie_kernel',
    'PythocKernel',
    'pythoc_to_numpy_type',
    'infer_kernel_signature',
]
