# compiler.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""PythoC kernel compilation for IRON.

This module provides functions to compile PythoC kernels to AIE-compatible
object files that can be linked into IRON programs.
"""

import ast
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _find_pythoc_installation() -> Path:
    """Find PythoC installation directory.

    Returns:
        Path to PythoC installation

    Raises:
        RuntimeError: If PythoC cannot be found
    """
    # Try to import pythoc to find its location
    try:
        import pythoc

        pythoc_path = Path(pythoc.__file__).parent.parent
        return pythoc_path
    except ImportError:
        # Fall back to environment variable or default location
        pythoc_env = os.environ.get("PYTHOC_PATH")
        if pythoc_env:
            return Path(pythoc_env)

        # Try default location relative to this file
        default_path = Path(__file__).parent.parent.parent.parent.parent / "PythoC"
        if default_path.exists():
            return default_path

        raise RuntimeError(
            "PythoC installation not found. Please either:\n"
            "1. Install PythoC: pip install pythoc\n"
            "2. Set PYTHOC_PATH environment variable\n"
            "3. Ensure PythoC is at /scratch/jefff/acdc/PythoC/"
        )


def _get_llc_path() -> str:
    """Get path to llc (LLVM compiler) from AIE environment.

    Returns:
        Path to llc executable

    Raises:
        RuntimeError: If llc cannot be found
    """
    # Try llvm-aie installation first (required for AIE2 target support)
    llvm_aie_bin = os.environ.get("LLVM_AIE_BIN")
    if llvm_aie_bin:
        llc_path = Path(llvm_aie_bin) / "llc"
        if llc_path.exists():
            return str(llc_path)

    llvm_aie_bin = os.environ.get("PEANO_INSTALL_DIR")
    if llvm_aie_bin:
        llc_path = Path(llvm_aie_bin) / "bin" / "llc"
        if llc_path.exists():
            return str(llc_path)

    raise RuntimeError(
        "llvm-aie llc not found. Please ensure:\n"
        "1. LLVM_AIE_BIN environment variable is set, or\n"
        "2. PEANO_INSTALL_DIR environment variable is set\n"
        "Note: System llc does not support AIE2/AIE2P target"
    )


def compile_pythoc_kernel(
    kernel_path: str,
    function_name: str,
    target_arch: str = "aie2",
    output_dir: Optional[str] = None,
    optimization_level: int = 2,
    verbose: bool = False,
) -> Path:
    """Compile a PythoC kernel file to an AIE object file.

    Args:
        kernel_path: Path to PythoC kernel file (.py)
        function_name: Name of the function to compile
        target_arch: Target architecture (aie2, aie2p)
        output_dir: Directory for output files (default: temp directory)
        optimization_level: LLVM optimization level (0-3)
        verbose: Enable verbose output

    Returns:
        Path to compiled object file (.o)

    Raises:
        RuntimeError: If compilation fails
    """
    kernel_path = Path(kernel_path).resolve()
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    # Read kernel source
    with open(kernel_path, "r") as f:
        source_code = f.read()

    # Use compile_pythoc_source for the actual compilation
    return compile_pythoc_source(
        source_code=source_code,
        function_name=function_name,
        target_arch=target_arch,
        output_dir=output_dir,
        optimization_level=optimization_level,
        verbose=verbose,
    )


def compile_pythoc_source(
    source_code: str,
    function_name: str,
    target_arch: str = "aie2",
    output_dir: Optional[str] = None,
    optimization_level: int = 2,
    verbose: bool = False,
    extra_globals: Optional[dict] = None,
) -> Path:
    """Compile PythoC source code to an AIE object file.

    Args:
        source_code: PythoC source code as string
        function_name: Name of the function to compile
        target_arch: Target architecture (aie2, aie2p)
        output_dir: Directory for output files (default: temp directory)
        optimization_level: LLVM optimization level (0-3)
        verbose: Enable verbose output

    Returns:
        Path to compiled object file (.o)

    Raises:
        RuntimeError: If compilation fails
    """
    # Find PythoC installation
    pythoc_path = _find_pythoc_installation()

    # Add PythoC to Python path
    pythoc_parent = str(pythoc_path.parent)
    if pythoc_parent not in sys.path:
        sys.path.insert(0, pythoc_parent)

    # Import PythoC compiler
    try:
        from pythoc.compiler import LLVMCompiler
    except ImportError as e:
        raise RuntimeError(f"Failed to import PythoC: {e}")

    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="pythoc_iron_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set target triple for AIE
    target_triple = f"{target_arch}-none-unknown-elf"
    old_triple = os.environ.get("PYTHOC_TARGET_TRIPLE")
    os.environ["PYTHOC_TARGET_TRIPLE"] = target_triple

    try:
        # Parse source code
        tree = ast.parse(source_code)

        # Find the target function and remove @compile decorator
        func_node = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                func_node = node
                # Remove @compile decorator to avoid execution issues
                func_node.decorator_list = []
                break

        if func_node is None:
            raise ValueError(f"Function '{function_name}' not found in source code")

        # Build user globals by importing PythoC types
        # We need to provide the same namespace that the kernel file has
        user_globals = {}

        # Import all PythoC types and utilities
        try:
            import pythoc
            from pythoc import (
                compile,
                ptr,
                i8,
                i16,
                i32,
                i64,
                u8,
                u16,
                u32,
                u64,
                f16,
                f32,
                f64,
                bf16,
                void,
                bool as pbool,
            )
            from pythoc.aie import (
                aie_vector,
                load_v,
                store_v,
                vector_add,
                vector_mul,
                vector_sub,
                zeros,
                broadcast,
                vector_max,
                vector_min,
                vector_lt,
                vector_gt,
                vector_eq,
                vector_select,
                extract_elem,
                insert_elem,
                shift_bytes,
                vector_and,
                vector_or,
                concat,
                filter_even,
                filter_odd,
                interleave_unzip,
                accumulate,
                accum_to_vector,
                replicate_4x,
            )
            from pythoc.aie.loop_hints import prepare_for_pipelining, loop_range
            from pythoc.aie.profiling import event0, event1
            from pythoc.aie.utils import bitcast_i32_to_f32, fast_exp2_i32_to_f32
            from pythoc.aie.operations import read_tm, write_tm

            # Populate user_globals with all imported names
            user_globals.update(
                {
                    "compile": compile,
                    "ptr": ptr,
                    "i8": i8,
                    "i16": i16,
                    "i32": i32,
                    "i64": i64,
                    "u8": u8,
                    "u16": u16,
                    "u32": u32,
                    "u64": u64,
                    "f16": f16,
                    "f32": f32,
                    "f64": f64,
                    "bf16": bf16,
                    "void": void,
                    "bool": pbool,
                    "aie_vector": aie_vector,
                    "load_v": load_v,
                    "store_v": store_v,
                    "vector_add": vector_add,
                    "vector_mul": vector_mul,
                    "vector_sub": vector_sub,
                    "zeros": zeros,
                    "broadcast": broadcast,
                    "vector_max": vector_max,
                    "vector_min": vector_min,
                    "vector_lt": vector_lt,
                    "vector_gt": vector_gt,
                    "vector_eq": vector_eq,
                    "vector_select": vector_select,
                    "extract_elem": extract_elem,
                    "insert_elem": insert_elem,
                    "shift_bytes": shift_bytes,
                    "vector_and": vector_and,
                    "vector_or": vector_or,
                    "concat": concat,
                    "filter_even": filter_even,
                    "filter_odd": filter_odd,
                    "interleave_unzip": interleave_unzip,
                    "accumulate": accumulate,
                    "accum_to_vector": accum_to_vector,
                    "replicate_4x": replicate_4x,
                    "prepare_for_pipelining": prepare_for_pipelining,
                    "loop_range": loop_range,
                    "event0": event0,
                    "event1": event1,
                    "bitcast_i32_to_f32": bitcast_i32_to_f32,
                    "fast_exp2_i32_to_f32": fast_exp2_i32_to_f32,
                    "read_tm": read_tm,
                    "write_tm": write_tm,
                    "range": range,  # Add Python's built-in range for standard loop syntax
                }
            )
        except ImportError as e:
            raise RuntimeError(f"Failed to import PythoC types: {e}")

        # Merge caller-provided globals (e.g. regdb constants)
        if extra_globals:
            user_globals.update(extra_globals)

        # Create compiler instance with user globals
        compiler = LLVMCompiler(user_globals=user_globals)

        # Compile function to LLVM IR
        if verbose:
            print(f"Compiling {function_name} to LLVM IR...")

        llvm_func = compiler.compile_function_from_ast(
            func_node,
            source_code=source_code,
            reset_module=True,
            user_globals=user_globals,
        )

        # Optimize if requested
        if optimization_level > 0:
            if verbose:
                print(f"Optimizing with -O{optimization_level}...")
            compiler.optimize_module(optimization_level)

        # Save LLVM IR to file
        ll_file = output_dir / f"{function_name}.ll"
        ir_str = compiler.get_ir()
        with open(ll_file, "w") as f:
            f.write(ir_str)

        if verbose:
            print(f"LLVM IR saved to: {ll_file}")

        # Compile LLVM IR to object file using llc
        obj_file = output_dir / f"{function_name}.o"
        llc_path = _get_llc_path()

        llc_cmd = [
            llc_path,
            f"-march={target_arch}",
            "-filetype=obj",
            f"-o={obj_file}",
            str(ll_file),
        ]

        if verbose:
            print(f"Compiling LLVM IR to object file: {' '.join(llc_cmd)}")

        result = subprocess.run(llc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(f"llc compilation failed:\n{error_msg}")

        if verbose:
            print(f"Object file created: {obj_file}")

        return obj_file

    finally:
        # Restore original target triple
        if old_triple is not None:
            os.environ["PYTHOC_TARGET_TRIPLE"] = old_triple
        elif "PYTHOC_TARGET_TRIPLE" in os.environ:
            del os.environ["PYTHOC_TARGET_TRIPLE"]
