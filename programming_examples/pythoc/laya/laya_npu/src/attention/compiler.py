"""M9b: ``compile_pythoc_source`` of the installed toolchain (``aie/iron/pythoc/compiler.py``), copied with one added
parameter, ``noinline``: the named helper functions get LLVM's ``noinline`` attribute, so E1's ``exp_e1c`` stays one
real function in every kernel object (the G7 on-core proof covers its bytes; an inlined copy would not be the proved
code). Everything else is the installed function verbatim (its private helpers are imported from the installed module).
With ``noinline=()`` it compiles exactly what the installed function compiles (checked by ``m9b.build``'s recompile
control on the accepted S2 text).
"""

import ast, os, re, shlex, subprocess, sys, tempfile
from pathlib import Path
from typing import Optional
from aie.iron.pythoc.compiler import (
    _find_pythoc_installation,
    _make_helper_wrapper,
    _strip_unsupported_flags,
    _get_opt_path,
    _get_llc_path,
    _get_extra_llc_flags,
    _promote_to_linkonce,
)


def compile_pythoc_source_noinline(
    source_code: str,
    function_name: str,
    target_arch: str = "aie2",
    output_dir: Optional[str] = None,
    optimization_level: int = 2,
    verbose: bool = False,
    extra_globals: Optional[dict] = None,
    inline: bool = False,
    exported_helpers: tuple[str, ...] = (),
    noinline: tuple[str, ...] = (),
) -> Path:
    """Compile PythoC source code to an AIE object file (or inline IR).

    Args:
        source_code: PythoC source code as string
        function_name: Name of the function to compile
        target_arch: Target architecture (aie2, aie2p)
        output_dir: Directory for output files (default: temp directory)
        optimization_level: LLVM optimization level (0-3)
        verbose: Enable verbose output
        exported_helpers: Names of preceding helper definitions to retain as
            externally callable symbols. Other helpers remain private.
        inline: When True, stop at an optimized ``.ll`` (LLVM IR) marked
            ``alwaysinline`` instead of running ``llc`` to a ``.o``. aiecc
            merges this IR into the core module (llvm-link) and the inliner
            folds the kernel body into the core — no surviving ``func.call``
            and no separately object-linked kernel ``.o``. When False
            (default), behaves as before and returns a compiled ``.o``.

    Returns:
        Path to the compiled object file (``.o``) when ``inline`` is False,
        or to the optimized IR file (``.ll``) when ``inline`` is True.

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

    # Create output directory (PYTHOC_DEBUG_DIR overrides tmpdir for inspection)
    if output_dir is None:
        output_dir = os.environ.get("PYTHOC_DEBUG_DIR") or tempfile.mkdtemp(
            prefix="pythoc_iron_"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set target triple for AIE
    target_triple = f"{target_arch}-none-unknown-elf"
    old_triple = os.environ.get("PYTHOC_TARGET_TRIPLE")
    os.environ["PYTHOC_TARGET_TRIPLE"] = target_triple

    try:
        # Parse source code
        tree = ast.parse(source_code)

        # Find the target function and any helper functions defined before it.
        # Only functions appearing before the target are treated as helpers
        # (these are typically prepended by PythocKernel's helpers= parameter).
        # Functions after the target are ignored — they may be unrelated
        # functions in the same file (e.g., when compiling from a kernel file
        # with multiple independent functions).
        func_node = None
        helper_nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Remove decorators to avoid execution issues
                node.decorator_list = []
                if node.name == function_name:
                    func_node = node
                    break
                else:
                    helper_nodes.append(node)

        unknown_exports = set(exported_helpers) - {node.name for node in helper_nodes}
        if unknown_exports:
            raise ValueError(
                f"Exported helpers must precede {function_name}: {sorted(unknown_exports)}"
            )

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
                struct,
            )
            from pythoc.aie import (
                aie_vector,
                load_v,
                store_v,
                load_unaligned_v,
                store_unaligned_v,
                load_floor_v,
                store_floor_v,
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
                extract_v4bfloat16_broadcast_to_v32bfloat16,
                extract_v8bfloat16_broadcast_to_v32bfloat16,
                T16_32x2_lo,
                T16_32x2_hi,
                T16_8x4,
                T16_4x8,
                T16_16x2,
                T16_2x16,
            )
            from pythoc.aie.loop_hints import prepare_for_pipelining, loop_range
            from pythoc.aie.profiling import event0, event1
            from pythoc.aie.utils import bitcast_i32_to_f32, fast_exp2_i32_to_f32
            from pythoc.aie.operations import (
                read_tm,
                write_tm,
                lock_acquire,
                lock_release,
                use_lock,
                LockAction,
            )

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
                    "load_unaligned_v": load_unaligned_v,
                    "store_unaligned_v": store_unaligned_v,
                    "load_floor_v": load_floor_v,
                    "store_floor_v": store_floor_v,
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
                    "lock_acquire": lock_acquire,
                    "lock_release": lock_release,
                    "use_lock": use_lock,
                    "LockAction": LockAction,
                    "struct": struct,
                    "range": range,  # Add Python's built-in range for standard loop syntax
                    # AIE API bf16 mmul<4,8,8> emulation helpers (aie2p)
                    "extract_v4bfloat16_broadcast_to_v32bfloat16": extract_v4bfloat16_broadcast_to_v32bfloat16,
                    "extract_v8bfloat16_broadcast_to_v32bfloat16": extract_v8bfloat16_broadcast_to_v32bfloat16,
                    # AIE2P bfloat16 shuffle-mode constants
                    "T16_32x2_lo": T16_32x2_lo,
                    "T16_32x2_hi": T16_32x2_hi,
                    "T16_8x4": T16_8x4,
                    "T16_4x8": T16_4x8,
                    "T16_16x2": T16_16x2,
                    "T16_2x16": T16_2x16,
                }
            )
        except ImportError as e:
            raise RuntimeError(f"Failed to import PythoC types: {e}")

        # Merge caller-provided globals (e.g. regdb constants)
        if extra_globals:
            user_globals.update(extra_globals)

        # Create compiler instance with user globals
        compiler = LLVMCompiler(user_globals=user_globals)

        # Compile helper functions first (if any) so they're available
        # in the same LLVM module when the target function calls them.
        # After compiling each helper, register it in PythoC's unified
        # registry so the main function's call resolver can find it.
        first = True
        if helper_nodes:
            from pythoc.registry import get_unified_registry, FunctionInfo
            from pythoc.type_resolver import TypeResolver

            registry = get_unified_registry()

            for helper in helper_nodes:
                if verbose:
                    print(f"Compiling helper {helper.name} to LLVM IR...")
                llvm_helper = compiler.compile_function_from_ast(
                    helper,
                    source_code=source_code,
                    reset_module=first,
                    user_globals=user_globals,
                )
                # Helpers are private implementation details of this artifact.
                # Internal linkage lets opt remove inlined copies and specialize
                # remaining calls without retaining redundant exported bodies.
                if helper.name not in exported_helpers:
                    llvm_helper.linkage = "internal"
                if helper.name in noinline:
                    llvm_helper.attributes.add("noinline")
                first = False

                # Build param type hints from annotations so call resolution works
                resolver = TypeResolver(
                    compiler.module.context, user_globals=user_globals
                )
                param_hints = {}
                param_names = []
                for arg in helper.args.args:
                    param_names.append(arg.arg)
                    if arg.annotation:
                        param_hints[arg.arg] = resolver.parse_annotation(arg.annotation)
                ret_hint = (
                    resolver.parse_annotation(helper.returns)
                    if helper.returns
                    else None
                )

                func_info = FunctionInfo(
                    name=helper.name,
                    source_file="<inline>",
                    ast_node=helper,
                    llvm_function=llvm_helper,
                    return_type_hint=ret_hint,
                    param_type_hints=param_hints,
                    param_names=param_names,
                    is_compiled=True,
                )
                registry.register_function(func_info)

                # Create a callable wrapper so visit_Name can find the
                # helper and _get_callable can locate handle_call on it.
                _make_helper_wrapper(helper.name, func_info, user_globals)

        # Compile target function to LLVM IR
        if verbose:
            print(f"Compiling {function_name} to LLVM IR...")

        llvm_func = compiler.compile_function_from_ast(
            func_node,
            source_code=source_code,
            reset_module=first,
            user_globals=user_globals,
        )

        # For inline lowering, mark the kernel ``alwaysinline`` so that when
        # aiecc llvm-links this IR into the core module, the inliner folds the
        # body in even under Peano's conservative ``-inline-threshold``. The
        # attribute serializes via ``get_ir()`` (str(module)) below.
        if inline:
            llvm_func.attributes.add("alwaysinline")

        # ── Pipeline matching mlir-aie/aiecc flow ──────────────────────
        # 1. Get raw IR from llvmlite (no in-process optimization)
        # 2. Preserve explicit memory alignment; removing it silently promises
        #    ABI/vector alignment for unaligned pointers.
        # 3. Peano opt -O2 (preserves loop metadata and memory contracts)
        # 4. Peano llc       (generates object file)

        ll_file = output_dir / f"{function_name}.ll"
        ir_str = compiler.get_ir()
        ir_str = _strip_unsupported_flags(ir_str)
        with open(ll_file, "w") as f:
            f.write(ir_str)

        if verbose:
            print(f"Raw IR saved to: {ll_file}")

        # Run Peano opt -O2
        opt_path = _get_opt_path()
        opt_ll_file = output_dir / f"{function_name}.opt.ll"
        opt_cmd = [
            opt_path,
            f"-O{optimization_level}" if optimization_level > 0 else "-O2",
            "-S",
            str(ll_file),
            "-o",
            str(opt_ll_file),
        ]

        if verbose:
            print(f"Running Peano opt: {' '.join(opt_cmd)}")

        result = subprocess.run(opt_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(f"Peano opt failed:\n{error_msg}")

        if verbose:
            print(f"Optimized IR saved to: {opt_ll_file}")

        # Keep the optimizer's proven alignment, including align 1 accesses.
        with open(opt_ll_file, "r") as f:
            opt_ir = f.read()
        opt_ir = _strip_unsupported_flags(opt_ir)

        # Inline mode: stop at the optimized .ll. aiecc llvm-links it into the
        # core module and the inliner (driven by alwaysinline) folds it in, so
        # there is no separately object-linked kernel and no surviving call.
        if inline:
            # Promote the kernel to linkonce_odr *after* the standalone opt so
            # that, once aiecc merges it into the core and the always-inliner
            # folds it in, globaldce removes the now-dead definition (external
            # linkage would leave a dead define for llc to codegen). It must be
            # done post-opt: linkonce_odr set before the standalone opt above
            # would let O2 DCE the (callerless) kernel away to nothing.
            opt_ir = _promote_to_linkonce(opt_ir, function_name)
            with open(opt_ll_file, "w") as f:
                f.write(opt_ir)
            if verbose:
                print(f"Inline IR (alwaysinline, linkonce_odr) ready: {opt_ll_file}")
            return opt_ll_file

        with open(opt_ll_file, "w") as f:
            f.write(opt_ir)

        # Compile optimized IR to object file using llc
        obj_file = output_dir / f"{function_name}.o"
        llc_path = _get_llc_path()

        llc_cmd = [
            llc_path,
            f"-march={target_arch}",
            *_get_extra_llc_flags(),
            "-filetype=obj",
            f"-o={obj_file}",
            str(opt_ll_file),
        ]

        if verbose:
            print(f"Compiling to object: {' '.join(llc_cmd)}")

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
