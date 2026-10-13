#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check AIE4 intrinsic signatures and memory effects against installed Peano.

Scan include/ and lib/ for intrinsic names, then verify their snapshot
signatures and memory effects with Peano opt. --build-dir cross-checks XLLVM
names against generated TableGen output; --all checks every snapshot entry.
XLLVM operand/result types are checked separately by xllvm_verify.test.

The snapshot is a checked-in baseline, regenerated manually from llvm-aie
sources. Wheel/pin differences produce warnings; missing names, signature
mismatches, and memory-effect mismatches fail the check. See README.md for
scope and regeneration instructions.

Uses PyYAML when available, with a fallback parser for the snapshot schema.
Defaults to $PEANO_INSTALL_DIR; use --peano-install-dir to override it.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent  # utils/aie4 -> utils -> <mlir-aie repo root>

DEFAULT_SNAPSHOT = THIS_DIR / "aie4_intrinsics_snapshot.yaml"
DEFAULT_PEANO_REQUIREMENTS = REPO_ROOT / "utils" / "aie4-peano-requirements.txt"


def _executable_name(name):
    return f"{name}.exe" if os.name == "nt" else name


# The fallback loader supports the machine-generated snapshot schema only.
try:
    import yaml as _yaml

    def load_snapshot(path: Path) -> dict:
        with open(path) as f:
            return _yaml.safe_load(f)

    _YAML_BACKEND = "PyYAML"
except ImportError:
    _yaml = None
    _YAML_BACKEND = "minimal hand-rolled parser (PyYAML not importable)"

    def load_snapshot(path: Path) -> dict:
        return _minimal_yaml_load(path)

    def _minimal_yaml_load(path: Path) -> dict:
        """Read the intrinsics list emitted by yaml.safe_dump(default_flow_style=False).

        Supports only the snapshot's flat record schema.
        """
        intrinsics = []
        cur = None
        cur_list_key = None
        with open(path) as f:
            lines = f.readlines()
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if line.startswith("#") or not line.strip():
                i += 1
                continue
            if line.startswith("- name:"):
                if cur is not None:
                    intrinsics.append(cur)
                cur = {"name": line.split(":", 1)[1].strip()}
                cur_list_key = None
                i += 1
                continue
            if cur is None:
                i += 1
                continue
            m = re.match(r"^  (\w+):\s*(.*)$", line)
            if m:
                key, rest = m.group(1), m.group(2).strip()
                if rest == "" or rest == "[]":
                    if rest == "[]":
                        cur[key] = []
                        cur_list_key = None
                    else:
                        cur[key] = []
                        cur_list_key = key
                else:
                    if rest.isdigit():
                        cur[key] = int(rest)
                    else:
                        cur[key] = rest
                    cur_list_key = None
                i += 1
                continue
            m = re.match(r"^  - (.*)$", line)
            if m and cur_list_key is not None:
                cur[cur_list_key].append(m.group(1).strip())
                i += 1
                continue
            i += 1
        if cur is not None:
            intrinsics.append(cur)
        return {"intrinsics": intrinsics}


# ---------------------------------------------------------------------------
# Step 1: collect referenced llvm.aie4.* names from include/ and lib/.
# ---------------------------------------------------------------------------

# These XLLVM classes forward their first argument as the mnemonic.
# ExtIntrOpBase prefixes it with "llvm.aie4.".
_TD_MNEMONIC_CLASS_RE = re.compile(
    r'\b(?:AIEVec4_IntrOp|AIE4\w*IntrOp)\s*<\s*"([^"]+)"'
)

# Arch-parameterized classes build the name from their argument, e.g.
# `class ReadTmIntrOp<string arch> : ExtIntrOpBase<..., arch#".read.tm", ...>`
# instantiated as `def ... : ReadTmIntrOp<"aie4">`.
_TD_ARCH_CLASS_RE = re.compile(
    r"\bclass\s+(\w+)\s*<\s*string\s+(\w+)\s*>\s*:\s*ExtIntrOpBase\s*<"
    r'[^,]*,\s*(\w+)\s*#\s*"([^"]+)"'
)
_TD_ARCH_DEF_RE = re.compile(r'\b(\w+)\s*<\s*"aie4"\s*>')

_TD_DEF_SPLIT_RE = re.compile(r"^(?=def\s)", re.MULTILINE)

# Trailing-dot prefixes such as "llvm.aie4.put." are skipped below;
# this scan does not resolve C++ string concatenation.
_LITERAL_NAME_RE = re.compile(r'"(llvm\.aie4\.[A-Za-z0-9_.]*)"')


_INC_LITERAL_NAME_RE = re.compile(r'"(llvm\.aie4\.[A-Za-z0-9_.]*)"')


def find_inc_names(inc_path: Path) -> set[str]:
    """Collect intrinsic names from TableGen-generated XLLVMConversions.inc."""
    text = inc_path.read_text(errors="replace")
    return set(_INC_LITERAL_NAME_RE.findall(text))


def find_referenced_names(repo_root: Path) -> tuple[set[str], list[str], set[str]]:
    """Return (names, skipped_partials, td_derived_names) from include/ and lib/.

    Only td_derived_names can be compared with XLLVMConversions.inc; C++
    declarations are outside that generated file's scope.
    """
    names: set[str] = set()
    td_derived_names: set[str] = set()
    skipped: list[str] = []
    src_dirs = [repo_root / "include", repo_root / "lib"]
    exts = {".td", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".inc"}
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file() or path.suffix not in exts:
                continue
            text = path.read_text(errors="replace")
            if (
                "llvm.aie4" not in text
                and "AIEVec4_IntrOp" not in text
                and "AIE4" not in text
            ):
                continue
            if path.suffix == ".td":
                # An op with a custom llvmBuilder emits the names spelled in
                # that builder (e.g. one op selecting among several intrinsics,
                # or reusing another op's); only the others emit the mnemonic.
                arch_suffix = {
                    m.group(1): m.group(4)
                    for m in _TD_ARCH_CLASS_RE.finditer(text)
                    if m.group(2) == m.group(3)
                }
                for block in _TD_DEF_SPLIT_RE.split(text):
                    arch_def = _TD_ARCH_DEF_RE.search(block)
                    if arch_def and arch_def.group(1) in arch_suffix:
                        derived = {"llvm.aie4" + arch_suffix[arch_def.group(1)]}
                        names |= derived
                        td_derived_names |= derived
                        continue
                    mnemonic = _TD_MNEMONIC_CLASS_RE.search(block)
                    if mnemonic is None:
                        continue
                    builder = block[block.find("llvmBuilder") :]
                    custom = (
                        set(_LITERAL_NAME_RE.findall(builder))
                        if "llvmBuilder" in block
                        else set()
                    )
                    derived = custom or {"llvm.aie4." + mnemonic.group(1)}
                    names |= derived
                    td_derived_names |= derived
            for m in _LITERAL_NAME_RE.finditer(text):
                literal = m.group(1)
                if literal.endswith("."):
                    skipped.append(
                        f"{literal!r} in {path.relative_to(repo_root)} "
                        "(partial string-concatenation prefix, not a full name)"
                    )
                    continue
                names.add(literal)
    return names, skipped, td_derived_names


# ---------------------------------------------------------------------------
# Step 2/3: type mapping and LLVM IR module synthesis.
# ---------------------------------------------------------------------------

_SCALAR_SUFFIX_TO_IR = {
    "f16": "half",
    "f32": "float",
    "f64": "double",
    "bf16": "bfloat",
}
_VEC_RE = re.compile(r"^llvm_v(\d+)(i\d+|f16|f32|f64|bf16)_ty$")
_INT_RE = re.compile(r"^llvm_i(\d+)_ty$")
_SCALAR_NAME_TO_IR = {
    "llvm_float_ty": "float",
    "llvm_double_ty": "double",
    "llvm_bfloat_ty": "bfloat",
    "llvm_half_ty": "half",
    "llvm_f16_ty": "half",
    "llvm_void_ty": "void",
    "llvm_ptr_ty": "ptr",
    "llvm_anyptr_ty": "ptr",
}


class UnknownTypeError(ValueError):
    pass


def map_type(token: str) -> str:
    """Map a TableGen type token to its LLVM IR spelling.

    llvm_v64i32_ty -> '<64 x i32>', llvm_i32_ty -> 'i32', llvm_bfloat_ty
    -> 'bfloat', llvm_i264_ty -> 'i264', llvm_ptr_ty -> 'ptr', etc.
    """
    if token in _SCALAR_NAME_TO_IR:
        return _SCALAR_NAME_TO_IR[token]
    m = _INT_RE.match(token)
    if m:
        return f"i{m.group(1)}"
    m = _VEC_RE.match(token)
    if m:
        count, scalar = m.group(1), m.group(2)
        scalar_ir = _SCALAR_SUFFIX_TO_IR.get(scalar, scalar)  # iN passes through
        return f"<{count} x {scalar_ir}>"
    raise UnknownTypeError(token)


def ret_type_ir(ret_types: list[str]) -> str:
    if not ret_types:
        return "void"
    if len(ret_types) == 1:
        return map_type(ret_types[0])
    return "{ " + ", ".join(map_type(t) for t in ret_types) + " }"


_IMMARG_RE = re.compile(r"^ImmArg<ArgNo=(\d+)>$")
# Snapshot ArgNo stores TableGen AttrIndex: 0 is the return value, and
# zero-based parameter N has index N+1 (llvm/include/llvm/IR/Intrinsics.td).


def zero_literal(ir_type: str) -> str:
    """Return a constant (not `undef`) usable for an ImmArg argument."""
    if ir_type in ("float", "double", "half", "bfloat"):
        return "0.0"
    if ir_type == "ptr":
        return "null"
    if ir_type.startswith("<"):
        return "zeroinitializer"
    return "0"  # integer


def build_module(
    name: str, ret_types: list[str], param_types: list[str], properties: list[str]
) -> str:
    ret_ir = ret_type_ir(ret_types)
    immarg_indices = set()
    for p in properties:
        m = _IMMARG_RE.match(p)
        if m:
            arg_no = int(m.group(1))
            if arg_no == 0:
                # The return value cannot be an immediate argument.
                continue
            immarg_indices.add(arg_no - 1)

    param_irs = [map_type(t) for t in param_types]
    decl_params = ", ".join(param_irs)

    call_args = []
    for idx, ir_type in enumerate(param_irs):
        if idx in immarg_indices:
            call_args.append(f"{ir_type} {zero_literal(ir_type)}")
        else:
            call_args.append(f"{ir_type} undef")
    call_args_str = ", ".join(call_args)

    result_assign = "" if ret_ir == "void" else "%r = "
    return (
        f'declare {ret_ir} @"{name}"({decl_params})\n\n'
        f"define void @use() {{\n"
        f'  {result_assign}call {ret_ir} @"{name}"({call_args_str})\n'
        f"  ret void\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# Step 3: expected memory attribute from snapshot properties.
# ---------------------------------------------------------------------------


def expected_memory_attr(properties: list[str]) -> str | None:
    """Return the expected memory(...) attribute, or None if absent.

    Observed with Peano: IntrHasSideEffects suppresses memory(none) when
    combined with IntrNoMem (e.g. event/acquire). With IntrArgMemOnly it
    forces argmem: readwrite, even with IntrReadMem or IntrWriteMem.
    """
    props = set(properties)
    has_side_effects = "IntrHasSideEffects" in props
    if "IntrNoMem" in props:
        if has_side_effects:
            return None
        return "memory(none)"
    if "IntrInaccessibleMemOnly" in props:
        read = "IntrReadMem" in props
        write = "IntrWriteMem" in props
        if read and not write:
            return "memory(inaccessiblemem: read)"
        if write and not read:
            return "memory(inaccessiblemem: write)"
        # Neither or both explicit -> full inaccessible-mem readwrite.
        return "memory(inaccessiblemem: readwrite)"
    if "IntrArgMemOnly" in props:
        read = "IntrReadMem" in props
        write = "IntrWriteMem" in props
        if has_side_effects:
            return "memory(argmem: readwrite)"
        if read and not write:
            return "memory(argmem: read)"
        if write and not read:
            return "memory(argmem: write)"
        # Neither or both explicit -> full argmem readwrite.
        return "memory(argmem: readwrite)"
    # Without a location restriction, effects apply to all memory.
    read = "IntrReadMem" in props
    write = "IntrWriteMem" in props
    if read and not write:
        return "memory(read)"
    if write and not read:
        return "memory(write)"
    # No memory attribute means unrestricted reads and writes.
    return None


_MEMORY_ATTR_RE = re.compile(r"memory\([^)]*\)")


def parse_opt_output(stdout: str) -> tuple[str | None, bool]:
    """Return (memory_attr_or_None, unknown_intrinsic: bool) from opt's output.

    Expects a successful (exit 0) `opt -S` run over a single-declare module.
    """
    unknown = "Unknown intrinsic" in stdout
    m = _MEMORY_ATTR_RE.search(stdout)
    return (m.group(0) if m else None), unknown


# ---------------------------------------------------------------------------
# Main per-intrinsic check.
# ---------------------------------------------------------------------------


class Result:
    def __init__(self, name, ok, message):
        self.name = name
        self.ok = ok
        self.message = message


def check_one(opt_path: Path, name: str, record: dict) -> Result:
    try:
        module_text = build_module(
            name,
            record.get("ret_types", []),
            record.get("param_types", []),
            record.get("properties", []),
        )
    except UnknownTypeError as e:
        return Result(
            name, False, f"checker limitation: unrecognized snapshot type token {e}"
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(module_text)
        tmp_path = f.name
    try:
        try:
            proc = subprocess.run(
                [str(opt_path), "-S", "-mtriple=aie4", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return Result(
                name,
                False,
                f"{name}: opt did not finish within 30s -- treating as a "
                f"failure rather than hanging the whole run.",
            )
    finally:
        os.unlink(tmp_path)

    combined = proc.stdout + proc.stderr

    if proc.returncode != 0:
        if "Intrinsic has incorrect argument type" in combined:
            return Result(
                name,
                False,
                f"SIGNATURE DRIFT: {name}: opt rejects the snapshot's "
                f"declared argument type(s) ('Intrinsic has incorrect "
                f"argument type!'). The wheel's real signature no longer "
                f"matches the snapshot.",
            )
        if "Intrinsic has incorrect return type" in combined:
            return Result(
                name,
                False,
                f"SIGNATURE DRIFT: {name}: opt rejects the snapshot's "
                f"declared return type ('Intrinsic has incorrect return "
                f"type!'). The wheel's real signature no longer matches "
                f"the snapshot.",
            )
        if "Callsite was not defined with variable arguments!" in combined:
            return Result(
                name,
                False,
                f"SIGNATURE DRIFT: {name}: opt rejects the call site's "
                f"arity ('Callsite was not defined with variable "
                f"arguments!'). The wheel's real signature now has a "
                f"different (likely variadic) argument count than the "
                f"snapshot records.",
            )
        if "immarg" in combined.lower():
            return Result(
                name,
                False,
                f"checker limitation for {name}: this intrinsic needs a "
                f"real immediate (ImmArg) constant the checker did not "
                f"supply correctly -- not a confirmed signature drift. "
                f"opt said: {combined.strip()!r}",
            )
        return Result(
            name,
            False,
            f"{name}: opt exited {proc.returncode} for an unrecognized "
            f"reason: {combined.strip()!r}",
        )

    mem_attr, unknown = parse_opt_output(combined)
    if unknown:
        return Result(
            name,
            False,
            f"NOT AN INTRINSIC IN THIS WHEEL: {name}: opt printed "
            f"'; Unknown intrinsic' -- this wheel's intrinsic table does "
            f"not know this name at all, so it is declared as a plain "
            f"external function with no nocallback/nounwind/willreturn "
            f"attributes and no verifier type-checking happened.",
        )

    expected = expected_memory_attr(record.get("properties", []))
    if expected != mem_attr:
        return Result(
            name,
            False,
            f"MEMORY ATTRIBUTE MISMATCH: {name}: snapshot properties "
            f"{record.get('properties', [])} imply {expected!r}, but opt "
            f"printed {mem_attr!r}.",
        )

    return Result(name, True, f"{name}: OK (memory={mem_attr!r})")


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------


def print_provenance(peano_install_dir: Path):
    print("== provenance ==")
    try:
        import importlib.metadata as _im

        wheel_version = _im.version("llvm-aie")
        print(f"installed llvm-aie wheel version (importlib.metadata): {wheel_version}")
    except Exception as e:
        wheel_version = None
        print(f"could not read llvm-aie wheel version via importlib.metadata: {e}")

    clang_path = peano_install_dir / "bin" / _executable_name("clang")
    clang_commit = None
    if clang_path.exists():
        try:
            out = subprocess.run(
                [str(clang_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            print(f"{clang_path} --version:")
            for line in out.splitlines():
                print(f"  {line}")
            m = re.search(r"\(https://[^ ]+ ([0-9a-f]{8,40})\)", out)
            if m:
                clang_commit = m.group(1)
        except Exception as e:
            print(f"could not run {clang_path} --version: {e}")
    else:
        print(f"clang not found at {clang_path}")

    pin_line = None
    if DEFAULT_PEANO_REQUIREMENTS.exists():
        for line in DEFAULT_PEANO_REQUIREMENTS.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pin_line = line
                break
    print(f"pin line ({DEFAULT_PEANO_REQUIREMENTS}): {pin_line}")

    pinned_version = None
    if pin_line and wheel_version:
        pin_m = re.search(r"llvm_aie-([^\s-]+)-py3", pin_line) or re.search(
            r"llvm[_-]aie==([^\s]+)", pin_line
        )
        pinned_version = pin_m.group(1) if pin_m else None
        if pinned_version and pinned_version != wheel_version:
            print(
                f"WARNING: pinned wheel version ({pinned_version}) differs from "
                f"the installed wheel version ({wheel_version}). This is not "
                f"fatal by itself (install/peano may simply not have been "
                f"reinstalled since the pin moved) -- but it means this run is "
                f"not necessarily checking the pinned wheel."
            )

    # Compare the pin's +g<hash> with clang's reported source revision.
    if pinned_version and clang_commit:
        pin_hash_m = re.search(r"\+g([0-9a-f]{7,40})", pinned_version)
        pin_hash = pin_hash_m.group(1) if pin_hash_m else None
        if (
            pin_hash
            and not clang_commit.startswith(pin_hash)
            and not pin_hash.startswith(clang_commit)
        ):
            print(
                f"WARNING: pin's commit hash ({pin_hash}) differs from clang "
                f"--version's reported commit ({clang_commit}). Not fatal by "
                f"itself (same rationale as the version-string check above)."
            )
    print()


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Signature snapshot YAML to check against.",
    )
    ap.add_argument(
        "--peano-install-dir",
        type=Path,
        default=(
            Path(os.environ["PEANO_INSTALL_DIR"])
            if os.environ.get("PEANO_INSTALL_DIR")
            else None
        ),
        help="Peano install root (containing bin/opt, bin/clang). "
        "Defaults to $PEANO_INSTALL_DIR.",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="mlir-aie repo root to scan include/ and lib/ under.",
    )
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Build directory to look for the generated "
        "include/aie/Dialect/XLLVM/IR/XLLVMConversions.inc "
        "under, for an independent cross-check of the "
        ".td-regex-derived name set. Skipped if not given "
        "or the file doesn't exist there.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Self-check every record in the snapshot (not just "
        "the names this repo currently references). Slower "
        "(hundreds of opt invocations), but catches "
        "checker bugs (e.g. immarg/memory-mapping errors) "
        "that happen not to affect any name this repo "
        "currently uses.",
    )
    args = ap.parse_args()

    print(f"YAML backend: {_YAML_BACKEND}")
    if not args.snapshot.exists():
        sys.exit(f"error: snapshot not found at {args.snapshot}")
    doc = load_snapshot(args.snapshot)
    if doc is None or not doc.get("intrinsics"):
        sys.exit(
            f"error: {args.snapshot} parsed to no intrinsics (empty or "
            f"malformed file?)"
        )
    by_name = {r["name"]: r for r in doc["intrinsics"]}
    print(f"loaded {len(by_name)} intrinsics from {args.snapshot}")

    if args.peano_install_dir is None:
        sys.exit("error: no Peano install dir given and $PEANO_INSTALL_DIR is unset")
    opt_path = args.peano_install_dir / "bin" / _executable_name("opt")
    if not opt_path.exists():
        sys.exit(f"error: opt not found at {opt_path}")

    print_provenance(args.peano_install_dir)

    if args.all:
        names = set(by_name.keys())
        skipped = []
        td_derived_names = set()
        print(
            f"== --all: self-checking all {len(names)} snapshot record(s), not "
            f"just names this repo currently references =="
        )
    else:
        names, skipped, td_derived_names = find_referenced_names(args.repo_root)
        print(
            f"== {len(names)} referenced llvm.aie4.* intrinsic name(s) found in "
            f"{args.repo_root}/include and {args.repo_root}/lib =="
        )
        for n in sorted(names):
            print(f"  {n}")
        if skipped:
            print(
                f"({len(skipped)} partial/concatenated string literal(s) ignored, "
                f"not full names:)"
            )
            for s in skipped:
                print(f"  skipped: {s}")

        if len(names) == 0:
            sys.exit(
                "error: 0 referenced llvm.aie4.* names found under "
                f"{args.repo_root}/include and {args.repo_root}/lib -- this "
                "almost certainly means the scan itself is broken (wrong "
                "--repo-root, or the regexes stopped matching after a "
                "rename), not that AIE4 support was removed. Failing hard "
                "instead of silently passing on zero checks."
            )

        # TableGen output independently checks coverage of the mnemonic regex.
        if args.build_dir is not None:
            inc_path = (
                args.build_dir
                / "include"
                / "aie"
                / "Dialect"
                / "XLLVM"
                / "IR"
                / "XLLVMConversions.inc"
            )
            if inc_path.exists():
                inc_names = find_inc_names(inc_path)
                # Compare only XLLVM names; the C++ declarations are not generated.
                missing_from_inc = td_derived_names - inc_names
                extra_in_inc = inc_names - td_derived_names
                print(
                    f"cross-check: {inc_path} has {len(inc_names)} llvm.aie4.* "
                    f"name(s); {len(td_derived_names)} XLLVM-op-derived name(s) "
                    f"found by this script's own .td scan"
                )
                if missing_from_inc or extra_in_inc:
                    if missing_from_inc:
                        failures_note = ", ".join(sorted(missing_from_inc))
                        print(
                            f"MISMATCH: name(s) referenced in source but absent "
                            f"from {inc_path}: {failures_note}"
                        )
                    if extra_in_inc:
                        failures_note = ", ".join(sorted(extra_in_inc))
                        print(
                            f"MISMATCH: name(s) in {inc_path} not found by this "
                            f"script's own .td scan: {failures_note}"
                        )
                    sys.exit(1)
                print("cross-check: OK (identical name sets)")
            else:
                print(
                    f"cross-check skipped: {inc_path} does not exist "
                    f"(build dir not given or not built yet)"
                )
        print()

    failures = []
    for name in sorted(names):
        record = by_name.get(name)
        if record is None:
            failures.append(
                Result(
                    name,
                    False,
                    f"MISSING FROM SNAPSHOT: {name}: referenced in "
                    f"code but not present in {args.snapshot}.",
                )
            )
            print(failures[-1].message)
            continue
        result = check_one(opt_path, name, record)
        print(result.message)
        if not result.ok:
            failures.append(result)

    print()
    print(
        f"== {len(names)} {'snapshot record(s) self-checked' if args.all else 'referenced'}, "
        f"{len(names) - len(failures)} OK, {len(failures)} FAILED =="
    )
    if failures:
        print("FAIL:")
        for r in failures:
            print(f"  - {r.message}")
        sys.exit(1)
    if args.all:
        print(
            "PASS: every snapshot record self-checks clean against the installed "
            "wheel."
        )
    else:
        print(
            "PASS: every referenced AIE4 intrinsic matches the installed wheel "
            "(snapshot vs installed wheel only; mlir-aie's declared signatures "
            "-- e.g. AIECoreToStandard.cpp's own C++ intrinsic table -- are "
            "checked by xllvm_verify.test and bead aie4-wp6-t4, not here)."
        )


if __name__ == "__main__":
    main()
