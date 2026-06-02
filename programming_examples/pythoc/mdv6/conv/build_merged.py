#!/usr/bin/env python3
"""Build a merged-device ELF for MDV6.

Takes one or more sub-device specs and produces a single ELF holding all
sub-devices plus a dispatcher whose ``aie.runtime_sequence @main`` chains
them via ``aiex.configure``/``aiex.run``:

  module {
    aie.device(npu2) @sub0 { ... aie.runtime_sequence @sub0_seq(...) ... }
    aie.device(npu2) @sub1 { ... aie.runtime_sequence @sub1_seq(...) ... }
    aie.device(npu2) {
      aie.runtime_sequence @main(<union of sub args>) {
        aiex.configure @sub0 { aiex.run @sub0_seq(...) }
        aiex.configure @sub1 { aiex.run @sub1_seq(...) }
      }
    }
  }

``aiecc.py --generate-full-elf --expand-load-pdis`` materializes the
``load_pdi`` switch between configs and packs every sub-device's CDO/PDI
into one ELF. XRT loads the resulting ELF via ``xrt.elf`` + ``xrt.ext.kernel``
with the entry point ``main``.

Options the caller controls:
  - ``share_arg_idxs={1}``: promote the per-sub arg at index 1 (weight by
    convention) to a single shared dispatcher arg. Standard for cloned
    fanout where all N copies use the same weights but different inputs.
  - ``chain_links=[(src_sub, src_arg, dst_sub, dst_arg)]``: alias one
    sub's arg to another's — used for the GEMM pair pattern where two
    sub-devices share one input BO.

The same MLIR pattern is used by llama32_1b/builders/*.py.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gemm_conv1x1")))
# MC_CONFIGS is the per-shape tuple list (label, n_cores, tile_h, tile_w,
# ic, oc, ks, stride, ppc). Sub-device kind="mc" build args are derived
# from this table.
from mc_configs import CONFIGS as MC_CONFIGS  # noqa: E402

KERNELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "kernels", "build")
)

_MC_SCRIPT = os.path.join(os.path.dirname(__file__), "aie2_multicore.py")
_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "gemm_conv1x1",
                 "aie2_gemm_conv1x1.py")
)


def _resolve_build_dir():
    # Subdir name must match run_tiled_mc.py's _MERGED_BD lookup ("build_merged"),
    # otherwise CI/lit runs that set MDV6_BUILD_DIR see no merged ELFs.
    root = os.environ.get("MDV6_BUILD_DIR")
    if root:
        return os.path.abspath(os.path.join(root, "build_merged"))
    return os.path.join(os.path.dirname(__file__), "build_merged")


def _stage_kernel_obj(name, build_dir):
    src = os.path.join(KERNELS_DIR, f"{name}.o")
    dst = os.path.join(build_dir, f"{name}.o")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"PythoC kernel object missing: {src} (run mdv6/kernels/build_kernels.py first)"
        )
    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
        shutil.copyfile(src, dst)


def _resolve_sub_spec(name, kind):
    """Resolve a sub-device name to (script_path, [args]) for MLIR generation.

    kind="mc"   → (aie2_multicore.py, [n_cores,tile_h,tile_w,ic,oc,ks,stride,ppc,input_depth])
    kind="gemm" → (aie2_gemm_conv1x1.py, [n_cores,tile_m,ic,oc,ppc,k_block])
    """
    if kind == "mc":
        for cfg in MC_CONFIGS:
            if cfg[0] == name:
                _, n_cores, tile_h, tile_w, ic, oc, ks, stride, ppc, *extra = cfg
                input_depth = extra[0] if extra else 1
                return _MC_SCRIPT, [str(n_cores), str(tile_h), str(tile_w),
                                    str(ic), str(oc), str(ks), str(stride),
                                    str(ppc), str(input_depth)]
        raise KeyError(f"no MC CONFIGS entry named {name!r}")
    if kind == "gemm":
        # GEMM specs are dynamically derived per-layer, not in a fixed CONFIGS
        # list. The build script (build_x1_gemm.py) hands us a tuple via the
        # sub_spec_override path; if a caller hits this with a bare name, it's
        # a build bug.
        raise KeyError(
            f"GEMM sub-device {name!r}: callers must use sub_spec_override "
            f"with (script, args) since GEMM names are derived from layer shape"
        )
    raise ValueError(f"unknown kind={kind!r}")


def _generate_sub_mlir(name, build_dir, kind="mc", sub_spec_override=None):
    """Emit a sub-device MLIR for one config name.

    sub_spec_override: optional (script_path, args_list) tuple bypassing
    _resolve_sub_spec — used by GEMM builds where names map to dynamic shape
    tuples not stored in a CONFIGS list.
    """
    if sub_spec_override is not None:
        script, cmd_args = sub_spec_override
    else:
        script, cmd_args = _resolve_sub_spec(name, kind)
    cmd = ["python3", script] + cmd_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{os.path.basename(script)} failed for {name}: "
                           f"{result.stderr.strip().splitlines()[-1]}")
    return result.stdout


_DEVICE_RE = re.compile(r"^(\s*)aie\.device\(npu2\)\s*\{", re.MULTILINE)
_RUNTIME_SEQ_RE = re.compile(
    r"(\s*)aie\.runtime_sequence\((.*?)\)\s*\{", re.DOTALL
)


def _rewrite_sub(mlir_text, dev_sym, seq_sym):
    """Inject sym_names + capture the runtime_sequence arg signature.

    Returns (rewritten_text, arg_type_list). ``arg_type_list`` is the list of
    MLIR types (e.g. ``memref<430336xui16>``) extracted from the runtime
    sequence — needed to type the dispatcher's @main signature.
    """
    # Strip the outer `module {` wrapper. Each sub MLIR starts with `module {`
    # and ends with `\n}\n`; we keep only the inner aie.device block.
    text = mlir_text.strip()
    if not text.startswith("module"):
        raise ValueError("sub MLIR does not start with `module`")
    # Find the opening brace after `module` and the matching closing brace.
    open_idx = text.index("{") + 1
    # Strip trailing `}` (the module's close).
    inner = text[open_idx:].rstrip()
    if not inner.endswith("}"):
        raise ValueError("sub MLIR has unexpected tail")
    inner = inner[:-1].rstrip()

    # Inject the device sym_name.
    inner, n = _DEVICE_RE.subn(
        lambda m: f"{m.group(1)}aie.device(npu2) @{dev_sym} {{",
        inner, count=1,
    )
    if n != 1:
        raise ValueError("could not find aie.device(npu2) in sub MLIR")

    # Inject the runtime_sequence sym_name. Capture arg list to thread into
    # the dispatcher signature.
    seq_match = _RUNTIME_SEQ_RE.search(inner)
    if not seq_match:
        raise ValueError("could not find aie.runtime_sequence in sub MLIR")
    indent = seq_match.group(1)
    arg_block = seq_match.group(2)
    inner = (
        inner[:seq_match.start()]
        + f"{indent}aie.runtime_sequence @{seq_sym}({arg_block}) {{"
        + inner[seq_match.end():]
    )

    # Parse `%arg0: memref<...>, %arg1: memref<...>, ...` → list of types.
    arg_types = []
    for piece in [p.strip() for p in arg_block.split(",")]:
        if not piece:
            continue
        _, ty = piece.split(":", 1)
        arg_types.append(ty.strip())

    return inner, arg_types


def _make_dispatcher_block(subs, share_arg_idxs=None, chain_links=None):
    """Emit the dispatcher aie.device with one aiex.configure/aiex.run per sub.

    Args:
        subs: list of (dev_sym, seq_sym, arg_types) — one per sub-device.
        share_arg_idxs: optional set of *per-sub* arg indices that should be
            shared across all sub-devices. The dispatcher emits ONE dispatcher
            arg for each shared index (taken from sub 0's types — they must
            match across subs) and reuses it in every aiex.run. The remaining
            per-sub args are flattened in order. Typical use: share the wt
            arg (idx=1) when cloning the same kernel for N batches of the
            same OCB.
        chain_links: optional list of (sub_a, arg_a, sub_b, arg_b) tuples
            (Phase C). Each link aliases sub_b's arg_b to the dispatcher %arg
            that sub_a's arg_a uses — useful for chaining "sub_a output ==
            sub_b input" so one BO threads through both sub-devices with no
            host plumbing between them. Types of the two args must match.
            chain_links is applied *after* share_arg_idxs.

    Returns a string containing the indented dispatcher device block.
    """
    share_arg_idxs = set(share_arg_idxs or ())
    chain_links = list(chain_links or ())

    if share_arg_idxs:
        # Validate the shared args have the same MLIR type across every sub.
        ref_types = subs[0][2]
        for dev_sym, _, ats in subs[1:]:
            for idx in share_arg_idxs:
                if ats[idx] != ref_types[idx]:
                    raise ValueError(
                        f"share-arg-idx {idx} type mismatch: sub {subs[0][0]!r} "
                        f"has {ref_types[idx]} but {dev_sym!r} has {ats[idx]}"
                    )

    # Validate chain links: types must match, and a destination (sub_b, arg_b)
    # may only be the target of ONE chain link (else the alias is ambiguous).
    chain_dest = {}  # (sub_b, arg_b) -> (sub_a, arg_a)
    for sub_a, arg_a, sub_b, arg_b in chain_links:
        if sub_a >= len(subs) or sub_b >= len(subs):
            raise ValueError(f"chain_links references out-of-range sub index")
        ta = subs[sub_a][2][arg_a]
        tb = subs[sub_b][2][arg_b]
        if ta != tb:
            raise ValueError(
                f"chain_links type mismatch: sub{sub_a}.arg{arg_a} ({ta}) "
                f"vs sub{sub_b}.arg{arg_b} ({tb})"
            )
        key = (sub_b, arg_b)
        if key in chain_dest:
            raise ValueError(
                f"chain_links destination (sub{sub_b}, arg{arg_b}) already "
                f"aliased to (sub{chain_dest[key][0]}, arg{chain_dest[key][1]})"
            )
        chain_dest[key] = (sub_a, arg_a)

    # Build dispatcher arg list: shared args first (taken from sub 0), then
    # per-sub args in sub order. Track the dispatcher %arg index for every
    # (sub_i, sub_j-arg) so aiex.run can reference them.
    flat_types = []
    sub_arg_refs = [[None] * len(ats) for _, _, ats in subs]  # sub_arg_refs[i][j] = "%argK"

    if share_arg_idxs:
        ref_types = subs[0][2]
        shared_refs = {}
        for idx in sorted(share_arg_idxs):
            shared_refs[idx] = f"%arg{len(flat_types)}"
            flat_types.append(ref_types[idx])

    for i, (_, _, ats) in enumerate(subs):
        for j, ty in enumerate(ats):
            if j in share_arg_idxs:
                sub_arg_refs[i][j] = shared_refs[j]
                continue
            if (i, j) in chain_dest:
                # Alias to the source's dispatcher arg. Source is always an
                # earlier sub (we process subs in order), so its ref is set.
                src_i, src_j = chain_dest[(i, j)]
                if sub_arg_refs[src_i][src_j] is None:
                    raise ValueError(
                        f"chain_links source (sub{src_i}, arg{src_j}) not yet "
                        f"emitted — chain target sub must come after its source"
                    )
                sub_arg_refs[i][j] = sub_arg_refs[src_i][src_j]
                continue
            sub_arg_refs[i][j] = f"%arg{len(flat_types)}"
            flat_types.append(ty)

    sig = ", ".join(f"%arg{i}: {ty}" for i, ty in enumerate(flat_types))

    body_lines = []
    for i, (dev_sym, seq_sym, ats) in enumerate(subs):
        argv = ", ".join(sub_arg_refs[i])
        types_tuple = ", ".join(ats)
        body_lines.append(f"      aiex.configure @{dev_sym} {{")
        body_lines.append(
            f"        aiex.run @{seq_sym}({argv}) : ({types_tuple})"
        )
        body_lines.append("      }")
    body = "\n".join(body_lines)

    return (
        "  aie.device(npu2) {\n"
        f"    aie.runtime_sequence @main({sig}) {{\n"
        f"{body}\n"
        "    }\n"
        "  }"
    )


def build_merged(out_name, sub_names, build_dir=None, share_arg_idxs=None,
                 kind="mc", sub_spec_overrides=None, chain_links=None,
                 sub_extra_args=None):
    """Generate and compile a merged-device ELF.

    Args:
        out_name: base filename (without extension) for the output .elf.
        sub_names: list of sub-device labels. For kind="mc" these must match
            MC_CONFIGS names. For kind="gemm" they are arbitrary labels keyed
            into sub_spec_overrides.
        build_dir: target directory (default: $MDV6_BUILD_DIR/merged).
        share_arg_idxs: per-sub arg indices to share across all sub-devices
            (one dispatcher arg, reused in every aiex.run). Use {1} for the
            common "shared weight" pattern.
        kind: "mc" (default) or "gemm" — selects the MLIR generator script.
        sub_spec_overrides: dict {sub_name: (script_path, args_list)} bypassing
            the CONFIGS lookup. Required for GEMM since GEMM shapes are
            dynamically derived per-layer (no fixed CONFIGS list).
        chain_links: optional list of (sub_a, arg_a, sub_b, arg_b) tuples
            (Phase C). Aliases sub_b's arg_b to sub_a's arg_a so one
            dispatcher %arg is shared between them — useful for cross-layer
            chaining where sub_a's output BO is the same buffer as sub_b's
            input.
        sub_extra_args: optional dict {sub_idx: [extra_cli_arg, ...]} appended
            to the generator command for that sub (e.g. ["--trace-size",
            str(N), "--trace-worker", "0"]). Lets us instrument one sub-
            device with hardware tracing while keeping siblings unchanged.
    """
    if build_dir is None:
        build_dir = _resolve_build_dir()
    os.makedirs(build_dir, exist_ok=True)

    # Stage both conv kernel .o files; aiecc resolves link_with relative to
    # the build cwd, so they need to be next to the MLIR. K-blocked GEMM uses
    # gemm_conv1x1_kblocked_bf16.o — staged too if present.
    # Always-required kernels for the standard conv/gemm sub-devices.
    obj_names = ["conv3x3_fused_packed_bf16", "gemm_conv1x1_fused_packed_bf16"]
    # Optional kernels — staged if present in kernels/build/, ignored if not.
    # The sub-device that actually link_with's one of these still fails at
    # aiecc time if its specific .o is missing, but unrelated builds (e.g.
    # an OCB conv build that doesn't touch GEMM) don't have to pay the
    # bring-up cost of every optional kernel.
    optional = ["gemm_conv1x1_kblocked_bf16", "rn3_pair_fused_bf16"]
    for k in obj_names:
        _stage_kernel_obj(k, build_dir)
    for k in optional:
        try:
            _stage_kernel_obj(k, build_dir)
        except FileNotFoundError:
            pass

    print(f"  {out_name}: generating {len(sub_names)} sub-MLIRs...")
    # Cache key includes the sub label AND any per-sub extra args so
    # an instrumented sub doesn't accidentally reuse a non-traced clone.
    raw_cache = {}
    subs = []
    sub_chunks = []
    for idx, name in enumerate(sub_names):
        extra = (sub_extra_args or {}).get(idx)
        cache_key = (name, tuple(extra) if extra else None)
        if cache_key not in raw_cache:
            override = (sub_spec_overrides or {}).get(name)
            if extra:
                # Append extra args to the override's args list, or build a
                # fresh one from _resolve_sub_spec.
                if override is None:
                    script, base_args = _resolve_sub_spec(name, kind)
                    override = (script, list(base_args) + list(extra))
                else:
                    script, base_args = override
                    override = (script, list(base_args) + list(extra))
            raw_cache[cache_key] = _generate_sub_mlir(
                name, build_dir, kind=kind, sub_spec_override=override
            )
        sub_text = raw_cache[cache_key]
        dev_sym = f"sub{idx}_{name}"
        seq_sym = f"sub{idx}_{name}_seq"
        rewritten, arg_types = _rewrite_sub(sub_text, dev_sym, seq_sym)
        subs.append((dev_sym, seq_sym, arg_types))
        sub_chunks.append(rewritten)
        tag = " [traced]" if extra else ""
        print(f"    [{idx}] {name} ({len(arg_types)} args){tag}")

    dispatcher = _make_dispatcher_block(subs, share_arg_idxs=share_arg_idxs,
                                         chain_links=chain_links)

    merged_mlir = "module {\n" + "\n".join(sub_chunks) + "\n" + dispatcher + "\n}\n"
    mlir_path = os.path.join(build_dir, f"{out_name}.mlir")
    with open(mlir_path, "w") as f:
        f.write(merged_mlir)
    print(f"  {out_name}: merged MLIR written to {mlir_path}")

    elf_path = os.path.join(build_dir, f"{out_name}.elf")
    cmd = (
        f"cd {build_dir} && aiecc.py --no-aiesim --no-xchesscc --no-xbridge "
        f"--no-compile-host "
        f"--generate-full-elf --expand-load-pdis "
        f"--full-elf-name={out_name}.elf {out_name}.mlir"
    )
    print(f"  {out_name}: compiling...", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        last = (result.stderr.strip().splitlines() or [""])[-1]
        print(f"  {out_name}: FAIL\n    {last}")
        return None
    print(f"  {out_name}: OK -> {elf_path}")
    return elf_path


def main():
    parser = argparse.ArgumentParser(
        description="Build a merged-device ELF from MDV6 sub-device configs."
    )
    parser.add_argument(
        "--out", default="merged_ftconv01",
        help="output ELF name (without extension)",
    )
    parser.add_argument(
        "subs", nargs="*",
        help="CONFIGS names of sub-devices to merge (default: mc_ftconv0 mc_ftconv1)",
    )
    parser.add_argument(
        "--share", type=str, default="",
        help="comma-sep per-sub arg indices to share across all subs (e.g. '1' to share the wt arg in batch-merge patterns)",
    )
    args = parser.parse_args()

    sub_names = args.subs or ["mc_ftconv0", "mc_ftconv1"]
    share = {int(s) for s in args.share.split(",") if s.strip()}
    path = build_merged(args.out, sub_names, share_arg_idxs=share)
    return 0 if path is not None else 1


if __name__ == "__main__":
    sys.exit(main())
