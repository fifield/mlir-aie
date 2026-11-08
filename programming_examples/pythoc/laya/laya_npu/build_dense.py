"""Rebuild native dense-engine cores and apply the inherited dead-scratch transform."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location("laya_npu_dense_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(out, target="dense_native"):
    out = Path(out).resolve()
    if out.exists():
        raise RuntimeError("output must not exist; preserve prior builds")
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (64 * 1024**3, 64 * 1024**3))
    os.environ["PYTHOC_LLC_FLAGS"] = "-enable-pipeliner=false -stack-size-section"
    sys.dont_write_bytecode = True
    src = ROOT / "src/dense"
    flags = json.loads((src / "flags.json").read_text())
    peano = Path(os.environ["PEANO_INSTALL_DIR"]) / "bin"
    aiecc = Path(shutil.which("aiecc")).resolve()
    commands = []
    objects = {}
    started = time.monotonic()

    def run(cmd):
        commands.append(list(map(str, cmd)))
        with (out / "compile.log").open("a") as log:
            subprocess.run(
                cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    from aie.iron.pythoc.compiler import compile_pythoc_source

    for name in [
        "cast_bias",
        "residual",
        "rope",
        "gelu_observable",
        "laya_gemm_native",
        "bf16_zero_kernel",
    ]:
        mod = load(src / (name + ".py"))
        fn = getattr(mod, name)
        extra = vars(mod).copy()
        if name == "bf16_zero_kernel":
            extra["C_ELEMS_CONST"] = 2048
        level = 2 if name in ("laya_gemm_native", "bf16_zero_kernel") else 1
        obj = Path(
            compile_pythoc_source(
                fn.__aie_source__,
                name,
                target_arch="aie2p",
                output_dir=out,
                optimization_level=level,
                extra_globals=extra,
            )
        )
        objects[obj.name] = digest(obj)
    for name, defines, extra_flags in [
        ("controls", [], []),
        ("dispatch", [], ["-fno-jump-tables"]),
        # dense_m2n2 (fast B1/B2/B4 L64) and dense_m2n2_cb (fast B1/L64's device_cb
        # engine) use the branchless GELU epilogue (M8B_GELU_FAST), bit-identical to
        # gelu_observable on every BF16 input.
        (
            "wrapper_m8b_d1",
            ["M8B_CHUNKS=8", "M8B_GELU"]
            + (["M8B_GELU_FAST"] if target in ("dense_m2n2", "dense_m2n2_cb") else []),
            [],
        ),
        ("wrapper_m8d", ["M8D_M=32"], []),
        ("wrapper_ln", [], []),
    ]:
        source = out / (name + ".cc")
        shutil.copyfile(src / (name + ".cc"), source)
        if name == "dispatch" and target.endswith("_s8"):
            # device_s8 (M37) runs 3 qkvR tiles per core (Q, K, V): RoPE on the first two thirds of ntile,
            # which is device_02's j < 4 at ntile 6. The source of every other target is unchanged.
            text = source.read_text()
            if text.count("if (j < 4)") != 1:
                raise RuntimeError("dispatch.cc RoPE rule not found")
            source.write_text(text.replace("if (j < 4)", "if (3 * j < 2 * h[2])"))
        obj = source.with_suffix(".o")
        run(
            [
                str(peano / "clang++"),
                *flags,
                *extra_flags,
                *[f"-D{x}" for x in defines],
                "-I",
                str(aiecc.parents[1] / "include"),
                "-c",
                str(source),
                "-o",
                str(obj),
            ]
        )
        objects[obj.name] = digest(obj)
    assembly = out / "m8b_crrnd_set_0.s"
    assembly.write_text(
        (src / "crrnd_thunk.s.in").read_text().replace("CRRND_VALUE", "0")
    )
    run(
        [
            str(peano / "clang"),
            "--target=aie2p-none-unknown-elf",
            "-x",
            "assembler",
            "-c",
            str(assembly),
            "-o",
            str(assembly.with_suffix(".o")),
        ]
    )
    objects["m8b_crrnd_set_0.o"] = digest(assembly.with_suffix(".o"))
    dependencies = [
        "dispatch",
        "wrapper_m8b_d1",
        "wrapper_m8d",
        "wrapper_ln",
        "m8b_crrnd_set_0",
        "controls",
        "residual",
        "cast_bias",
        "gelu_observable",
        "rope",
    ]
    run(
        [
            str(peano / "ld.lld"),
            "-r",
            *[str(out / (name + ".o")) for name in dependencies],
            "-o",
            str(out / "engine.o"),
        ]
    )
    objects["engine.o"] = digest(out / "engine.o")
    # dense_m2n2_cb: device_cb's 16-core engine half; dense_*_s8: device_s8's 32-core B1-split
    # engine (M37). Both are generated from their IRON designs (src/graphs/families/).
    # Every other target is the 32-core device_02.
    count = 16 if target == "dense_m2n2_cb" else 32
    if target == "dense_m2n2_cb" or target.endswith("_s8"):
        sys.path.insert(0, str(ROOT / "src/graphs"))
        from families import device_cb, device_s8

        text = device_cb.engine_core_template() if target == "dense_m2n2_cb" else device_s8.core_template()
    else:
        text = (src / "dense_native.mlir.in").read_text()
    for name in ["laya_gemm_native", "bf16_zero_kernel", "engine"]:
        placeholder = "@" + name.upper() + "@"
        if placeholder not in text:
            raise RuntimeError("missing core object placeholder")
        text = text.replace(placeholder, str(out / (name + ".o")))
    graph = out / "device.mlir"
    graph.write_text(text)
    run(
        [
            str(aiecc),
            "--no-progress",
            "--get-core-elfs",
            # Graph templates fix every buffer address; keep data-size measurement
            # from moving buffers away from them.
            "--no-measure-data-size",
            f"--tmpdir={out}/work",
            str(graph),
        ]
    )
    cores = sorted(out.glob("elfs_*/*.elf"))
    if len(cores) != count:
        raise RuntimeError(f"expected {count} dense cores")
    # Fast leaves replace the native leaf in place: (source, defines, leaf bytes).
    leaves = {
        "dense_fast": ("mmul_reuse.cc", ["-DMATRIX_NBLOCK=2"], 576),
        "dense_m2n2": ("mmul_m2n2.cc", [], 752),
        "dense_m2n2_cb": ("mmul_m2n2.cc", [], 752),
        "dense_fast_s8": ("mmul_reuse.cc", ["-DMATRIX_NBLOCK=2"], 576),
    }
    fast_object = None
    if target in leaves:
        leaf_source, leaf_defines, leaf_size = leaves[target]
        fast_source = out / leaf_source
        shutil.copyfile(src / leaf_source, fast_source)
        fast_object = out / "bfp_leaf.o"
        fast_flags = [
            f
            for f in flags
            if not f.startswith("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=")
        ]
        run(
            [
                str(peano / "clang++"),
                *fast_flags,
                *leaf_defines,
                "-I",
                str(aiecc.parents[1] / "include"),
                "-c",
                str(fast_source),
                "-o",
                str(fast_object),
            ]
        )
        objects[fast_object.name] = digest(fast_object)
    leaf_patch = load(src / "leaf_patch.py")
    scratch = load(src / "scratch.py")
    dead = out / "dead"
    dead.mkdir()
    patched = {}
    rawcores = {}
    linked_leaves = {}
    for core in cores:
        raw = core.read_bytes()
        if fast_object is not None:
            address = leaf_patch.address(raw)
            if address not in linked_leaves:
                linked_dir = out / "linked_bfp" / str(address)
                linked_dir.mkdir(parents=True)
                script = linked_dir / "leaf.ld"
                script.write_text(
                    "SECTIONS { .text " + str(address) + " : { *(.text*) } }\n"
                )
                linked = linked_dir / "leaf.elf"
                run(
                    [
                        str(peano / "ld.lld"),
                        "-e",
                        "laya_gemm_native",
                        "-T",
                        str(script),
                        str(fast_object),
                        "-o",
                        str(linked),
                    ]
                )
                linked_leaves[address] = linked
            raw = leaf_patch.patch(raw, linked_leaves[address].read_bytes(), leaf_size)
        header, section, symbols, index = scratch.inspect(raw)
        changed = bytearray(raw)
        struct.pack_into(
            "<I", changed, header["e_phoff"] + index * header["e_phentsize"], 0
        )
        # The scratch BSS aliases the graph's buffers. With its segment unloaded,
        # also give its section no size, so aiecc reserves none of that memory.
        struct.pack_into("<I", changed, section["sh_entry"] + 20, 0)
        dst = dead / core.name
        dst.write_bytes(changed)
        patched[dst.name] = {"path": str(dst.relative_to(out)), "sha256": digest(dst)}
        rawcores[core.name] = {
            "path": str(core.relative_to(out)),
            "sha256": digest(core),
        }
    result = {
        "status": "built_not_qualified",
        "target": target,
        "seconds": time.monotonic() - started,
        "objects": objects,
        "cores": patched,
        "native_cores": rawcores,
        "commands": commands,
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in src.iterdir() if p.is_file()
        },
        "transform": ("Source-compiled BFP leaf patch; " if target in leaves else "")
        + "Proven BSS program header type PT_LOAD -> PT_NULL and .bss sh_size -> 0; raw source-built ELFs preserved.",
    }
    (out / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target",
        choices=["dense_native", "dense_fast", "dense_m2n2", "dense_m2n2_cb", "dense_native_s8", "dense_fast_s8"],
        required=True,
    )
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    r = build(a.output, a.target)
    print(json.dumps({"target": a.target, "cores": len(r["cores"])}))


if __name__ == "__main__":
    main()
