"""Rebuild cast and normalization cores from local Python, C++, assembly and MLIR."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TARGETS = ("cast8", "castrelu8", "lnf", "lnr8")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(target, out):
    out = Path(out).resolve()
    if out.exists():
        raise RuntimeError("output must not exist; preserve prior builds")
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (64 * 1024**3, 64 * 1024**3))
    sys.dont_write_bytecode = True
    os.environ["PYTHOC_LLC_FLAGS"] = "-enable-pipeliner=false -stack-size-section"
    source = ROOT / "src/scalar"
    shared = source / "shared"
    recipes = json.loads((source / "recipes.json").read_text())
    recipe = recipes[target]
    commands = []
    objects = {}
    started = time.monotonic()
    peano = Path(os.environ["PEANO_INSTALL_DIR"]) / "bin"
    aiecc = Path(shutil.which("aiecc")).resolve()
    include = aiecc.parents[1] / "include"

    def run(command):
        commands.append(list(map(str, command)))
        with (out / "compile.log").open("a") as log:
            subprocess.run(
                command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    def python_leaf(name):
        spec = importlib.util.spec_from_file_location(
            "laya_npu_scalar_" + name, shared / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from aie.iron.pythoc.compiler import compile_pythoc_source

        fn = getattr(module, name)
        obj = Path(
            compile_pythoc_source(
                fn.__aie_source__,
                fn.__aie_name__,
                target_arch="aie2p",
                output_dir=out,
                optimization_level=1,
                extra_globals={
                    k: getattr(module, k)
                    for k in (
                        "set_ctrl_reg",
                        "sqrtf",
                        "ACC2048_accfloat_add_conf",
                        "v32accfloat_to_v32bf16",
                        "vector_extract",
                    )
                },
            )
        )
        objects[obj.name] = digest(obj)
        return obj

    def cpp(name, defines=()):
        path = out / (name + ".cc")
        shutil.copyfile(shared / (name + ".cc"), path)
        obj = out / (name + ".o")
        run(
            [
                str(peano / "clang++"),
                *recipes["cpp_flags"],
                *[f"-D{x}" for x in defines],
                "-I",
                str(include),
                "-c",
                str(path),
                "-o",
                str(obj),
            ]
        )
        objects[obj.name] = digest(obj)
        return obj

    cast = python_leaf("cast_bias")
    if target == "cast8":
        leaf = cast
    elif target == "castrelu8":
        relu = cpp("relu")
        leaf = out / "kernels.o"
        run([str(peano / "ld.lld"), "-r", str(cast), str(relu), "-o", str(leaf)])
    else:
        wrapper = cpp("wrapper", recipe["defines"])
        controls = cpp("controls")
        assembly = out / "m8b_crrnd_set_0.s"
        assembly.write_text(
            (shared / "crrnd_thunk.s.in").read_text().replace("CRRND_VALUE", "0")
        )
        thunk = assembly.with_suffix(".o")
        run(
            [
                str(peano / "clang"),
                "--target=aie2p-none-unknown-elf",
                "-x",
                "assembler",
                "-c",
                str(assembly),
                "-o",
                str(thunk),
            ]
        )
        objects[thunk.name] = digest(thunk)
        dependencies = [wrapper, thunk, controls, cast]
        if target == "lnr8":
            dependencies.append(python_leaf("residual"))
        leaf = out / ("m8b_lnf.o" if target == "lnf" else "m8b_lnr.o")
        run([str(peano / "ld.lld"), "-r", *map(str, dependencies), "-o", str(leaf)])
    objects[leaf.name] = digest(leaf)
    template = (source / target / "device.mlir.in").read_text()
    if "@LEAF_OBJECT@" not in template:
        raise RuntimeError("missing local object placeholder")
    graph = out / "device.mlir"
    graph.write_text(template.replace("@LEAF_OBJECT@", str(leaf)))
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
    if len(cores) != recipe["core_count"]:
        raise RuntimeError(f"core count: {len(cores)}")
    result = {
        "status": "built_not_qualified",
        "target": target,
        "seconds": time.monotonic() - started,
        "objects": objects,
        "commands": commands,
        "cores": {
            p.name: {"path": str(p.relative_to(out)), "sha256": digest(p)}
            for p in cores
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in source.rglob("*")
            if p.is_file()
        },
        "pythoc_flags": os.environ["PYTHOC_LLC_FLAGS"],
        "pythoc_optimization_level": 1,
    }
    (out / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=TARGETS, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = build(a.target, a.output)
    print(
        json.dumps(
            {
                "target": a.target,
                "cores": len(result["cores"]),
                "seconds": result["seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
