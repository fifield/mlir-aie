"""Rebuild special attention cores from local high-level leaves and MLIR."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(target, out):
    out = Path(out).resolve()
    if out.exists():
        raise RuntimeError("output must not exist; preserve prior builds")
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (64 * 1024**3, 64 * 1024**3))
    sys.dont_write_bytecode = True
    os.environ["PYTHOC_LLC_FLAGS"] = "-enable-pipeliner=false -stack-size-section"
    src = ROOT / "src/special"
    recipes = json.loads((src / "recipes.json").read_text())
    compiler = module(src / "compiler.py", "laya_npu_attention_compiler")
    externs = module(src / "externs.py", "laya_npu_attention_externs")
    original_strip = compiler._strip_unsupported_flags

    def strip(text):
        text = original_strip(text)
        names = tuple(
            n
            for n in ("exp_v64", "m12a_vdiv", "m12a_vscale", "attention_max16")
            if re.search(r'^define internal void @"' + n + r'"\(', text, re.M)
        )
        return externs.declare_externs(text, names) if names else text

    compiler._strip_unsupported_flags = strip
    objects = {}
    paths = {}
    commands = []
    cores = {}
    started = time.monotonic()

    def run(command, cwd):
        commands.append(list(map(str, command)))
        subprocess.run(command, cwd=cwd, check=True)

    def leaf(key):
        if key in paths:
            return paths[key]
        recipe = recipes["leaves"][key]
        helpers = [leaf(k) for k in recipe["helpers"]]
        source = src / recipe["source"]
        name = recipe["symbol"]
        directory = out / "leaves" / key
        if recipe.get("language") == "c++":
            directory.mkdir(parents=True)
            obj = directory / (name + ".o")
            run(
                [
                    shutil.which("clang++"),
                    *json.loads((src / "flags.json").read_text()),
                    "-I",
                    str(Path(shutil.which("aiecc")).resolve().parents[1] / "include"),
                    "-c",
                    str(source),
                    "-o",
                    str(obj),
                ],
                directory,
            )
        else:
            mod = module(source, "laya_npu_attention_" + key)
            obj = Path(
                compiler.compile_pythoc_source_noinline(
                    source.read_text(),
                    name,
                    target_arch="aie2p",
                    output_dir=directory,
                    optimization_level=1,
                    extra_globals=vars(mod),
                    noinline=tuple(recipe["noinline"]),
                )
            )
            if recipe.get("transform") == "pass2_lifetime":
                ir = directory / (name + ".opt.ll")
                text = ir.read_text()
                for old, new in json.loads((src / "pass2_edits.json").read_text()):
                    if text.count(old) != 1:
                        raise RuntimeError(
                            "pass2 transform precondition changed: " + old
                        )
                    text = text.replace(old, new, 1)
                ir.rename(directory / (name + ".original.opt.ll"))
                ir.write_text(text)
                run(
                    [
                        shutil.which("llc"),
                        "-march=aie2p",
                        "-enable-pipeliner=false",
                        "-stack-size-section",
                        "-filetype=obj",
                        "-o=" + str(obj),
                        str(ir),
                    ],
                    directory,
                )
        if helpers:
            pre = obj.with_suffix(".pre.o")
            obj.rename(pre)
            run(
                [
                    shutil.which("ld.lld"),
                    "-r",
                    str(pre),
                    *map(str, helpers),
                    "-o",
                    str(obj),
                ],
                directory,
            )
        paths[key] = obj
        objects[key + ".o"] = digest(obj)
        return obj

    for key in recipes["leaves"]:
        leaf(key)
    for name, recipe in recipes["devices"].items():
        directory = out / name
        directory.mkdir()
        if "design" in recipe:  # generated from an IRON design in src/graphs/families/
            sys.path.insert(0, str(ROOT / "src/graphs"))
            design, function = recipe["design"].split(".")
            text = getattr(importlib.import_module("families." + design), function)()
        else:
            text = (src / recipe["template"]).read_text()
        for placeholder, key in recipe["bindings"].items():
            if placeholder not in text:
                raise RuntimeError("missing link placeholder: " + placeholder)
            text = text.replace(placeholder, str(paths[key]))
        graph = directory / "device.mlir"
        graph.write_text(text)
        run(
            [
                shutil.which("aiecc"),
                "--no-progress",
                "--get-core-elfs",
                # Graph templates fix every buffer address; keep data-size measurement
                # from moving buffers away from them.
                "--no-measure-data-size",
                "--tmpdir=" + str(directory / "work"),
                str(graph),
            ],
            directory,
        )
        elfs = sorted(directory.glob("elfs_*/*.elf"))
        if len(elfs) != recipe.get("cores", 32):
            raise RuntimeError("unexpected attention core count for " + name)
        for p in elfs:
            cores[name + "/" + p.name] = {
                "path": str(p.relative_to(out)),
                "sha256": digest(p),
            }
    result = {
        "status": "built_not_qualified",
        "target": target,
        "seconds": time.monotonic() - started,
        "objects": objects,
        "commands": commands,
        "cores": cores,
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in src.rglob("*") if p.is_file()
        },
    }
    (out / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["attention_special"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = build(a.target, a.output)
    print(json.dumps({"target": a.target, "cores": len(result["cores"])}))


if __name__ == "__main__":
    main()
