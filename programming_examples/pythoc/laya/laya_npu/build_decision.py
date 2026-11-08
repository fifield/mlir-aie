"""Rebuild two decision-device families from local PythoC and MLIR sources."""

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
TARGETS = ("decision_k1024", "decision_k4096")


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
    src = ROOT / "src/decision"
    objects = {}
    started = time.monotonic()
    from aie.iron.pythoc.compiler import compile_pythoc_source

    for name in ["laya_gemm_native", "bf16_zero_kernel"]:
        spec = importlib.util.spec_from_file_location(
            "laya_npu_" + name, src / (name + ".py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        fn = getattr(mod, name)
        extra = (
            {"C_ELEMS_CONST": 2048, "vshuffle": mod.vshuffle}
            if name == "bf16_zero_kernel"
            else {
                k: getattr(mod, k)
                for k in (
                    "set_ctrl_reg",
                    "vector_cast",
                    "vshuffle",
                    "vector_extract",
                    "I512_I512_ACC1024_bf_mac_conf",
                )
            }
        )
        obj = Path(
            compile_pythoc_source(
                fn.__aie_source__,
                name,
                target_arch="aie2p",
                output_dir=out,
                optimization_level=2,
                extra_globals=extra,
            )
        )
        objects[obj.name] = digest(obj)
    text = (src / (target + ".mlir.in")).read_text()
    for name in ["laya_gemm_native", "bf16_zero_kernel"]:
        placeholder = "@" + name.upper() + "@"
        if text.count(placeholder) != 1:
            raise RuntimeError("unexpected link placeholder")
        text = text.replace(placeholder, str(out / (name + ".o")))
    graph = out / "device.mlir"
    graph.write_text(text)
    command = [
        shutil.which("aiecc"),
        "--no-progress",
        "--get-core-elfs",
        # Graph templates fix every buffer address; keep data-size measurement
        # from moving buffers away from them.
        "--no-measure-data-size",
        f"--tmpdir={out}/work",
        str(graph),
    ]
    with (out / "compile.log").open("x") as log:
        subprocess.run(
            command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    cores = sorted(out.glob("elfs_*/*.elf"))
    if len(cores) != 32:
        raise RuntimeError("expected 32 decision cores")
    result = {
        "status": "built_not_qualified",
        "target": target,
        "seconds": time.monotonic() - started,
        "objects": objects,
        "commands": [command],
        "cores": {
            p.name: {"path": str(p.relative_to(out)), "sha256": digest(p)}
            for p in cores
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in src.iterdir() if p.is_file()
        },
        "pythoc_optimization_level": 2,
        "pythoc_flags": os.environ["PYTHOC_LLC_FLAGS"],
    }
    (out / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=TARGETS, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = build(a.target, a.output)
    print(json.dumps({"target": a.target, "cores": len(result["cores"])}))


if __name__ == "__main__":
    main()
