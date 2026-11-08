"""Build supported laya_npu source targets into a new output directory.

The builder reads local sources and a declared external toolchain only.
Run audit.py separately to compare against frozen binary evidence.
"""

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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["residual"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError("output must not exist; preserve prior builds")
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (64 * 1024**3, 64 * 1024**3))
    os.environ["PYTHOC_LLC_FLAGS"] = "-enable-pipeliner=false -stack-size-section"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "laya_npu_residual_leaf", ROOT / "src/residual/leaf.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from aie.iron.pythoc.compiler import compile_pythoc_source

    fn = module.residual
    started = time.monotonic()
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
    source = (ROOT / "src/residual/device.mlir.in").read_text()
    if source.count("@RESIDUAL_OBJECT@") != 1:
        raise RuntimeError("unexpected core template")
    graph = out / "device.mlir"
    graph.write_text(source.replace("@RESIDUAL_OBJECT@", str(obj)))
    aiecc = shutil.which("aiecc")
    if not aiecc:
        raise RuntimeError("aiecc is unavailable")
    command = [
        aiecc,
        "--no-progress",
        "--get-core-elfs",
        # Graph templates fix every buffer address; keep data-size measurement
        # from moving buffers away from them.
        "--no-measure-data-size",
        f"--tmpdir={out}/work",
        str(graph),
    ]
    with (out / "aiecc.log").open("x") as log:
        subprocess.run(
            command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    cores = sorted(out.glob("elfs_*/*.elf"))
    if len(cores) != 1:
        raise RuntimeError(f"expected one residual core, found {len(cores)}")
    report = dict(
        status="built_not_qualified",
        target=args.target,
        command=command,
        seconds=time.monotonic() - started,
        python=sys.version,
        source_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in (ROOT / "src/residual").glob("*")
            if p.is_file()
        },
        aiecc_sha256=digest(aiecc),
        leaf={"path": str(obj.relative_to(out)), "sha256": digest(obj)},
        cores=[{"path": str(p.relative_to(out)), "sha256": digest(p)} for p in cores],
    )
    (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
