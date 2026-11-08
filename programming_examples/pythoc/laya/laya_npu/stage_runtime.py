"""Create a relocatable laya_npu runtime package from a rebuild.py output directory."""

import argparse, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2) + "\n")


def required_targets(policy):
    required = {
        mode + "/" + shape
        for mode, v in policy["variants"].items()
        for shape in v["eligible_shapes"]
    }
    required |= {
        "pair/" + mode + "/" + shape
        for mode, v in policy["phase_pairs"].items()
        for shape in v
    }
    return required | {
        "mixed/" + mode + "/" + shape
        for mode, v in policy["mixed_requests"].items()
        for shape in v
    }


def stage(build, out):
    build = Path(build).resolve()
    out = Path(out).resolve()
    record = json.loads((build / "build.json").read_text())
    policy = json.loads((ROOT / "src/runtime/policy.json").read_text())
    required = required_targets(policy)
    missing = required - set(record["graphs"])
    if missing:
        raise RuntimeError(
            "build lacks eligible graph targets: " + ", ".join(sorted(missing))
        )
    if not record.get("packing"):
        raise RuntimeError("build lacks packed model data")
    out.mkdir(parents=True, exist_ok=False)
    package = out / "laya_npu_runtime"
    shutil.copytree(
        ROOT / "src/runtime",
        package,
        ignore=shutil.ignore_patterns("__pycache__", "policy.json"),
    )
    data = package / "data"
    data.mkdir()
    shutil.copytree(ROOT / "inputs/model", data / "model")
    shutil.copytree(build / "data/blobs", data / "blobs")
    # Content identity of the device artifacts; part of the runtime identity and memo keys.
    artifacts = {
        "graphs": {t: record["graphs"][t]["elf_sha256"] for t in sorted(required)},
        "blobs": {
            k: v["compressed_sha256"]
            for k, v in sorted(record["packing"]["blobs"].items())
        },
    }
    bundle = {
        "version": "laya_npu",
        "release_status": "staged",
        "default_variant": "exact",
        "complete_shape_matrix": True,
        "complete_validation_matrix": False,
        "variants": {},
        "phase_pairs": {},
        "mixed_requests": {},
        "sources_sha256": hashlib.sha256(
            json.dumps(artifacts, sort_keys=True).encode()
        ).hexdigest(),
    }
    for target in sorted(required):
        result = record["graphs"][target]
        plan = json.loads((build / result["plan"]).read_text())
        elf = build / plan["elf"]
        if sha(elf) != result["elf_sha256"]:
            raise RuntimeError("built ELF changed after the build: " + target)
        dest = package / "graphs" / target
        dest.mkdir(parents=True)
        shutil.copyfile(elf, dest / "graph.elf")
        plan["elf"] = str((dest / "graph.elf").relative_to(package))
        write(dest / "plan.json", plan)
        item = {
            "path": str((dest / "plan.json").relative_to(package)),
            "sha256": sha(dest / "plan.json"),
        }
        if target.startswith("pair/"):
            _, mode, shape = target.split("/")
            bundle["phase_pairs"].setdefault(mode, {})[shape] = item
        elif target.startswith("mixed/"):
            _, mode, shape = target.split("/")
            metadata = policy["mixed_requests"][mode][shape]
            item.update(
                {k: v for k, v in metadata.items() if k not in ("path", "sha256")}
            )
            item["candidate_only"] = True
            bundle["mixed_requests"].setdefault(mode, {})[shape] = item
        else:
            mode, shape = target.split("/")
            v = policy["variants"][mode]
            entry = bundle["variants"].setdefault(
                mode,
                {
                    "arithmetic": v["arithmetic"],
                    "complete_shape_matrix": mode == "exact",
                    "plans": {},
                    "rejected_shapes": v["rejected_shapes"],
                },
            )
            entry["plans"][shape] = item
    mixed = bundle["mixed_requests"]["exact"]["b2_b1_l64"]["sha256"]
    # The runtime verifies these at load time. pins.py is generated last and anchors the bundle.
    files = {
        str(p.relative_to(package)): sha(p)
        for p in sorted(package.rglob("*"))
        if p.is_file()
        and (p.suffix == ".py" or p.is_relative_to(data / "model"))
        and p.name != "pins.py"
    }
    write(data / "host_sources.json", files)
    bundle["host_sources_sha256"] = sha(data / "host_sources.json")
    write(data / "bundle.json", bundle)
    (package / "pins.py").write_text(
        "BUNDLE_SHA256 = "
        + repr(sha(data / "bundle.json"))
        + "\nMIXED_PLAN_SHA256 = "
        + repr(mixed)
        + "\n"
    )
    return {"package": str(package), "targets": len(required)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--build", type=Path, required=True, help="rebuild.py output directory"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(stage(a.build, a.output)))


if __name__ == "__main__":
    main()
