"""Fresh-output source build: kernels, packed data, and selected full-model graphs."""

import argparse, copy, hashlib, importlib.util, json, os, re, resource, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Core build targets per family; each is built by build_<family>.py (build.py for residual).
FAMILIES = {
    "residual": ["residual"],
    "scalar": ["cast8", "castrelu8", "lnf", "lnr8"],
    "decision": ["decision_k1024", "decision_k4096"],
    "dense": ["dense_native", "dense_fast", "dense_m2n2", "dense_m2n2_cb", "dense_native_s8", "dense_fast_s8"],
    "attention": ["attention_baseline"],
    "special": ["attention_special"],
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def emit(prefix, plan, types):
    parts = [prefix, "aie.device(npu2) @main {"]
    arenas = [f"memref<{n}xi8>" for n in plan["arenas"]]
    args = ", ".join(f"%arena{i}: {t}" for i, t in enumerate(arenas))
    for group in plan["groups"]:
        parts.append(f'aie.runtime_sequence @{group["instance"]}({args}) {{')
        for i in range(group["start"], group["end"]):
            d = plan["dispatches"][i]
            signature = types[d["sequence"]]
            if len(signature) != len(d["bindings"]):
                raise RuntimeError("sequence binding arity")
            for j, ((slot, offset, nbytes), typ) in enumerate(
                zip(d["bindings"], signature)
            ):
                a = plan["allocations"][slot]
                arena = a["arena"]
                off = a["offset"] + offset
                if not 0 <= offset <= offset + nbytes <= a["nbytes"]:
                    raise RuntimeError("binding outside allocation")
                if not 0 <= off <= off + nbytes <= plan["arenas"][arena]:
                    raise RuntimeError("binding outside arena")
                parts += [
                    f"%offset_{i}_{j} = arith.constant {off} : index",
                    f"%view_{i}_{j} = memref.view %arena{arena}[%offset_{i}_{j}][] : {arenas[arena]} to {typ}",
                ]
        cursor = group["start"]
        for command in plan["commands"]:
            if not group["start"] <= command["start"] < group["end"]:
                continue
            if command["start"] != cursor or command["end"] > group["end"]:
                raise RuntimeError("command partition mismatch")
            cursor = command["end"]
            dev = command["device"]
            # --expand-load-pdis starts every configure with a firmware reset (an
            # empty-PDI load, alternating so none is skipped), so a region needs no
            # reset or scrub of its own; one reset_b ends each runtime sequence.
            parts.append(f"aiex.configure @{dev} {{")
            for i in range(command["start"], command["end"]):
                d = plan["dispatches"][i]
                operands = ", ".join(
                    f"%view_{i}_{j}" for j in range(len(d["bindings"]))
                )
                parts.append(
                    f'aiex.run @{d["sequence"]}({operands}) : ('
                    + ", ".join(types[d["sequence"]])
                    + ")"
                )
            parts.append("}")
            if command["end"] == group["end"]:
                parts.append("aiex.configure @reset_b {} {expand_mode = 0 : i32}")
        if cursor != group["end"]:
            raise RuntimeError("incomplete group commands")
        parts.append("}")
    return "\n".join(parts + ["}", "}"]) + "\n"


def emit_target(prefix, plan, types, target):
    text = emit(prefix, plan, types)
    if target.startswith("mixed/"):
        alternate = copy.deepcopy(plan)
        alternate["mixed_execution_order"] = [1, 2]
        alternate["groups"][0]["instance"] = "mixed_b1_b2"
        for command in alternate["commands"]:
            start, end = command["start"], command["end"]
            alternate["dispatches"][start:end] = sorted(
                alternate["dispatches"][start:end],
                key=lambda d: 0 if d["sequence"].startswith("b1_") else 1,
            )
        second = emit("", alternate, types)
        second = second[second.index("aie.runtime_sequence") :]
        text = text.rstrip()[:-3] + second
    return text


def build(out, targets, pack=True):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (64 * 1024**3, 64 * 1024**3))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    cores = {}
    records = {}
    graphs = {}

    def run(command, log, cwd=ROOT):
        with log.open("x") as f:
            subprocess.run(
                list(map(str, command)),
                cwd=cwd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for family, names in FAMILIES.items():
        for target in names:
            directory = out / "cores" / target
            builder = "build.py" if family == "residual" else "build_" + family + ".py"
            print("building", family, target, flush=True)
            run(
                [
                    sys.executable,
                    ROOT / builder,
                    "--target",
                    target,
                    "--output",
                    directory,
                ],
                out / (target + ".log"),
            )
            record = json.loads((directory / "build.json").read_text())
            items = record["cores"]
            if isinstance(items, list):
                items = {str(i): v for i, v in enumerate(items)}
            for name, item in items.items():
                path = directory / item["path"]
                if sha(path) != item["sha256"]:
                    raise RuntimeError("core output changed")
                cores[target + "/" + name] = str(path)
            records[target] = record
    if pack:
        print("building packed model data", flush=True)
        run(
            [sys.executable, ROOT / "build_packing.py", "--output", out / "data"],
            out / "packing.log",
        )
        packing = json.loads((out / "data/build.json").read_text())
    else:
        packing = None
    recipes = json.loads((ROOT / "src/graphs/recipes.json").read_text())
    # Device templates are generated from the IRON design families (src/graphs/emit.py).
    print("emitting device templates", flush=True)
    sys.path.insert(0, str(ROOT / "src/graphs"))
    import emit as template_emitter

    templates = template_emitter.emit(out / "templates", sorted({recipes[t]["devices"] for t in targets}))
    for target in targets:
        print("building graph", target, flush=True)
        recipe = recipes[target]
        directory = out / "graphs" / target
        directory.mkdir(parents=True)
        prefix = templates[recipe["devices"]].read_text()
        # A recipe may swap one core build for another with the same core names,
        # e.g. the n2 fast dense cores for the m2n2 ones.
        swap = {}
        if "swap_cores" in recipe:
            old, new = recipe["swap_cores"]
            for name in records[old]["cores"]:
                swap[old + "/" + name] = cores[new + "/" + name]
        for core in recipe["cores"]:
            if core not in cores:
                raise RuntimeError("missing freshly built core " + core)
            path = swap.pop(core, cores[core])
            prefix = prefix.replace("@CORE_" + core + "@", path)
        if swap:
            raise RuntimeError("swapped cores missing from the recipe")
        if "@CORE_" in prefix:
            raise RuntimeError("unresolved source core")
        plan = json.loads((ROOT / "src/graphs" / recipe["plan"]).read_text())
        graph = directory / "graph.mlir"
        graph.write_text(emit_target(prefix, plan, recipe["sequences"], target))
        elf = directory / "graph.elf"
        command = [
            shutil.which("aiecc"),
            "--no-progress",
            "--get-full-elf",
            "--expand-load-pdis",
            f"--full-elf-name={elf}",
            f"--tmpdir={directory}/work",
            str(graph),
        ]
        run(command, directory / "compile.log", directory)
        # aiecc writes every PDI with revoke_id = 0, so the ELF is deterministic.
        spec = importlib.util.spec_from_file_location(
            "packaging_identity", ROOT / "src/packaging/identity.py"
        )
        identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity)
        wrapper_identity = identity.normalized(
            elf.read_bytes(), require_revoke_zero=True
        )
        # aiecc keeps the control streams only inside the ELF, which the
        # normalized identity covers.
        plan.update(
            elf=str(elf.relative_to(out)), elf_sha256=sha(elf), mlir_sha256=sha(graph)
        )
        (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        graphs[target] = {
            "elf_sha256": sha(elf),
            "elf_normalized": wrapper_identity,
            "plan": str((directory / "plan.json").relative_to(out)),
            "plan_sha256": sha(directory / "plan.json"),
            "command": command,
        }
        (out / "progress.json").write_text(
            json.dumps({"graphs": graphs}, indent=2) + "\n"
        )
    result = {
        "status": "built_not_qualified",
        "seconds": time.monotonic() - started,
        "templates": {name: str(path.relative_to(out)) for name, path in templates.items()},
        "cores": cores,
        "core_builds": records,
        "packing": packing,
        "graphs": graphs,
    }
    (out / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", nargs="+")
    p.add_argument("--skip-packing", action="store_true")
    a = p.parse_args()
    recipes = json.loads((ROOT / "src/graphs/recipes.json").read_text())
    targets = a.targets or list(recipes)
    if not set(targets) <= set(recipes):
        raise RuntimeError("unsupported target")
    r = build(a.output, targets, not a.skip_packing)
    print(json.dumps({"cores": len(r["cores"]), "graphs": len(r["graphs"])}))


if __name__ == "__main__":
    main()
