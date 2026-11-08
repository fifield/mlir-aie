"""Add the rebuilt rejected fast shapes to a staged runtime for explicit research-only runs."""

import argparse, json, shutil
from pathlib import Path
from stage_runtime import sha, write


def stage_research(build, stage):
    build = Path(build).resolve()
    stage = Path(stage).resolve()
    record = json.loads((build / "build.json").read_text())
    package = stage / "laya_npu_runtime"
    bundle = json.loads((package / "data/bundle.json").read_text())
    plans = {}
    for shape in bundle["variants"]["fast"]["rejected_shapes"]:
        item = record["graphs"]["fast/" + shape]
        plan = json.loads((build / item["plan"]).read_text())
        elf = build / plan["elf"]
        if sha(elf) != item["elf_sha256"]:
            raise RuntimeError("built ELF changed after the build: fast/" + shape)
        dest = package / "research/fast" / shape
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(elf, dest / "graph.elf")
        plan["elf"] = str((dest / "graph.elf").relative_to(package))
        write(dest / "plan.json", plan)
        plans[shape] = {
            "path": str((dest / "plan.json").relative_to(package)),
            "sha256": sha(dest / "plan.json"),
        }
    write(
        stage / "research_plans.json",
        {"status": "explicit_research_only_ineligible", "plans": plans},
    )
    return plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--build", type=Path, required=True, help="rebuild.py output directory"
    )
    p.add_argument("--stage", type=Path, required=True)
    a = p.parse_args()
    print(
        len(stage_research(a.build, a.stage)),
        "excluded fast targets staged for research only",
    )


if __name__ == "__main__":
    main()
