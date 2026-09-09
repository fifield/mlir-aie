#!/usr/bin/env python3
"""Compile-only TCT-route reductions; NEVER executable designs.

Remove only selected explicit TileControl packet_flow operations from the
addressed numerical phase-A gather IR. Runtime waits are deliberately unchanged:
these variants have disconnected completion routes and MUST NOT run on hardware.
This tool invokes aie-opt's routing pass only, never aiecc, xclbin tools, or NPU.

After sourcing env.sh, reproduce the smallest observed failure and a larger
passing placement (use a NEW output directory):

  python sppelan/reduce_phase_a_gather_routes.py INPUT_WITH_ADDRESSES.mlir OUTDIR \
      --case min_pair --case passing_triple

Tested with the frozen distinct-ID phase-A-gather candidate: min_pair keeps core
(0,4),(0,5), while passing_triple keeps (0,2),(0,3),(0,5). The latter passing
does not establish routability of the original 20-route design. Other cases
retain no explicit routes, four memtiles, one core per column, or one column.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

FLOW = re.compile(r"^    aie\.packet_flow\((?:0x[0-9a-fA-F]+|\d+)\) \{\n.*?^    \}(?: \{[^\n]*\})?\n", re.M | re.S)
SOURCE = re.compile(r"aie\.packet_source<%(\w+), TileControl : 0>")
EXPECTED = {f"mem_tile_{c}_1" for c in range(4)} | {
    f"tile_{c}_{r}" for c in range(4) for r in range(2,6)}
CASES = {
    "none": set(),
    "mt_only": {f"mem_tile_{c}_1" for c in range(4)},
    "core_row2": {f"tile_{c}_2" for c in range(4)},
    "core_single": {"tile_0_2"},
    "core_col0": {f"tile_0_{r}" for r in range(2,6)},
    "min_pair": {"tile_0_4", "tile_0_5"},
    "passing_triple": {"tile_0_2", "tile_0_3", "tile_0_5"},
    "full": EXPECTED,
}
WARNING = "NON-EXECUTABLE ROUTING DIAGNOSTIC: required runtime TCT waits remain unchanged"


def reduce_routes(text, selected):
    """Preserve every byte outside removed explicit route blocks."""
    records = []
    for match in FLOW.finditer(text):
        source = SOURCE.search(match.group())
        if source and not source.group(1).startswith("shim_"):
            records.append((match.start(),match.end(),source.group(1)))
    names = [name for _,_,name in records]
    if len(names) != 20 or set(names) != EXPECTED:
        raise ValueError("input must contain exactly the expected 20 explicit core/memtile TCT routes")
    if not selected <= EXPECTED:
        raise ValueError("unknown selected route source")
    result = text
    for start,end,name in reversed(records):
        if name not in selected:
            result = result[:start] + result[end:]
    return result


def self_test():
    blocks = []
    for name in sorted(EXPECTED):
        blocks.append(f"    aie.packet_flow(26) {{\n      aie.packet_source<%{name}, TileControl : 0>\n      aie.packet_dest<%shim_noc_tile_0_0, South : 0>\n    }} {{keep_pkt_header = true}}\n")
    shim = "    aie.packet_flow(15) {\n      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>\n      aie.packet_dest<%shim_noc_tile_0_0, South : 0>\n    }\n"
    original = "PREFIX\n" + "".join(blocks) + shim + "UNCHANGED_RUNTIME\n"
    if reduce_routes(original,EXPECTED) != original:
        raise RuntimeError("full-case identity failed")
    if reduce_routes(original,set()) != "PREFIX\n" + shim + "UNCHANGED_RUNTIME\n":
        raise RuntimeError("route-only deletion failed")
    reduced = reduce_routes(original,CASES["min_pair"])
    if len(FLOW.findall(reduced)) != 3 or "UNCHANGED_RUNTIME" not in reduced:
        raise RuntimeError("minimum-pair selection failed")
    try:
        reduce_routes(original.replace(blocks[0],"",1),set())
    except ValueError:
        pass
    else:
        raise RuntimeError("missing-source guard failed")
    print("PASS: 4 parser/identity/selection/guard checks (also under python -O)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("addressed_ir",nargs="?",type=Path)
    parser.add_argument("output_dir",nargs="?",type=Path,help="must not already exist")
    parser.add_argument("--case",action="append",choices=sorted(CASES),dest="cases")
    parser.add_argument("--timeout",type=int,default=210,help="seconds per routing case, max300")
    parser.add_argument("--jobs",type=int,choices=(1,2),default=2)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.addressed_ir is None or args.output_dir is None or not args.cases:
        parser.error("addressed_ir, NEW output_dir, and at least one --case are required")
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be1..300 seconds")
    executable = shutil.which("aie-opt")
    if executable is None:
        parser.error("aie-opt missing; source the configured environment first")
    text = args.addressed_ir.read_text()
    cases = list(dict.fromkeys(args.cases))
    generated = {name:reduce_routes(text,CASES[name]) for name in cases}
    args.output_dir.mkdir()  # Intentionally refuse overwrite of prior evidence.
    manifest = {"warning":WARNING,"source":str(args.addressed_ir.resolve()),
                "sha256":hashlib.sha256(text.encode()).hexdigest(),"aie_opt":executable,
                "cases":{name:{"kept":sorted(CASES[name]),"removed":20-len(CASES[name])} for name in cases}}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.output_dir/"DO_NOT_EXECUTE.txt").write_text(WARNING+"\n")
    for name,reduced in generated.items():
        (args.output_dir/f"{name}.mlir").write_text(reduced)

    def run(name):
        command = [executable,"--mlir-print-ir-after-failure","--mlir-disable-threading",
                   "--pass-pipeline=builtin.module(aie.device(aie-create-pathfinder-flows))",
                   str(args.output_dir/f"{name}.mlir"),"-o",str(args.output_dir/f"{name}.routed.mlir")]
        start = time.monotonic()
        with (args.output_dir/f"{name}.log").open("w") as log:
            try:
                code = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        result = {"case":name,"exit_code":code,"seconds":round(time.monotonic()-start,2),
                  "explicit_routes":len(CASES[name]),"command":command}
        (args.output_dir/f"{name}.result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run,cases))
    (args.output_dir/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    return int(any(result["exit_code"] != 0 for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
