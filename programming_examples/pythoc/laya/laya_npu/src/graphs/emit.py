"""Emit the graph device templates from the IRON design families.

A graph recipe (recipes.json) names one template: a module holding the devices the graph uses, each with
its runtime sequences in NPU-instruction form, then the reset_b device. composition.json declares, per
template, the devices in order, the core build of their core tiles (a path under the build's cores/ with
{tile} for the tile, per column range where a device mixes builds), and the sequences each device carries
(name, or name:fast for the fast variant). Everything else comes from the families in
families/: each module defines

  DEVICES:   {device_name: build}              build() -> MLIR text of an IRON design
  SEQUENCES: [(sequence_name, variant, build)]  build() -> MLIR text whose one runtime sequence is that one

A device is lowered with aiecc's input-with-addresses pipeline without routing (aiecc routes when it
compiles the graph), and its core bodies are replaced by the placeholders that rebuild.py binds to freshly
built core ELFs. A sequence is lowered to blockwrite/address_patch/write32/maskwrite32/sync instructions.

  python src/graphs/emit.py OUT_DIR [TEMPLATE ...]     # from laya_npu/, SDK environment active
"""

import importlib, json, os, re, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

COMPOSITION = HERE / "composition.json"
FAMILIES = sorted(p.stem for p in (HERE / "families").glob("*.py")
                  if not p.stem.startswith("_") and hasattr(importlib.import_module(f"families.{p.stem}"), "DEVICES"))
# device_05f (M37) is device_05 with other cores; the composition binds them.
ALIASES = {"device_05f": "device_05"}

PIPELINE = (
    "builtin.module(aie.device(aie-place-tiles,aie-objectFifo-stateful-transform,aie-objectFifo-unroll,"
    "aie-normalize-dma-bd-dims,aie-assign-lock-ids,aie-reserve-runtime-bd-ids,aie-assign-bd-ids,"
    "aie-lower-cascade-flows,aie-lower-broadcast-packet,aie-lower-multicast,aie-assign-tile-controller-ids),"
    "aie-generate-column-control-overlay{route-shim-to-tile-ctrl=false},"
    "aie.device(aie-prepare-buffers,aie-assign-buffer-addresses))"
)
# Laya's sequences never fill a DMA task queue, so they carry no queue-depth polls.
LOWER = [
    "--aie-materialize-bd-chains", "--aie-substitute-shim-dma-allocations",
    "--aie-unroll-runtime-sequence-loops", "--canonicalize", "--aie-normalize-dma-bd-dims",
    "--aie-decompose-large-dma-bd", "--aie-assign-runtime-sequence-bd-ids=enforce-queue-depth=false",
    "--aie-dma-tasks-to-npu", "--aie-dma-to-npu=enforce-queue-depth=false", "--aie-lower-set-lock", "--strip-debuginfo",
]
SEQUENCE = r"aie\.runtime_sequence(?: @\w+)?\s*\("


def family(name):
    return importlib.import_module(f"families.{name}")


@lru_cache(None)
def builders():
    devices, seqs = {}, {}
    for fam in FAMILIES:
        mod = family(fam)
        for name in mod.DEVICES:
            devices[name] = fam
        for i, (name, variant, _) in enumerate(mod.SEQUENCES):
            seqs[(name, variant)] = (fam, i)
    for alias, name in ALIASES.items():
        devices[alias] = devices[name]
    return devices, seqs


def opt(text, *args):
    r = subprocess.run([shutil.which("aie-opt"), *args, "-"], input=text, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def block_end(text, start):
    """Index just past the brace block that opens at or after `start`."""
    i = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return i


def drop_blocks(text, pattern):
    out, i = [], 0
    for m in re.finditer(pattern, text):
        if m.start() < i:
            continue
        out.append(text[i: m.start()])
        i = block_end(text, m.end())
    out.append(text[i:])
    return "".join(out)


def empty_cores(text):
    out, i = [], 0
    for m in re.finditer(r"aie\.core\(%\w+\)\s*\{", text):
        if m.start() < i:
            continue
        out.append(text[i: m.end()] + "\n      aie.end\n    }")
        i = block_end(text, m.start())
    out.append(text[i:])
    return "".join(out)


def core_placeholder(cores, tile):
    """@CORE_<build>/<elf>@, which rebuild.py replaces with the freshly built core ELF."""
    if isinstance(cores, dict):
        col = int(tile.split("_")[0])
        (cores,) = [p for r, p in cores.items() if int(r.split("-")[0]) <= col <= int(r.split("-")[1])]
    return "@CORE_" + cores.format(tile=tile) + "@"


def device_text(job):
    """Template form of one device: lowered, no sequences, core bodies bound to core-build placeholders."""
    name, cores = job
    cores = json.loads(cores)
    build = family(builders()[0][name]).DEVICES[ALIASES.get(name, name)]
    lowered = opt(opt(build(), f"--pass-pipeline={PIPELINE}"), "--strip-debuginfo")
    m = re.search(r"aie\.device\(npu2\w*\)(?: @\w+)?\s*\{", lowered)
    block = lowered[m.start(): block_end(lowered, m.start())]
    block = re.sub(r"aie\.device\(npu2\w*\)(?: @\w+)?", f"aie.device(npu2) @{name}", block, count=1)
    block = drop_blocks(block, SEQUENCE)
    block = empty_cores(block)
    block = re.sub(r',?\s*link_with = "[^"]*"', "", block)
    block = re.sub(r" attributes \{\s*\}", "", block)

    def bind(m):
        tile, attrs = m[1], (m[2] or "").strip()
        return (f'aie.core(%tile_{tile}) {{\n      aie.end\n    }} {{elf_file = "{core_placeholder(cores, tile)}"'
                + (f", {attrs}" if attrs else "") + "}")

    block = re.sub(r"aie\.core\(%tile_(\d+_\d+)\) \{\n      aie\.end\n    \}(?: \{([^}]*)\})?", bind, block)
    if len(re.findall(r"elf_file", block)) != len(re.findall(r"aie\.core\(", block)):
        raise RuntimeError(f"{name}: unbound core")
    return block[: block.rindex("}")].rstrip() + "\n"


def sequence_text(job):
    """Template form of one lowered sequence: its globals (renamed per sequence), then the sequence."""
    fam, i, name = job
    _, _, build = family(fam).SEQUENCES[i]
    physical = opt(opt(build(), f"--pass-pipeline={PIPELINE}"), *LOWER)
    (seq,) = [physical[m.start(): block_end(physical, m.end())] for m in re.finditer(SEQUENCE, physical)]
    types = re.findall(r"%arg\d+: (memref<[^>]+>)", seq.split("{")[0])
    seq = re.sub(r"aie\.runtime_sequence(?:\s+@\w+)?\s*\(", f"aie.runtime_sequence @{name}(", seq, count=1)

    def buffer(m):
        plus, typ, attrs = m.groups()
        idx = int(re.search(r"arg_idx = (\d+) : i32", attrs)[1])
        attrs = re.sub(r",?\s*arg_idx = \d+ : i32,?", "", attrs).strip(", ")
        return f"aiex.npu.address_patch({plus} : {typ}) buffer %arg{idx} : {types[idx]} {{{attrs}}}"

    seq = re.sub(r"aiex\.npu\.address_patch\((%\w+) : (i\d+)\) \{([^}]+)\}", buffer, seq)
    globals_ = [l.strip() for l in physical.splitlines() if l.strip().startswith("memref.global ")]
    text = "\n".join(globals_) + "\n" + seq + "\n"
    for symbol in re.findall(r"memref\.global[^@]*@([\w.$]+)", "\n".join(globals_)):
        text = re.sub(r"@" + re.escape(symbol) + r"\b", f"@{name}_{symbol}", text)
    return text


def sequence_job(entry):
    """name or name:fast. The mixed route's b1_/b2_ sequences are the plain ones under another name."""
    name, _, variant = entry.partition(":")
    variant = variant or "exact"
    base = name[3:] if name.startswith(("b1_", "b2_")) else name
    fam, i = builders()[1][(base, variant)]
    return fam, i, name


def emit(out, names=None, workers=None):
    """Write the named templates (default: all) into `out` as <name>.mlir; returns their paths."""
    composition = json.loads(COMPOSITION.read_text())
    names = list(composition) if names is None else list(names)
    dev_jobs, seq_jobs = {}, {}
    for name in names:
        for e in composition[name]:
            key = (e["device"], json.dumps(e["cores"], sort_keys=True))
            dev_jobs[key] = key
            for s in e["sequences"]:
                seq_jobs[s] = sequence_job(s)
    workers = workers or min(32, os.cpu_count() or 1)
    with ProcessPoolExecutor(workers) as pool:
        dev_text = dict(zip(dev_jobs, pool.map(device_text, dev_jobs.values())))
        seq_text = dict(zip(seq_jobs, pool.map(sequence_text, seq_jobs.values(), chunksize=1)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names:
        parts = ["module {"]
        for e in composition[name]:
            parts.append(dev_text[(e["device"], json.dumps(e["cores"], sort_keys=True))])
            parts += [seq_text[s] for s in e["sequences"]]
            parts.append("}")
        # Left open: rebuild.emit appends the @main device and closes the module.
        parts.append("aie.device(npu2) @reset_b {}")
        paths[name] = out / f"{name}.mlir"
        paths[name].write_text("\n".join(parts))
    return paths


if __name__ == "__main__":
    written = emit(sys.argv[1], sys.argv[2:] or None)
    print(f"wrote {len(written)} templates to {sys.argv[1]}")
