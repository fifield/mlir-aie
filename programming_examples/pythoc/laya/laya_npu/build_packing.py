"""Regenerate native and compact initializers from raw model data and local recipes."""

import argparse, hashlib, importlib.util, json, sys, time, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "src/packing" / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "blobs").mkdir()
    sys.dont_write_bytecode = True
    started = time.monotonic()
    native = module("native")
    compact = module("compact")
    src = ROOT / "src/packing"
    recipes = json.loads((src / "native_recipes.json").read_text())
    fixtures = json.loads((src / "fixtures.json").read_text())
    layouts = json.loads((src / "compact_layouts.json").read_text())
    # device_cb (fast B1/L64): compact engine weights re-laid out for the 4-column B1-split engine.
    cb_layouts = json.loads((src / "engine_cb_layouts.json").read_text())["weights"]
    engine_cb = module("engine_cb")
    # device_s8 (exact/b1_l64, B1/L64 pairs): the same compact weights re-laid out for the 8-column B1 split.
    s8_layouts = json.loads((src / "engine_s8_layouts.json").read_text())["weights"]
    maps = {
        seq: compact.mapping(v["records"], v["nbytes"])
        for seq, v in layouts["layouts"].items()
    }
    for seq, mapping in maps.items():
        if sha(mapping.tobytes()) != layouts["layouts"][seq]["permutation_sha256"]:
            raise RuntimeError("compact mapping identity mismatch")
    selected = {}
    for identity, recipe in recipes.items():
        r = dict(recipe)
        repeat = r.pop("repeat")
        selected.setdefault(json.dumps(r, sort_keys=True), []).append(
            (identity, repeat)
        )
    blobs = {}

    def emit(identity, raw, recipe):
        if sha(raw) != identity:
            raise RuntimeError("initializer identity mismatch: " + identity)
        path = out / "blobs" / (identity + ".z")
        packed = zlib.compress(raw, 1)
        path.write_bytes(packed)
        blobs[identity] = {
            "sha256": identity,
            "compressed_sha256": sha(packed),
            "nbytes": len(raw),
            "path": str(path.relative_to(out)),
            "recipe": recipe,
        }

    for recipe, value in native.generate(ROOT / "inputs/model/model.safetensors"):
        for identity, repeat in selected.get(json.dumps(recipe, sort_keys=True), []):
            raw = value.tobytes() * repeat
            emit(identity, raw, dict(recipe, repeat=repeat))
            if identity in layouts["weights"]:
                target = layouts["weights"][identity]
                packed = np.frombuffer(raw, "<u4")[maps[target["sequence"]]].tobytes()
                emit(
                    target["packed_initial"],
                    packed,
                    {
                        "kind": "compact",
                        "native": identity,
                        "sequence": target["sequence"],
                    },
                )
                cb = cb_layouts.get(target["packed_initial"])
                if cb is not None:
                    emit(
                        cb["initial"],
                        engine_cb.interleave(packed, cb["op"]),
                        {"kind": "engine_cb", "compact": target["packed_initial"], "op": cb["op"]},
                    )
                s8 = s8_layouts.get(target["packed_initial"])
                if s8 is not None:
                    emit(
                        s8["initial"],
                        engine_cb.interleave(packed, s8["op"], 8),
                        {"kind": "engine_s8", "compact": target["packed_initial"], "op": s8["op"]},
                    )
    from safetensors.torch import load_file

    weights = load_file(str(ROOT / "inputs/model/model.safetensors"))
    for identity, recipe in fixtures.items():
        emit(identity, native.fixture(recipe, weights).tobytes(), recipe)
    expected = (
        set(recipes)
        | set(fixtures)
        | {v["packed_initial"] for v in layouts["weights"].values()}
        | {v["initial"] for v in cb_layouts.values()}
        | {v["initial"] for v in s8_layouts.values()}
    )
    if set(blobs) != expected:
        raise RuntimeError("incomplete initializer recipe coverage")
    result = {
        "status": "built_not_qualified",
        "seconds": time.monotonic() - started,
        "blobs": blobs,
        "model_sha256": sha((ROOT / "inputs/model/model.safetensors").read_bytes()),
    }
    (out / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = build(a.output)
    print(json.dumps({"blobs": len(r["blobs"])}))


if __name__ == "__main__":
    main()
