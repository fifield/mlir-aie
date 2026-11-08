"""Fetch the raw model checkpoint, which is too large to keep in git.

Downloads the Hugging Face revision recorded in manifests/model.json and verifies
its identity. A copy that is already present is verified and left alone.
"""

import argparse, hashlib, json, os, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    pin = json.loads((ROOT / "manifests/model.json").read_text())
    dest = ROOT / pin["path"]
    if dest.exists():
        if sha(dest) != pin["sha256"]:
            raise RuntimeError(
                f"{dest} differs from the pinned checkpoint; remove it to fetch again"
            )
        print("present and verified:", dest)
        return
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(pin["repository"], pin["file"], revision=pin["revision"])
    if sha(source) != pin["sha256"]:
        raise RuntimeError("downloaded checkpoint differs from the pin")
    partial = dest.with_name(dest.name + ".partial")
    shutil.copyfile(source, partial)
    os.replace(partial, dest)
    print("fetched and verified:", dest)


if __name__ == "__main__":
    main()
