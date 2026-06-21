#!/usr/bin/env python3
"""B2c-1 probe — wrap the re8 rn3 chain device through build_merged.

This is the first half of the B2c-1 keystone seam: take the standard single
`aie.device` emitted by ``rn3_chain_geo('re8', n_iters, rnm=0)`` (runtime
sequence ``(A: memref<50176xui16>, WT: memref<221952xui16>, B: memref<50176xui16>)``)
and run it through ``build_merged``'s ``_rewrite_sub`` + ``_make_dispatcher_block``
+ ``aiecc --generate-full-elf --expand-load-pdis`` path, producing ONE merged
ELF loadable via ``xrt.elf`` + ``xrt.ext.kernel("main")``.

This proves plumbing items #1 (BFP kernel .o staging) and #3 (migrate the chain
off ResidentXCLBinRunner onto the merged xrt.elf dispatch) are tractable, and is
the prerequisite for chain_link'ing the chain output into a consumer sub-device.

Run:  source env.sh && python3 conv/build_re8_chain_merged.py
"""
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MDV6 = _HERE.parent
for _p in (str(_HERE), str(_MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_merged import _rewrite_sub, _make_dispatcher_block, _resolve_build_dir
from aie2_rn3_chain_geo import rn3_chain_geo, geo_params


def build(geo="re8", n_iters=3, build_dir=None):
    if build_dir is None:
        build_dir = _resolve_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    elf_name = f"re8_chain_i{n_iters}_merged"
    elf_path = os.path.join(build_dir, f"{elf_name}.elf")

    module = rn3_chain_geo(geo, n_iters=n_iters, rnm=0)
    assert module.operation.verify()
    sub_text = str(module)

    dev_sym = "sub0_chain"
    seq_sym = "sub0_chain_seq"
    rewritten, arg_types = _rewrite_sub(sub_text, dev_sym, seq_sym)
    print(f"  chain sub arg types: {arg_types}")
    dispatcher = _make_dispatcher_block([(dev_sym, seq_sym, arg_types)])
    merged_mlir = "module {\n" + rewritten + "\n" + dispatcher + "\n}\n"

    mlir_path = os.path.join(build_dir, f"{elf_name}.mlir")
    with open(mlir_path, "w") as f:
        f.write(merged_mlir)

    cmd = (
        f"cd {build_dir} && aiecc.py --no-aiesim --no-xchesscc --no-xbridge "
        f"--no-compile-host --generate-full-elf --expand-load-pdis "
        f"--full-elf-name={elf_name}.elf {elf_name}.mlir"
    )
    print(f"  {elf_name}: compiling merged ELF ...", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n    ".join((r.stderr.strip().splitlines() or [""])[-8:])
        print(f"  {elf_name}: FAIL (rc={r.returncode})\n    {tail}")
        return None
    print(f"  {elf_name}: OK -> {elf_path}")
    return elf_path


if __name__ == "__main__":
    p = build()
    sys.exit(0 if p else 1)
