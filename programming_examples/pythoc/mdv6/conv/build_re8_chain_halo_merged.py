#!/usr/bin/env python3
"""B2c3-1 — chain -> halo_c3 in ONE merged ELF (one hw_context).

Packs the re8 rn3 chain (padded-HWC producer) and the on-device halo-gather 3x3
conv (consumer) as TWO sub-devices inside ONE merged ELF, wired producer->
consumer so the chain's PAD(2)-padded HWC output buffer is the halo-conv's input
WITHOUT a host round-trip and WITHOUT host im2col:

    chain_links=[(0, 2, 1, 0)]   # halo.arg0 (in img) <- chain.arg2 (out img)

The two args are the SAME MLIR type (memref<50176xui16> = 28x28x64 PAD(2) HWC),
so build_merged's chain_link type-check passes with no on-device reformat.

Plumbing resolved here:
  #2 origin offset: the halo sub is generated with shift=PAD-1 so its TAP reads
     the chain's valid feature map (parked at [PAD:PAD+G]) at the pad-1 phase --
     the host-side SHIFT in test_halo_conv_seam_hw.py is GONE, baked into the
     consumer's gather TAP.
  #1 weight streaming: for oc<=64 the OC weight slot fits one L1 buffer; for the
     full re8 c3 OC=128 the per-oc-block streaming variant is built when
     stream_oc=True (see aie2_halo_conv stream path).

CONTEXT MODEL (the B2c3-1 headline):
  BEFORE: merged chain ELF (1 ctx) + halo-conv xclbin (1 ctx) = 2 hw_contexts,
          host bounces the 50176-u16 padded buffer between them + applies SHIFT.
  AFTER:  one merged ELF, 2 aiex.configure sub-devices = 1 hw_context; one
          on-device PDI swap replaces the host bounce, and the shift is baked.

Both sub-MLIRs are generated IN THIS PROCESS so their PythoC kernel .o temp dirs
(absolute link_with paths) are still alive when aiecc links the merged ELF.

Run:  source env.sh && python3 conv/build_re8_chain_halo_merged.py
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
from aie2_halo_conv import halo_conv, PAD


def build(geo="re8", n_iters=3, oc=32, build_dir=None, stream_oc=False):
    if build_dir is None:
        build_dir = _resolve_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    p = geo_params(geo)
    ic, G = p["IC"], p["GBOUND"]
    elf_name = f"{geo}_chain_halo_i{n_iters}_oc{oc}_merged"
    elf_path = os.path.join(build_dir, f"{elf_name}.elf")

    # ---- sub0: rn3 chain (producer; padded-HWC out at arg2) ----
    chain_mod = rn3_chain_geo(geo, n_iters=n_iters, rnm=0)
    assert chain_mod.operation.verify()
    chain_sym, chain_seq = "sub0_chain", "sub0_chain_seq"
    chain_rw, chain_args = _rewrite_sub(str(chain_mod), chain_sym, chain_seq)

    # ---- sub1: halo-gather 3x3 conv (consumer; in img at arg0) ----
    # shift=PAD-1 bakes the seam origin offset (plumbing #2).
    halo_mod, meta = halo_conv(ic=ic, oc=oc, gbound=G, shift=PAD - 1,
                               stream_oc=stream_oc)
    assert halo_mod.operation.verify()
    halo_sym, halo_seq = "sub1_halo", "sub1_halo_seq"
    halo_rw, halo_args = _rewrite_sub(str(halo_mod), halo_sym, halo_seq)

    # chain.arg2 (out img) and halo.arg0 (in img) must be the same MLIR type.
    if chain_args[2] != halo_args[0]:
        raise RuntimeError(
            f"seam type mismatch: chain.out={chain_args[2]} halo.in={halo_args[0]}")
    print(f"  chain args: {chain_args}")
    print(f"  halo  args: {halo_args}")
    print(f"  seam: chain.arg2 == halo.arg0 == {chain_args[2]}")

    subs = [(chain_sym, chain_seq, chain_args),
            (halo_sym, halo_seq, halo_args)]
    # halo.arg0 (in) <- chain.arg2 (out): device-resident handoff.
    dispatcher = _make_dispatcher_block(subs, chain_links=[(0, 2, 1, 0)])
    merged = "module {\n" + chain_rw + "\n" + halo_rw + "\n" + dispatcher + "\n}\n"

    mlir_path = os.path.join(build_dir, f"{elf_name}.mlir")
    with open(mlir_path, "w") as f:
        f.write(merged)

    cmd = (
        f"cd {build_dir} && aiecc.py --no-aiesim --no-xchesscc --no-xbridge "
        f"--no-compile-host --generate-full-elf --expand-load-pdis "
        f"--full-elf-name={elf_name}.elf {elf_name}.mlir"
    )
    print(f"  {elf_name}: compiling merged ELF ...", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n    ".join((r.stderr.strip().splitlines() or [""])[-12:])
        print(f"  {elf_name}: FAIL (rc={r.returncode})\n    {tail}")
        return None, meta
    print(f"  {elf_name}: OK -> {elf_path}")
    return elf_path, meta


if __name__ == "__main__":
    oc = int(os.environ.get("HC_OC", "32"))
    stream = os.environ.get("HC_STREAM_OC", "0") not in ("", "0", "false", "False")
    p, _ = build(oc=oc, stream_oc=stream)
    sys.exit(0 if p else 1)
