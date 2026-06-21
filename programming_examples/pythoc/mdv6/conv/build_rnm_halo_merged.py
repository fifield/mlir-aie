#!/usr/bin/env python3
"""S2 — rnm GEMM -> halo_c3 in ONE merged ELF (one hw_context). The REAL model
seam: RepNCSP conv3 (rnm, 1x1 128->128) feeds run_re_mc c3 (3x3 128->128). Today
the two are separated by a host tile->HWC reassembly + host im2col; here they are
two sub-devices inside ONE merged ELF, wired producer->consumer so the rnm GEMM's
PAD(2)-padded HWC output IS the halo_c3 input with NO host round-trip and NO host
im2col:

    chain_links=[(0, 2, 1, 0)]   # halo.arg0 (in img) <- gemm.arg2 (out seam img)

Both args are the SAME MLIR type (memref<IMG*IMG*OC xui16> = 28x28x128 PAD(2)
HWC), so build_merged's chain_link type-check passes with no on-device reformat.

  - sub0 (producer): aie2_gemm_pad_out.gemm_pad_out — 1x1 GEMM IC->OC whose drain
    writes the PAD(2)-padded HWC seam (interior placed, border zero). [S1, proven]
  - sub1 (consumer): aie2_halo_conv.halo_conv(stream_oc="block") — OC=128 halo-
    gather 3x3 reading PAD(2)-padded HWC. shift=PAD-1 bakes the seam origin so NO
    host shift is needed. [keystone, proven OC=128]

CONTEXT MODEL (the S2 headline):
  BEFORE: rnm GEMM xclbin (1 ctx) + halo_c3 xclbin (1 ctx) = 2 hw_contexts, host
          bounces the 100352-u16 padded buffer between them (+ host shift).
  AFTER:  one merged ELF, 2 aiex.configure sub-devices = 1 hw_context; one
          on-device PDI swap replaces the host bounce, and the shift is baked.

Run:  source env.sh && python3 conv/build_rnm_halo_merged.py
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
from aie2_gemm_pad_out import gemm_pad_out
from aie2_halo_conv import halo_conv, PAD


def build(ic=128, oc=128, gbound=20, build_dir=None, stream_oc="block"):
    if build_dir is None:
        build_dir = _resolve_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    elf_name = f"rnm_halo_ic{ic}_oc{oc}_g{gbound}_merged"
    elf_path = os.path.join(build_dir, f"{elf_name}.elf")

    # ---- sub0: rnm 1x1 GEMM (producer; padded-HWC seam out at arg2) ----
    gemm_mod, gmeta = gemm_pad_out(ic=ic, oc=oc, gbound=gbound)
    assert gemm_mod.operation.verify()
    g_sym, g_seq = "sub0_rnm", "sub0_rnm_seq"
    g_rw, g_args = _rewrite_sub(str(gemm_mod), g_sym, g_seq)

    # ---- sub1: halo-gather 3x3 conv (consumer; in img at arg0) ----
    # shift=PAD-1 bakes the seam origin offset (no host shift).
    halo_mod, hmeta = halo_conv(ic=oc, oc=oc, gbound=gbound, shift=PAD - 1,
                                stream_oc=stream_oc)
    assert halo_mod.operation.verify()
    h_sym, h_seq = "sub1_halo", "sub1_halo_seq"
    h_rw, h_args = _rewrite_sub(str(halo_mod), h_sym, h_seq)

    # gemm.arg2 (out seam) and halo.arg0 (in img) must be the SAME MLIR type.
    if g_args[2] != h_args[0]:
        raise RuntimeError(
            f"seam type mismatch: gemm.out={g_args[2]} halo.in={h_args[0]}")
    print(f"  gemm args: {g_args}")
    print(f"  halo args: {h_args}")
    print(f"  seam: gemm.arg2 == halo.arg0 == {g_args[2]}")

    subs = [(g_sym, g_seq, g_args), (h_sym, h_seq, h_args)]
    # halo.arg0 (in) <- gemm.arg2 (out seam): device-resident handoff.
    dispatcher = _make_dispatcher_block(subs, chain_links=[(0, 2, 1, 0)])
    merged = "module {\n" + g_rw + "\n" + h_rw + "\n" + dispatcher + "\n}\n"

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
        tail = "\n    ".join((r.stderr.strip().splitlines() or [""])[-15:])
        print(f"  {elf_name}: FAIL (rc={r.returncode})\n    {tail}")
        return None, gmeta, hmeta
    print(f"  {elf_name}: OK -> {elf_path}")
    return elf_path, gmeta, hmeta


if __name__ == "__main__":
    ic = int(os.environ.get("RNM_IC", "128"))
    oc = int(os.environ.get("RNM_OC", "128"))
    gb = int(os.environ.get("RNM_GBOUND", "20"))
    p, _, _ = build(ic=ic, oc=oc, gbound=gb)
    sys.exit(0 if p else 1)
