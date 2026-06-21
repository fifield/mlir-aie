#!/usr/bin/env python3
"""C2 — re8 rn3 chain -> [de-pad+concat] -> rnm 1x1 GEMM in ONE merged ELF.

The chain->rnm device-resident seam: the rn3 chain (re8, BFP576 conv1+conv2res)
produces the bottleneck output as a PAD(2)-padded HWC image (28x28x64). The model
then does concat(depad(chain), x2) = 20x20x128 and feeds the rnm GEMM (RepNCSP
conv3, 1x1 128->128 + BN + SiLU). Here those are TWO sub-devices in ONE merged ELF,
wired producer->consumer so the chain's padded output threads device-side into the
depad_concat_gemm's stacked input with NO host touch:

    chain_links=[(0, 2, 1, 0)]   # dcg.arg0 (stacked in) <- chain.arg2 (B output)

The chain is built with stack_x2_ch=HALF_ELEMS so its output BO is widened to
100352 = [chain_padded(50176) | x2_padded(50176)]; the chain writes only the lower
half (drain offsets unchanged), and the consumer reads the whole stacked BO. The
x2 half arrives as a SEPARATE host load into [50176:] of the shared BO (it is not
on the chain's compute path). The depad+concat is the consumer's input gather TAP
(de-pad = PAD offset + IMG*ic2 row stride; concat = HALF_ELEMS half stride), so the
chain's 64-ch padded output IS the 128-ch rnm input with no on-device reformat.

  - sub0 (producer): rn3_chain_geo('re8', n_iters, rnm=0, stack_x2_ch=50176)
  - sub1 (consumer): depad_concat_gemm(ic2=64, oc=128, gbound=20)

CONTEXT MODEL:
  BEFORE: chain xclbin (1 ctx) + rnm GEMM xclbin (1 ctx) = 2 hw_contexts, host
          bounces the chain output + host concat + host per-core repack.
  AFTER:  one merged ELF, 2 aiex.configure sub-devices = 1 hw_context; the
          on-device PDI swap replaces the host bounce + concat + repack.

Run:  source env.sh && python3 conv/build_chain_rnm_merged.py
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
from aie2_rn3_chain_geo import rn3_chain_geo
from aie2_depad_concat_gemm import depad_concat_gemm


def build(geo="re8", n_iters=3, ic2=64, oc=128, gbound=20, build_dir=None):
    if build_dir is None:
        build_dir = _resolve_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    elf_name = f"chain_rnm_{geo}_i{n_iters}_oc{oc}_g{gbound}_merged"
    elf_path = os.path.join(build_dir, f"{elf_name}.elf")

    # ---- sub0: rn3 chain (producer; padded chain out widened to hold x2 half) ----
    half_elems = ((gbound + 7) // 8 * 8 + 4) ** 2 * ic2   # IMG*IMG*ic2 (re8: 50176)
    chain_mod = rn3_chain_geo(geo, n_iters=n_iters, rnm=0, stack_x2_ch=half_elems)
    assert chain_mod.operation.verify()
    c_sym, c_seq = "sub0_chain", "sub0_chain_seq"
    c_rw, c_args = _rewrite_sub(str(chain_mod), c_sym, c_seq)

    # ---- sub1: de-pad + concat -> rnm GEMM (consumer; stacked in at arg0) ----
    dcg_mod, dmeta = depad_concat_gemm(ic2=ic2, oc=oc, gbound=gbound)
    assert dcg_mod.operation.verify()
    d_sym, d_seq = "sub1_dcg", "sub1_dcg_seq"
    d_rw, d_args = _rewrite_sub(str(dcg_mod), d_sym, d_seq)

    # chain.arg2 (B output, 100352) and dcg.arg0 (stacked in, 100352) must match.
    if c_args[2] != d_args[0]:
        raise RuntimeError(
            f"seam type mismatch: chain.out={c_args[2]} dcg.in={d_args[0]}")
    print(f"  chain args: {c_args}")
    print(f"  dcg   args: {d_args}")
    print(f"  seam: chain.arg2 == dcg.arg0 == {c_args[2]} "
          f"(HALF_ELEMS={dmeta['HALF_ELEMS']})")

    subs = [(c_sym, c_seq, c_args), (d_sym, d_seq, d_args)]
    # dcg.arg0 (stacked in) <- chain.arg2 (B output): device-resident handoff.
    dispatcher = _make_dispatcher_block(subs, chain_links=[(0, 2, 1, 0)])
    merged = "module {\n" + c_rw + "\n" + d_rw + "\n" + dispatcher + "\n}\n"

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
        tail = "\n    ".join((r.stderr.strip().splitlines() or [""])[-20:])
        print(f"  {elf_name}: FAIL (rc={r.returncode})\n    {tail}")
        return None, dmeta
    print(f"  {elf_name}: OK -> {elf_path}")
    return elf_path, dmeta


if __name__ == "__main__":
    n_iters = int(os.environ.get("CRM_N_ITERS", "3"))
    p, _ = build(n_iters=n_iters)
    sys.exit(0 if p else 1)
