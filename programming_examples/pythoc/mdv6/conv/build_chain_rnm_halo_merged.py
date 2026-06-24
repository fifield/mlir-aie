#!/usr/bin/env python3
"""C3 — re8 chain -> rnm GEMM -> halo_c3 (the FULL x3rn->x3 hop) in ONE merged ELF.

The entire re8 RepNCSP bottleneck -> conv3 hop, device-resident in ONE
hw_context. THREE sub-devices threaded producer->consumer with NO host touch
between them:

  sub0 chain : rn3_chain_geo('re8', rnm=0, stack_x2_ch=HALF) -> PAD(2) padded
               28x28x64 chain output (lower half of a 100352 stacked BO).
  sub1 dcg   : depad_concat_gemm -> de-pad(chain) + concat(x2) -> rnm 1x1
               128->128 GEMM + BN + SiLU, drains the PAD(2)-padded 28x28x128 seam.
  sub2 halo  : halo_conv(ic=128, oc=128, shift=PAD-1, stream_oc='block') -> the
               c3 3x3 128->128 halo-gather conv reading the PAD(2) seam directly.

  chain_links = [(0, 2, 1, 0),   # dcg.in   (stacked) <- chain.out (B)
                 (1, 2, 2, 0)]   # halo.in  (seam)    <- dcg.seam (out)

CONTEXT MODEL:
  BEFORE: chain xclbin + rnm GEMM xclbin + c3 3x3 xclbin = 3 hw_contexts, with
          host concat + host repack between chain/rnm AND host tile->HWC + host
          im2col between rnm/c3 (the model's current x3rn->x3 hop).
  AFTER:  one merged ELF, 3 aiex.configure sub-devices = 1 hw_context; both
          seams device-resident, the c3 halo shift baked (shift=PAD-1).

Run:  source env.sh && python3 conv/build_chain_rnm_halo_merged.py
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
from aie2_depad_concat_gemm import depad_concat_gemm
from aie2_halo_conv import halo_conv, PAD


def build(geo="re8", n_iters=3, ic2=64, oc=128, gbound=20, build_dir=None,
          stream_oc="block"):
    if build_dir is None:
        build_dir = _resolve_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    elf_name = f"chain_rnm_halo_{geo}_i{n_iters}_oc{oc}_g{gbound}_merged"
    elf_path = os.path.join(build_dir, f"{elf_name}.elf")

    # The chain emits a PAD(2)-padded HWC image whose HEIGHT can EXCEED the
    # square IMG width when WORKER_TILES sums to more grid-rows than the valid
    # feature map (re6: 6 tile-rows -> 52-row tall buffer, vs 44-wide square).
    # The chain->x2 stacking boundary must be the chain's REAL IMG_ELEMS so x2
    # stacks above the tall chain image; the dcg de-pad gather still reads only
    # the valid rows at the IMG-wide row stride (junk rows never read).
    cp = geo_params(geo)
    half_elems = cp["IMG_ELEMS"]                  # chain tall IMG_ELEMS (re8 square: 50176)
    chain_img_h = cp["IMG_H"]                     # chain image height (re6: 52, re8: 28)

    # ---- sub0: rn3 chain (producer; output widened to hold x2 half) ----
    chain_mod = rn3_chain_geo(geo, n_iters=n_iters, rnm=0, stack_x2_ch=half_elems)
    assert chain_mod.operation.verify()
    c_sym, c_seq = "sub0_chain", "sub0_chain_seq"
    c_rw, c_args = _rewrite_sub(str(chain_mod), c_sym, c_seq)

    # ---- sub1: de-pad + concat -> rnm GEMM (consumer/producer; seam at arg2) ----
    dcg_mod, dmeta = depad_concat_gemm(ic2=ic2, oc=oc, gbound=gbound,
                                       chain_img_h=chain_img_h)
    assert dcg_mod.operation.verify()
    d_sym, d_seq = "sub1_dcg", "sub1_dcg_seq"
    d_rw, d_args = _rewrite_sub(str(dcg_mod), d_sym, d_seq)

    # ---- sub2: halo-gather 3x3 conv (consumer; in img at arg0, shift baked) ----
    halo_mod, hmeta = halo_conv(ic=oc, oc=oc, gbound=gbound, shift=PAD - 1,
                                stream_oc=stream_oc)
    assert halo_mod.operation.verify()
    h_sym, h_seq = "sub2_halo", "sub2_halo_seq"
    h_rw, h_args = _rewrite_sub(str(halo_mod), h_sym, h_seq)

    if c_args[2] != d_args[0]:
        raise RuntimeError(f"seam1 mismatch: chain.out={c_args[2]} dcg.in={d_args[0]}")
    if d_args[2] != h_args[0]:
        raise RuntimeError(f"seam2 mismatch: dcg.seam={d_args[2]} halo.in={h_args[0]}")
    print(f"  chain args: {c_args}")
    print(f"  dcg   args: {d_args}")
    print(f"  halo  args: {h_args}")
    print(f"  seam1: chain.arg2 == dcg.arg0 == {c_args[2]}")
    print(f"  seam2: dcg.arg2   == halo.arg0 == {d_args[2]}")

    subs = [(c_sym, c_seq, c_args), (d_sym, d_seq, d_args), (h_sym, h_seq, h_args)]
    # dcg.in <- chain.out ; halo.in <- dcg.seam : both seams device-resident.
    dispatcher = _make_dispatcher_block(subs, chain_links=[(0, 2, 1, 0), (1, 2, 2, 0)])
    merged = "module {\n" + c_rw + "\n" + d_rw + "\n" + h_rw + "\n" + dispatcher + "\n}\n"

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
        tail = "\n    ".join((r.stderr.strip().splitlines() or [""])[-25:])
        print(f"  {elf_name}: FAIL (rc={r.returncode})\n    {tail}")
        return None, dmeta, hmeta
    print(f"  {elf_name}: OK -> {elf_path}")
    return elf_path, dmeta, hmeta


if __name__ == "__main__":
    n_iters = int(os.environ.get("CRH_N_ITERS", "3"))
    p, _, _ = build(n_iters=n_iters)
    sys.exit(0 if p else 1)
