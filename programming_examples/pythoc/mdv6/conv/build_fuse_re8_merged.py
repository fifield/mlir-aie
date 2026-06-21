#!/usr/bin/env python3
"""Build the fused concat->c4 op as a MERGED ELF (xrt.elf-loadable, @main entry).

The M0 concat_proof IRON module is a single aie.device with an anonymous
aie.runtime_sequence — loadable only as a standalone xclbin (its own hw_context,
purely additive => wedges the at-ceiling mdv6 frame). Wrapping it through
build_merged's _rewrite_sub + _make_dispatcher_block turns it into the standard

  module { aie.device @sub0_concat {...} aie.device { @main { aiex.configure ... } } }

form, compiled with aiecc --generate-full-elf --expand-load-pdis. The result
loads via xrt.elf + xrt.ext.kernel("main") into run_tiled_mc._MERGED_KERNELS,
sharing the model's context budget so it can DISPLACE the GEMM-only c4 ELF
(merged_gemm_t24_ic512_oc256_kb32_p1_x1) one-for-one — context-neutral.

Dispatcher @main arg order matches concat_proof's runtime_sequence: (I, wt, out).
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

from aie.iron.device import NPU2  # noqa: E402
from aie2_concat_proof import concat_proof  # noqa: E402
from build_merged import (  # noqa: E402
    _rewrite_sub, _make_dispatcher_block, _stage_kernel_obj, _resolve_build_dir,
)

ELF_NAME = "fused_concat_c4_t24_ic512_oc256_kb32_p1_merged"


def build(H=20, W=20, q_ic=128, n_q=4, oc=256, tile_m=24, k_block=32,
          n_cores=32, build_dir=None):
    if build_dir is None:
        build_dir = _resolve_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    elf_path = os.path.join(build_dir, f"{ELF_NAME}.elf")
    if os.path.exists(elf_path):
        return elf_path

    os.environ["CONCAT"] = "1"
    module = concat_proof(NPU2(), H, W, q_ic, n_q, oc, tile_m, k_block, n_cores)
    assert module.operation.verify()
    sub_text = str(module)

    dev_sym = "sub0_concat_c4"
    seq_sym = "sub0_concat_c4_seq"
    rewritten, arg_types = _rewrite_sub(sub_text, dev_sym, seq_sym)
    dispatcher = _make_dispatcher_block([(dev_sym, seq_sym, arg_types)])
    merged_mlir = "module {\n" + rewritten + "\n" + dispatcher + "\n}\n"

    mlir_path = os.path.join(build_dir, f"{ELF_NAME}.mlir")
    with open(mlir_path, "w") as f:
        f.write(merged_mlir)

    _stage_kernel_obj("gemm_conv1x1_kblocked_bf16", build_dir)

    cmd = (
        f"cd {build_dir} && aiecc.py --no-aiesim --no-xchesscc --no-xbridge "
        f"--no-compile-host --generate-full-elf --expand-load-pdis "
        f"--full-elf-name={ELF_NAME}.elf {ELF_NAME}.mlir"
    )
    print(f"  {ELF_NAME}: compiling merged ELF ...", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or [""])[-1]
        print(f"  {ELF_NAME}: FAIL\n    {last}")
        return None
    print(f"  {ELF_NAME}: OK -> {elf_path}")
    return elf_path


if __name__ == "__main__":
    p = build()
    sys.exit(0 if p else 1)
