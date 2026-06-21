#!/usr/bin/env python3
"""PoC-1 — on-device PRODUCER->CONSUMER operator stitching.

Builds a single merged ELF (one hw-context) holding two GEMM-conv1x1
sub-devices wired so that sub0's OUTPUT is sub1's INPUT, with the
intermediate buffer staying DEVICE-SIDE between the two ops (the host
never fills it after the producer writes it, and never reads it before
the consumer reads it).

This is distinct from the SHARED-INPUT chain_link in build_pair_rn1.py
(which aliases two subs' *inputs*: chain_links=[(0,0,1,0)]). Here we use a
PRODUCER->CONSUMER chain_link:

    chain_links=[(0, 2, 1, 0)]   # sub1.arg0 (in)  <-  sub0.arg2 (out)

So the dispatcher @main signature collapses from 6 host args
(in0,wt0,out0, in1,wt1,out1) to 5 (in0,wt0,out0, wt1,out1): the
intermediate `out0`/`in1` is a SINGLE dispatcher %arg shared by both subs.

Shape pinning (so producer.out layout == consumer.in layout bytewise, no
reformat between them):

  producer:  GEMM conv1x1  IC=64 -> OC=64,  tile_m=64, 32 cores, ppc=1
             out = 32*64*64 = 131072 u16  (arg2: memref<131072xui16>)
  consumer:  GEMM conv1x1  IC=64 -> OC=32,  tile_m=64, 32 cores, ppc=1
             in  = 32*64*64 = 131072 u16  (arg0: memref<131072xui16>)

Producer.out and consumer.in are bytewise-identical envelopes, so the
chain_link type-check (build_merged enforces equal MLIR types) passes and
no on-device reformat is needed for the mechanism PoC.

Build:
  source env.sh
  python3 conv/build_stitch_poc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gemm_conv1x1")))

from build_merged import build_merged, _resolve_build_dir

_GEMM_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "gemm_conv1x1", "aie2_gemm_conv1x1.py"))

# Pinned PoC shapes.
TILE_M = 64
N_CORES = 32
PPC = 1
IC = 64          # producer IC and consumer IC
MID = 64         # producer OC == consumer IC (the intermediate channel count)
OC = 32          # consumer OC

PRODUCER_ARGS = ["32", str(TILE_M), str(IC), str(MID), str(PPC), "0"]
CONSUMER_ARGS = ["32", str(TILE_M), str(MID), str(OC), str(PPC), "0"]

STITCH_ELF = "stitch_poc_prodcons"
# Standalone single-sub ELFs used to build the 2-dispatch baseline.
PROD_ELF = "stitch_poc_producer_x1"
CONS_ELF = "stitch_poc_consumer_x1"


def build_stitch_elf():
    """Merged producer->consumer ELF (one context, device-resident intermediate)."""
    elf = os.path.join(_resolve_build_dir(), f"{STITCH_ELF}.elf")
    if os.path.exists(elf):
        print(f"  {STITCH_ELF}: already built, skipping")
        return elf
    sub_names = ["stitch_prod", "stitch_cons"]
    path = build_merged(
        STITCH_ELF, sub_names, kind="gemm",
        sub_spec_overrides={
            sub_names[0]: (_GEMM_SCRIPT, PRODUCER_ARGS),
            sub_names[1]: (_GEMM_SCRIPT, CONSUMER_ARGS),
        },
        # PRODUCER->CONSUMER: sub1.arg0 (in) <- sub0.arg2 (out).
        chain_links=[(0, 2, 1, 0)],
    )
    return path


def build_single(out_name, sub_label, args):
    elf = os.path.join(_resolve_build_dir(), f"{out_name}.elf")
    if os.path.exists(elf):
        print(f"  {out_name}: already built, skipping")
        return elf
    return build_merged(
        out_name, [sub_label], kind="gemm",
        # arg0=in, arg1=wt, arg2=out (no share -> 3 dispatcher args)
        sub_spec_overrides={sub_label: (_GEMM_SCRIPT, args)},
    )


# ---------------------------------------------------------------------------
# No-op PDI-swap slope ELFs. These isolate the PURE on-device swap-mechanism
# cost (the canonical ~40 us figure) the same way llama32_1b's
# measure_dispatch_overhead.py does: tiny no-op sub-devices, a dispatcher that
# fires R alternating aiex.configure/aiex.run between two distinct sub-devices,
# compiled through the identical aiecc --expand-load-pdis path. The slope of
# wall-time vs R is the per-empty-swap cost.
# ---------------------------------------------------------------------------
def _noop_swap_mlir(run_count, distinct=2):
    """Emit a merged module: `distinct` no-op subs + a dispatcher firing
    `run_count` alternating aiex.configure/aiex.run (forces a PDI swap each
    time run_count cycles between two distinct sub-devices)."""
    subs = []
    for d in range(distinct):
        subs.append(
            f"  aie.device(npu2) @noop{d} {{\n"
            f"    aie.runtime_sequence @noop{d}_seq(%a: memref<1xui16>) {{\n"
            f"    }}\n"
            f"  }}"
        )
    body = []
    for i in range(run_count):
        d = i % distinct
        body.append(f"      aiex.configure @noop{d} {{")
        body.append(f"        aiex.run @noop{d}_seq(%arg0) : (memref<1xui16>)")
        body.append(f"      }}")
    disp = (
        "  aie.device(npu2) {\n"
        "    aie.runtime_sequence @main(%arg0: memref<1xui16>) {\n"
        + "\n".join(body) + ("\n" if body else "")
        + "    }\n"
        "  }"
    )
    return "module {\n" + "\n".join(subs) + "\n" + disp + "\n}\n"


def build_noop_swap(run_count, distinct=2):
    import subprocess
    bd = _resolve_build_dir()
    os.makedirs(bd, exist_ok=True)
    out_name = f"stitch_noop_r{run_count}_d{distinct}"
    elf = os.path.join(bd, f"{out_name}.elf")
    if os.path.exists(elf):
        return elf
    mlir_path = os.path.join(bd, f"{out_name}.mlir")
    with open(mlir_path, "w") as f:
        f.write(_noop_swap_mlir(run_count, distinct))
    cmd = (
        f"cd {bd} && aiecc.py --no-aiesim --no-xchesscc --no-xbridge "
        f"--no-compile-host --generate-full-elf --expand-load-pdis "
        f"--full-elf-name={out_name}.elf {out_name}.mlir"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {out_name}: FAIL\n    "
              + (r.stderr.strip().splitlines() or [""])[-1])
        return None
    return elf


NOOP_COUNTS = [0, 1, 2, 4, 8, 16]


def build_all():
    print("=== PoC-1 stitch: building merged + baseline ELFs ===")
    s = build_stitch_elf()
    p = build_single(PROD_ELF, "stitch_prod_solo", PRODUCER_ARGS)
    c = build_single(CONS_ELF, "stitch_cons_solo", CONSUMER_ARGS)
    print("  building no-op PDI-swap slope ELFs...")
    noop = [build_noop_swap(n) for n in NOOP_COUNTS]
    ok = all(x is not None for x in (s, p, c)) and all(x is not None for x in noop)
    print(f"=== build {'OK' if ok else 'FAILED'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if build_all() else 1)
