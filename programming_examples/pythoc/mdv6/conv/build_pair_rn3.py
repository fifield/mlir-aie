#!/usr/bin/env python3
"""Phase H step 1 — conv-pair prototype for the rn3 layers.

Builds a TWO-sub-device merged ELF where:
  sub_0 = mc_*_rn3 (conv1 of the bottleneck pair)
  sub_1 = mc_*_rn3 (conv2 of the bottleneck pair, same kernel/shape)
  chain_links aliases sub_1's input BO to sub_0's output BO.

Dispatcher @main signature (after chain_link aliasing):
  arg0: sub0 in (= patch-format input)
  arg1: sub0 wt (= conv1 weights — fuse_repconv output)
  arg2: sub0 out / sub1 in (= shared BO via chain_link)
  arg3: sub1 wt (= conv2 weights — fuse_bn output)
  arg4: sub1 out (= final conv2 output)

KNOWN LIMITATION (Phase H step 1 scope):
  sub_0 writes tile-format output to arg2; sub_1 reads arg2 as patch-format
  input. **Layouts don't match** — sub_1 will produce wrong results.

  This prototype is intentional: it validates the conv-pair dispatcher
  pattern compiles and runs end-to-end at the IR level, while leaving
  the format-conversion problem explicit for step 2.

  Step 2 will inject a memtile-internal DMA between sub_0's output and
  sub_1's input that reformats tile→patch-with-halo. Either via:
    (a) a third sub-device whose only job is the reformat, or
    (b) merging sub_0 and sub_1 into ONE aie.device with internal
        memtile L2 fifo + halo TAP on the read side.

  (b) is the llama-style fused-device target. (a) is the cheaper
  staging step.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_merged import build_merged

# (label, mc_config_name) — pulls shape from build_multicore.CONFIGS
_RN3_PAIRS = [
    ("re8_rn3", "mc_re8_rn3"),
    # re6_rn3 / re4_rn3 deferred until step 1 validates
]


def _build_one(label, mc_name):
    out_name = f"merged_{label}_pair_x1"
    bd = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_merged"))
    elf_path = os.path.join(bd, f"{out_name}.elf")
    if os.path.exists(elf_path):
        print(f"  {out_name}: already built, skipping")
        return True

    # Two stock mc_re8_rn3 sub-devices, no shared weight, chain output→input.
    # mc args: (n_cores, tile_h, tile_w, ic, oc, ks, stride, ppc, input_depth)
    # Pull from MC_CONFIGS by name — build_merged resolves kind="mc".
    sub_names = [f"{mc_name}_pair_a", f"{mc_name}_pair_b"]
    # Use sub_spec_overrides to give both subs the same underlying CONFIG.
    # _resolve_sub_spec on the base name to get the args, then reuse.
    from build_merged import _resolve_sub_spec
    script, base_args = _resolve_sub_spec(mc_name, "mc")
    path = build_merged(
        out_name, sub_names,
        kind="mc",
        sub_spec_overrides={
            sub_names[0]: (script, base_args),
            sub_names[1]: (script, base_args),
        },
        # sub args from aie2_multicore.py are (I, W, O):
        #   arg0 = input, arg1 = weight, arg2 = output
        # Alias sub_1's input (arg0) to sub_0's output (arg2):
        chain_links=[(0, 2, 1, 0)],
    )
    return path is not None


def main():
    print(f"Building {len(_RN3_PAIRS)} rn3 pair ELFs...")
    t0 = time.time()
    ok = fail = 0
    for label, mc_name in _RN3_PAIRS:
        print(f"=== {label}: mc_name={mc_name} ===")
        if _build_one(label, mc_name):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} OK, {fail} FAIL in {time.time() - t0:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
