#!/usr/bin/env python3
"""One-shot hardware trace runner for the mc_ftconv0 merged ELF.

Builds a copy of `merged_ftconv0_x8` whose sub-device 0 has IRON trace ops
configured on one worker (default worker 0). Runs one xrt.run launch with
synthetic input, reads the trace bytes appended to sub0's output BO, and
writes raw_trace.txt + trace.npy + meta.json to `<conv>/trace_ftconv0/`.

The default events (set by Runtime.enable_trace via the IRON layer) cover:
  - INSTR_VECTOR, INSTR_EVENT_0/1, LOCK_STALL, MEMORY_STALL,
    PORT_RUNNING_0/1/2 on the core trace unit
  - DMA_S2MM/MM2S start+finish + S2MM stream-starvation on the memory unit
That's enough to answer "is the per-frame NPU time compute-bound,
DMA-bound, or dispatch-overhead-bound?" — see PHASE_E_BOTTLENECK_MODEL.md
for the offline / first-principles analog of this measurement.

Usage:
  source env.sh && source venv/bin/activate
  python3 conv/trace_ftconv0.py [--trace-size BYTES] [--worker IDX]
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_merged import build_merged

# CONFIG for mc_ftconv0: (32, 20, 20, 8, 32, 3, 2, 1)
N_CORES = 32
TILE_H, TILE_W = 20, 20
IC, OC = 8, 32
KS, STRIDE, PPC = 3, 2, 1

PATCH_H = (TILE_H - 1) * STRIDE + KS  # 41
PATCH_W = (TILE_W - 1) * STRIDE + KS  # 41
PATCH_SIZE_RAW = PATCH_H * PATCH_W * IC  # 41*41*8 = 13448
PATCH_SIZE = PATCH_SIZE_RAW + (PATCH_SIZE_RAW % 2)  # round to even
WT_SIZE = OC * IC * KS * KS + 2 * OC  # 8 args after BN
CORE_INPUT_SIZE = PPC * PATCH_SIZE
CORE_OUTPUT_SIZE = PPC * TILE_H * TILE_W * OC
HOST_INPUT_SIZE = N_CORES * CORE_INPUT_SIZE
HOST_OUTPUT_SIZE = N_CORES * CORE_OUTPUT_SIZE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-size", type=int, default=8 * 1024 * 1024,
                   help="trace buffer bytes (default 8 MiB)")
    p.add_argument("--worker", type=int, default=0,
                   help="worker index to trace (0..31)")
    args = p.parse_args()

    # Build the instrumented ELF. Use a 1-clone variant (vs x8) so aiecc
    # places only 32 cores instead of 256 — compiles in ~30s instead of >10
    # minutes. One batch is enough to characterize per-batch NPU activity
    # (the kernel itself is the same; x8 only fans out launches at host).
    sub_names = ["mc_ftconv0"]
    out_name = f"merged_ftconv0_x1_traced_w{args.worker}_{args.trace_size}"
    sub_extra_args = {
        0: ["--trace-size", str(args.trace_size),
            "--trace-worker", str(args.worker)],
    }
    print(f"=== Building {out_name} (sub 0 traced, worker={args.worker}, "
          f"trace_size={args.trace_size}) ===")
    elf_path = build_merged(
        out_name, sub_names,
        share_arg_idxs={1},        # match the standard x8 layout: wt is arg0
        sub_extra_args=sub_extra_args,
    )
    if elf_path is None:
        return 1

    # Dispatch the ELF once. Sub 0's OUT BO must be enlarged by trace_size.
    import pyxrt as _xrt
    device = _xrt.device(0)
    elf = _xrt.elf(elf_path)
    ctx = _xrt.hw_context(device, elf)
    kernel = _xrt.ext.kernel(ctx, "main")

    # BO layout per dispatcher's @main (share_arg_idxs={1}):
    #   arg0: shared wt   ← WT_SIZE bf16
    #   arg1: sub0 in     ← HOST_INPUT_SIZE bf16
    #   arg2: sub0 out    ← HOST_OUTPUT_SIZE bf16 + trace_size bytes
    wt_bytes = WT_SIZE * 2
    in_bytes = HOST_INPUT_SIZE * 2
    out_bytes = HOST_OUTPUT_SIZE * 2
    out0_bytes = out_bytes + args.trace_size

    wt_bo = _xrt.ext.bo(device, wt_bytes)
    in_bo = _xrt.ext.bo(device, in_bytes)
    out_bo = _xrt.ext.bo(device, out0_bytes)

    # Synthetic input: small bf16 values so we don't NaN; weights small too.
    rng = np.random.default_rng(seed=0)
    def _fill(bo, nelem):
        mv = bo.map()
        arr = rng.integers(0x3c00, 0x4000, size=nelem, dtype=np.uint16)
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(arr, dtype=np.uint8), casting="no")
        bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _fill(wt_bo, WT_SIZE)
    _fill(in_bo, HOST_INPUT_SIZE)
    # Zero the out BO so any non-zero trace bytes after the run are real.
    mv = out_bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=out0_bytes),
              np.zeros(out0_bytes, dtype=np.uint8), casting="no")
    out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    print(f"=== Dispatching one xrt.run (out BO = {out0_bytes} bytes incl. "
          f"trace) ===")
    run = _xrt.run(kernel)
    run.set_arg(0, wt_bo)
    run.set_arg(1, in_bo)
    run.set_arg(2, out_bo)
    import time as _t
    t0 = _t.perf_counter()
    run.start()
    run.wait2()
    dt = _t.perf_counter() - t0
    print(f"    xrt.run wall: {dt*1000:.2f} ms")

    # Read the OUT BO back and slice off the trace tail.
    out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out0_all = np.frombuffer(out_bo.map(), dtype=np.uint8,
                              count=out0_bytes).copy()
    trace_bytes = out0_all[out_bytes:].tobytes()
    n_words = len(trace_bytes) // 4
    words = np.frombuffer(trace_bytes[:n_words * 4], dtype=np.uint32).copy()
    nonzero = int(np.count_nonzero(words))
    print(f"    trace bytes: {len(trace_bytes)} ({n_words} uint32 words, "
          f"{nonzero} non-zero)")

    # Dump to disk.
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "trace_ftconv0")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "trace.npy"), words)
    with open(os.path.join(out_dir, "raw_trace.txt"), "w") as f:
        for w in words:
            if int(w) != 0:
                f.write(f"{int(w):08x}\n")
    meta = {
        "target": out_name,
        "worker": args.worker,
        "trace_size_bytes": args.trace_size,
        "elf": elf_path,
        "xrt_run_wall_ms": dt * 1000,
        "trace_words_total": n_words,
        "trace_words_nonzero": nonzero,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    -> {out_dir}/trace.npy + raw_trace.txt + meta.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
