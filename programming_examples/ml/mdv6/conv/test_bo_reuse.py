#!/usr/bin/env python3
"""Compare host, external-BO and on-chip chains with distinct stage weights.

Build: mkdir /tmp/mdv6-bo-proof; cd /tmp/mdv6-bo-proof
       make -f <mdv6>/conv/Makefile.bo_reuse
Run: python <mdv6>/conv/test_bo_reuse.py --build-dir /tmp/mdv6-bo-proof

This is an isolated capability test, not a full-model speedup measurement.
Three independent tiles are packed [tile,H,W,C], bf16-as-uint16, contiguous.
Both single-stage submissions use the same context. External reuse does NOT
prove cross-context sharing or on-chip residency. The connected artifact does
not have an intermediate external DMA. All stages run synchronously.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from device_buffers import DeviceBuffer, TensorContract
from runtime_metrics import RuntimeMetrics


def require(condition, detail):
    """Keep capability gates active under python -O as well."""
    if not condition:
        raise RuntimeError(f"BO reuse proof failed: {detail}")


def contract(name, producer, consumers):
    return TensorContract(name, (3, 8, 8, 16), (3, 8, 8, 16),
                          (1024, 128, 16, 1), "tile-HWC", "bo_reuse_proof",
                          producer, consumers, 1, (0, 0, 0, 0), (3, 8, 8, 16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("at least two changing inputs are required")
    import numpy as np
    import torch
    import aie.iron as iron
    from aie.utils import DefaultNPURuntime as runtime, NPUKernel

    torch.manual_seed(args.seed)
    stage_weights = []
    params = []
    for _ in range(2):
        w = (torch.randn(16, 16) * 0.125).to(torch.bfloat16)
        scale = (1 + torch.randn(16) * 0.1).to(torch.bfloat16)
        bias = (torch.randn(16) * 0.05).to(torch.bfloat16)
        params.append((w, scale, bias))
        packed = torch.cat((w.flatten(), scale, bias)).view(torch.uint16).numpy()
        stage_weights.append(iron.tensor(packed, dtype=np.uint16))
    require(not torch.equal(params[0][0], params[1][0]), "stage weights must differ")
    handles = {name: runtime.load(NPUKernel(str(args.build_dir / f"{name}.xclbin"),
                                           str(args.build_dir / f"{name}.bin")))
               for name in ("single", "chain")}
    inp = iron.zeros(3072, dtype=np.uint16)
    inter = DeviceBuffer(contract("inter", "stage1", ("stage2",)),
                         iron.zeros(3072, dtype=np.uint16))
    host_bridge = iron.zeros(3072, dtype=np.uint16)
    out = iron.zeros(3072, dtype=np.uint16)
    previous = None
    print(json.dumps({"runtime_source": inspect.getfile(type(runtime)),
                      "build_dir": str(args.build_dir.resolve()), "frames": args.frames,
                      "reference_max_abs_tolerance": 0.05,
                      "route_comparison": "bitwise exact", "intermediate_bytes": 6144,
                      "timing_scope": "submit/wait plus internal host bridge; excludes ingress/egress"}))

    def upload(tensor, values):
        tensor.torch_view().copy_(values.reshape(-1).view(torch.uint16))
        tensor.to("npu")

    def reference(x):
        for w, scale, bias in params:
            x = x.float() @ w.float().t()
            x = x * scale.float() + bias.float()
            x = (x * (0.5 + 0.5 * x / (1 + x.abs()))).to(torch.bfloat16)
        return x

    for frame in range(args.frames):
        torch.manual_seed(args.seed + 1000 + frame)
        x = torch.randn(3, 8, 8, 16).to(torch.bfloat16)
        ref = reference(x)
        results = {}
        for route in ("host", "external_bo", "on_chip"):
            upload(inp, x)
            inter.invalidate()
            with RuntimeMetrics(runtime, type(inp)) as metrics:
                start = time.perf_counter()
                if route == "on_chip":
                    runtime.run(handles["chain"], [inp, *stage_weights, out])
                else:
                    runtime.run(handles["single"], [inp, stage_weights[0], inter.destination()])
                    inter.mark_written()
                    if route == "host":
                        values = torch.from_numpy(inter.download()).view(torch.bfloat16)
                        upload(host_bridge, values)
                        consumer = host_bridge
                    else:
                        consumer = inter.for_consumer(contract("stage2_input", "stage1", ("stage2",)))
                    runtime.run(handles["single"], [consumer, stage_weights[1], out])
                elapsed_ms = (time.perf_counter() - start) * 1000
                observed = metrics.snapshot()
            # Explicit final boundary, excluded from the internal-sync gate.
            result = out.numpy().copy()
            results[route] = result
            actual = torch.from_numpy(result).view(torch.bfloat16).reshape_as(ref)
            error = (ref.float() - actual.float()).abs().max().item()
            require(torch.isfinite(actual).all() and error < 0.05, (route, frame, error))
            expected_syncs = int(route == "host")
            require(observed["sync_from_calls"] == expected_syncs, observed)
            require(observed["sync_to_calls"] == expected_syncs, observed)
            require(observed["sync_from_bytes"] == expected_syncs * 6144, observed)
            require(observed["sync_to_bytes"] == expected_syncs * 6144, observed)
            expected_runs = 1 if route == "on_chip" else 2
            require(observed["completed_runs"] == expected_runs, observed)
            require(observed["run_calls"] == expected_runs, observed)
            print(json.dumps({"frame": frame, "seed": args.seed + 1000 + frame,
                              "route": route, "wall_ms": elapsed_ms,
                              "max_abs_diff": error, "metrics": observed}), flush=True)
        require(np.array_equal(results["host"], results["external_bo"]), "host/external mismatch")
        require(np.array_equal(results["host"], results["on_chip"]), "host/on-chip mismatch")
        require(previous is None or not np.array_equal(previous, results["host"]), "stale output")
        previous = results["host"]
    print(json.dumps({"status": "PASS", "frames": args.frames}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
