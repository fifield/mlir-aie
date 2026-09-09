#!/usr/bin/env python3
"""Changing-input, exact opaque-bit packet aggregation hardware gate.

Checks every element and source-worker order. Exactly one upload, completed
submission and readback per frame; no intermediate host transfer or retry.
Each worker XORs a distinct tag, so direct host-input forwarding cannot pass.
PASS proves this single-column routing capability, not fused SPP performance.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sppelan.packet_aggregate_host import PacketAggregate, frame_input, oracle, validate_output, ELEMENTS
from runtime_metrics import RuntimeMetrics


def validate_counts(counts):
    for key, expected in dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                              sync_to_calls=1, sync_from_calls=1,
                              sync_to_bytes=ELEMENTS * 2, sync_from_bytes=ELEMENTS * 2).items():
        if counts[key] != expected:
            raise RuntimeError(f'packet aggregation wrong {key}: expected {expected}, got {counts}')


def save_failure(directory, frame, case, bits, expected, observed, error):
    if directory is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'frame-{frame:04d}-{case}.npz'
    payload = dict(input=bits, expected=expected, error=np.array(str(error)))
    if observed is not None:
        payload['observed'] = observed
    with path.open('xb') as stream:
        np.savez(stream, **payload)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--failure-dir', type=Path)
    args = parser.parse_args()
    if args.frames < 6:
        parser.error('use at least six frames for sentinel/constants/edges and two changing random inputs')
    import aie.iron as iron
    from aie.utils import DefaultNPURuntime, NPUKernel
    runtime = DefaultNPURuntime
    probe = PacketAggregate(args.build_dir, SimpleNamespace(iron=iron, DefaultNPURuntime=runtime, NPUKernel=NPUKernel))
    for frame in range(args.frames):
        case, bits = frame_input(frame, args.seed)
        expected, observed = oracle(bits), None
        try:
            with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
                started = time.perf_counter()
                observed = probe.run(bits)
                wall_ms = (time.perf_counter() - started) * 1000
                counts = metrics.snapshot()
            validate_counts(counts)
            validate_output(observed, expected)
            print(json.dumps(dict(status='PASS', frame=frame, case=case, wall_ms=wall_ms, counts=counts)), flush=True)
        except Exception as error:
            probe.failed = True
            artifact = save_failure(args.failure_dir, frame, case, bits, expected, observed, error)
            print(json.dumps(dict(status='FAIL', frame=frame, case=case, error=str(error),
                                  failure_artifact=artifact)), flush=True)
            raise
    print(json.dumps(dict(status='PASS', frames=args.frames, comparison='every uint16 element exact',
                          scope='single-column four-worker packet aggregation; not fused SPP')))


if __name__ == '__main__':
    main()
