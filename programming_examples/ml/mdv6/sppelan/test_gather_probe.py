#!/usr/bin/env python3
"""Exact, fail-stop gather-only hardware gate with changing opaque uint16 data.

First two frames carry low/high halves of linear source indices. Neither alone
is unique; together they identify all 204800 source elements. Random, zero and
all-one patterns follow and repeat. All destinations/elements are checked.
This measures a layout/routing capability, not SPPELAN math or fused latency.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sppelan.gather_host import (GatherProbe, frame_input, segment_oracle, transpose_oracle,
                                 INPUT_ELEMENTS, OUTPUT_ELEMENTS)
from runtime_metrics import RuntimeMetrics


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


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
    if args.frames < 5:
        parser.error('use at least five frames to include both index halves and all patterns')
    import aie.iron as iron
    from aie.utils import DefaultNPURuntime, NPUKernel
    backend = SimpleNamespace(iron=iron, DefaultNPURuntime=DefaultNPURuntime, NPUKernel=NPUKernel)
    probe = GatherProbe(args.build_dir, backend)
    for frame in range(args.frames):
        case, bits = frame_input(frame, args.seed)
        expected = transpose_oracle(bits)
        require(np.array_equal(expected, segment_oracle(bits)), 'independent CPU gather oracles disagree')
        observed = None
        try:
            with RuntimeMetrics(DefaultNPURuntime, DefaultNPURuntime._tensor_class) as metrics:
                started = time.perf_counter()
                observed = probe.run(bits)
                wall_ms = (time.perf_counter() - started) * 1000
                counts = metrics.snapshot()
            for key, value in dict(run_calls=1, completed_runs=1, returned_runs=1,
                                   load_calls=0, sync_to_calls=1, sync_from_calls=1,
                                   sync_to_bytes=INPUT_ELEMENTS * 2,
                                   sync_from_bytes=OUTPUT_ELEMENTS * 2).items():
                require(counts[key] == value, f'wrong {key}: expected {value}, got {counts}')
            if not np.array_equal(observed, expected):
                coords = np.argwhere(observed != expected)
                print(json.dumps(dict(frame=frame, case=case, status='FAIL',
                                      mismatched_elements=int(len(coords)),
                                      first_mismatches=[dict(coordinate=c.tolist(),
                                                             observed=int(observed[tuple(c)]),
                                                             expected=int(expected[tuple(c)])) for c in coords[:16]])),
                      flush=True)
                raise RuntimeError('gather output differs from independent exact layout oracle')
            print(json.dumps(dict(frame=frame, case=case, seed=args.seed + frame, status='PASS',
                                  wall_ms=wall_ms, counts=counts)), flush=True)
        except Exception as error:
            probe.failed = True
            artifact = save_failure(args.failure_dir, frame, case, bits, expected, observed, error)
            print(json.dumps(dict(frame=frame, case=case, status='FAIL', error=str(error),
                                  failure_artifact=artifact)), flush=True)
            raise
    print(json.dumps(dict(status='PASS', frames=args.frames, comparison='every uint16 element exact',
                          scope='gather-only layout and routing, not SPPELAN mathematics')))


if __name__ == '__main__':
    main()
