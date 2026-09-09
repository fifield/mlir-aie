#!/usr/bin/env python3
"""Exact opaque one-core finite phase/alias gate, not SPP arithmetic.

One native upload, one submission and one readback cover A and all 25 B
stripes. Paired index halves identify every source word; changing random and
opaque patterns check rearm. No retries or host phase/stripe synchronization.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sppelan.phase_alias_host import PhaseAlias, FRAME_BYTES, frame_input, oracle, validate_output
from runtime_metrics import RuntimeMetrics


def validate_counts(counts):
    required = dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                    sync_to_calls=1, sync_from_calls=1,
                    sync_to_bytes=FRAME_BYTES, sync_from_bytes=FRAME_BYTES)
    for key, expected in required.items():
        if counts.get(key) != expected:
            raise RuntimeError(f'phase-alias wrong {key}: expected {expected}, got {counts}')


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


def checked_frame(probe, bits, expected, record, metrics_class=RuntimeMetrics):
    """Poison on semantic/count failures as well as runtime failures."""
    try:
        with metrics_class(probe.backend.DefaultNPURuntime,
                           probe.backend.DefaultNPURuntime._tensor_class) as metrics:
            started = time.perf_counter()
            observed = probe.run(bits)
            record['observed'] = observed
            wall_ms = (time.perf_counter() - started) * 1000
            counts = metrics.snapshot()
        validate_counts(counts)
        validate_output(observed, expected)
        return observed, wall_ms, counts
    except Exception:
        probe.failed = True
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--failure-dir', type=Path)
    args = parser.parse_args()
    if args.frames < 6:
        parser.error('use at least six frames for both index halves and all patterns')
    import aie.iron as iron
    from aie.utils import DefaultNPURuntime, NPUKernel
    runtime = DefaultNPURuntime
    probe = PhaseAlias(args.build_dir, SimpleNamespace(iron=iron, DefaultNPURuntime=runtime, NPUKernel=NPUKernel))
    for frame in range(args.frames):
        case, bits = frame_input(frame, args.seed)
        expected, record = oracle(bits), {}
        try:
            observed, wall_ms, counts = checked_frame(probe, bits, expected, record)
            print(json.dumps(dict(status='PASS', frame=frame, case=case, seed=args.seed + frame,
                                  wall_ms=wall_ms, counts=counts)), flush=True)
        except Exception as error:
            probe.failed = True
            artifact = save_failure(args.failure_dir, frame, case, bits, expected, record.get('observed'), error)
            print(json.dumps(dict(status='FAIL', frame=frame, case=case, error=str(error),
                                  failure_artifact=artifact)), flush=True)
            raise
    print(json.dumps(dict(status='PASS', frames=args.frames, comparison='every uint16 element exact',
                          scope='one-core finite A plus 25 B phase-alias transport; not numerical SPP')))


if __name__ == '__main__':
    main()
