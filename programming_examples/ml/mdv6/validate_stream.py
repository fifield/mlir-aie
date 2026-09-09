#!/usr/bin/env python3
"""Validate changing-input MDV6 frames; stop at the first failure.

Uses the existing model-reconstruction path, not a persistent-model video API.
RSS includes retained model/weight/runtime caches; context counts are observed
host handles, not a claim about driver-resident hardware contexts.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time


def resident_bytes():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def handle_inventory(runtime, npu_runtime):
    handles = runtime.cached_kernel_inventory()
    artifacts = sorted({row['xclbin'] for row in handles})
    contexts = getattr(npu_runtime, '_context_cache', None)
    return {'cached_handles': len(handles), 'distinct_artifacts': len(artifacts),
            'runtime_cached_contexts': None if contexts is None else len(contexts),
            'artifacts': artifacts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-tolerance', type=float, default=0.5)
    parser.add_argument('--vector-tolerance', type=float, default=0.1)
    parser.add_argument('--report', type=Path, required=True,
                        help='JSONL written after each frame, including failures')
    args = parser.parse_args(argv)
    if args.frames < 1 or any(not math.isfinite(t) or t <= 0 for t in
                             (args.class_tolerance, args.vector_tolerance)):
        parser.error('frames must be positive and tolerances must be finite and positive')
    # These diagnostic modes return before the full-model comparison, sometimes
    # unconditionally succeeding. Fail before importing any hardware runtime.
    debug_modes = [name for name in ('DEBUG_GEMM_TRAINED', 'DEBUG_GEMM', 'DEBUG_KERNEL')
                   if os.environ.get(name)]
    if debug_modes:
        parser.error('full-model validation requires unsetting ' + ', '.join(debug_modes))

    import test_full_model_mc as model_test
    from aie.utils import DefaultNPURuntime

    return run_frames(args, model_test, DefaultNPURuntime)


def run_frames(args, model_test, npu_runtime):
    """Run the validation loop with explicit dependencies for CPU-only tests."""
    calls = 0
    original_run = npu_runtime.run

    def counted_run(*a, **kw):
        nonlocal calls
        calls += 1
        return original_run(*a, **kw)

    npu_runtime.run = counted_run
    try:
        with args.report.open('x') as report:
            for frame in range(args.frames):
                calls = 0
                started = time.perf_counter()
                row = {'frame': frame, 'seed': args.seed + frame, 'ok': False,
                       'class_tolerance': args.class_tolerance,
                       'vector_tolerance': args.vector_tolerance}
                try:
                    metrics = {}
                    passed = bool(model_test.main(
                        seed=row['seed'], class_tolerance=args.class_tolerance,
                        vector_tolerance=args.vector_tolerance, metrics=metrics))
                    required = {'finite', 'max_class_diff', 'max_vector_diff', 'ok'}
                    if not required.issubset(metrics):
                        raise RuntimeError('Full-model comparison metrics are incomplete: missing '
                                           + ', '.join(sorted(required - metrics.keys())))
                    row.update(metrics)
                    row['ok'] = passed and bool(metrics['ok'])
                except Exception as error:
                    row['ok'] = False
                    row['error'] = f'{type(error).__name__}: {error}'
                    raise
                finally:
                    row['wall_ms'] = (time.perf_counter() - started) * 1000
                    row['wall_scope'] = 'model setup + CPU reference + hybrid forward + comparison; excludes gc'
                    row['launches'] = calls
                    gc.collect()
                    row['rss_bytes'] = resident_bytes()
                    row.update(handle_inventory(model_test.mcr, npu_runtime))
                    report.write(json.dumps(row, allow_nan=False) + '\n')
                    report.flush()
                    print('STREAM ' + json.dumps(row), flush=True)
                if not row['ok']:
                    return 1
    finally:
        npu_runtime.run = original_run
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
