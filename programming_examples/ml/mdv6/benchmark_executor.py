#!/usr/bin/env python3
"""Changing-input inference-only benchmark of the persistent hybrid baseline.

This is not directly comparable with the historical --profile wall metric.
Construction, CPU reference, comparison, metrics snapshots and reporting are
outside the frame timer. Host CPU islands and graph layout work remain inside.
The first frame includes lazy runtime/packing initialization and is reported
separately. The benchmark does not advertise device-resident intermediate data.
"""
import argparse
import json
import math
import hashlib
import os
from pathlib import Path
import sys
import time

from mdv6_executor import MDV6Executor, LegacyGraphExecutor, reject_diagnostics


def run_frames(executor, make_input, compare, frames, seed, report,
               snapshot=lambda: {}, clock=time.perf_counter, metadata=None):
    """Dependency-injected loop. Emit every result, stop at first failure."""
    rows = []
    for frame in range(frames):
        row = dict(metadata or {})
        row.update({'frame': frame, 'seed': seed + frame, 'ok': False,
               'phase': 'cold-runtime' if frame == 0 else 'warm',
               'route': executor.route,
               'wall_scope': 'hybrid forward including CPU islands/layouts/detection; excludes construction/reference/comparison',
               'intermediate_residency': 'host-materialized'})
        try:
            x = make_input(seed + frame)
            ref_started = time.perf_counter()
            ref = executor.reference(x)
            row['setup_and_reference_ms'] = (time.perf_counter() - ref_started) * 1000
            row['frame_setup_ms'] = getattr(executor, 'frame_setup_ms', 0.0)
            before = snapshot()
            started = clock()
            try:
                out = executor.run_frame(x)
            finally:
                row['inference_ms'] = (clock() - started) * 1000
                after = snapshot()
                row['runtime_delta'] = {
                    key: after[key] - before[key]
                    if isinstance(after[key], (int, float)) and isinstance(before.get(key), (int, float))
                    else None for key in after
                }
            result = compare(ref, out)
            required = {'finite', 'max_class_diff', 'max_vector_diff', 'ok'}
            if not required.issubset(result):
                raise RuntimeError('incomplete full-model validation metrics')
            row.update(result)
        except Exception as error:
            row['ok'] = False
            row['error'] = f'{type(error).__name__}: {error}'
        report.write(json.dumps(row, allow_nan=False) + '\n')
        report.flush()
        print('EXECUTOR ' + json.dumps(row, allow_nan=False), flush=True)
        rows.append(row)
        if not row['ok']:
            return 1, rows
    return 0, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--route', choices=('persistent', 'legacy'), default='persistent',
                        help='use separate processes for matched-scope route comparisons')
    parser.add_argument('--class-tolerance', type=float, default=0.5)
    parser.add_argument('--vector-tolerance', type=float, default=0.1)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.frames < 1 or any(not math.isfinite(t) or t <= 0 for t in
                             (args.class_tolerance, args.vector_tolerance)):
        parser.error('frames must be positive and tolerances finite and positive')
    try:
        reject_diagnostics()
    except ValueError as error:
        parser.error(str(error))
    # Exclusive creation avoids overwriting prior evidence; acquire before NPU.
    with args.report.open('x') as report:
        started = time.perf_counter()
        executor = (MDV6Executor if args.route == 'persistent' else LegacyGraphExecutor)()
        construction_ms = (time.perf_counter() - started) * 1000
        torch = executor.backend.torch
        from runtime_metrics import RuntimeMetrics
        runtime = executor.backend.DefaultNPURuntime
        weights_path = Path(executor.backend.__file__).parent / 'mdv6_bf16_weights.pt'
        if not weights_path.exists():
            weights_path = Path('/home/jfifield/mdv6/mdv6.pt')
        with weights_path.open('rb') as weight_file:
            weights_sha256 = hashlib.file_digest(weight_file, 'sha256').hexdigest()
        metadata = {
            'construction_ms': construction_ms,
            'class_tolerance': args.class_tolerance, 'vector_tolerance': args.vector_tolerance,
            'weights_path': str(weights_path), 'weights_sha256': weights_sha256,
            'runtime_module': getattr(sys.modules.get(type(runtime).__module__), '__file__', None),
            'build_root': os.environ.get('MDV6_BUILD_DIR'),
            'route_flags': {key: os.environ.get(key) for key in
                            ('MDV6_REGIME_ROUTE', 'USE_REGIME_XCLBINS', 'USE_REGIME_KBLOCKED', 'USE_GEMM_CONV1X1')},
        }
        with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
            def make_input(seed):
                generator = torch.Generator(device='cpu').manual_seed(seed)
                return torch.randn(1, 3, 640, 640, dtype=torch.bfloat16, generator=generator)

            def compare(ref, out):
                return executor.backend.compare_detection_outputs(
                    ref, out, args.class_tolerance, args.vector_tolerance)

            status, rows = run_frames(executor, make_input, compare, args.frames,
                                      args.seed, report, metrics.snapshot, metadata=metadata)
        warm = [r['inference_ms'] for r in rows if r['phase'] == 'warm' and r['ok']]
        print(json.dumps({'construction_ms': construction_ms, 'warm_frames': len(warm),
                          'warm_mean_ms': sum(warm) / len(warm) if warm else None,
                          'ok': status == 0}, allow_nan=False))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
