#!/usr/bin/env python3
"""Full 20x20 phase-A shard gate: trained projection equivalence + exact pools.

Default80 projection frames cover all16 trained eight-channel shards across
five input cases. f0 matches the fresh KB128 baseline exactly; f1-f3 match
independent CPU maxpools of observed f0. Separate pool-only cases validate
negative padding, corners, borders and channel isolation independently.
No retries, no tolerance relaxation; explicit checks remain under python -O.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sppelan.phase_a_host import PhaseAShard, pack_shard_weights, pool_levels, pool_cases, values
from runtime_metrics import RuntimeMetrics


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_counts(counts, pool_only=False):
    required = dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                    sync_to_calls=1 if pool_only else 2, sync_from_calls=2,
                    sync_to_bytes=6400 if pool_only else 204800 + 4224,
                    sync_from_bytes=25600 + 64)
    for key, wanted in required.items():
        require(counts[key] == wanted, f'wrong {key}: {counts}')


def checked_frame(probe, runtime, frame, name, bits, weights, f0_expected, failure_dir):
    observed = None
    expected = None
    try:
        with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
            started = time.perf_counter()
            observed = probe.run(bits, weights)
            wall_ms = (time.perf_counter() - started) * 1000
            counts = metrics.snapshot()
        validate_counts(counts, probe.pool_only)
        expected = pool_levels(observed[0])
        if f0_expected is not None:
            expected[0] = f0_expected
        require(np.array_equal(observed, expected), 'phase-A exact projection/pool check failed')
        print(json.dumps(dict(status='PASS', frame=frame, case=name, pool_only=probe.pool_only,
                              wall_ms=wall_ms, counts=counts)), flush=True)
    except Exception as error:
        probe.failed = True
        path = None
        if failure_dir is not None:
            failure_dir.mkdir(parents=True, exist_ok=True)
            path = failure_dir / f'{"pool" if probe.pool_only else "projection"}-{frame:04d}-{name}.npz'
            payload = dict(input=bits, error=np.array(str(error)))
            for key, array in (('weights', weights), ('observed', observed), ('expected', expected), ('f0_expected', f0_expected)):
                if array is not None:
                    payload[key] = array
            with path.open('xb') as stream:
                np.savez(stream, **payload)
        diagnostic = {}
        if observed is not None and expected is not None:
            coords = np.argwhere(observed != expected)
            diagnostic = dict(mismatch_count=len(coords), first_coordinates=coords[:16].tolist(),
                              max_abs_diff=float(np.max(np.abs(values(observed) - values(expected)))))
        print(json.dumps(dict(status='FAIL', frame=frame, case=name, error=str(error),
                              failure_artifact=str(path) if path else None, **diagnostic)), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--baseline-dir', help='fresh full-model build root containing gemm/; required unless --pool-only')
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--pool-only', action='store_true')
    parser.add_argument('--failure-dir', type=Path)
    args = parser.parse_args()
    if not args.pool_only and (not args.baseline_dir or args.frames < 16):
        parser.error('projection requires --baseline-dir and at least16 frames covering every trained shard')
    if args.baseline_dir:
        os.environ['MDV6_BUILD_DIR'] = str(Path(args.baseline_dir).resolve())
    import torch
    import test_full_model_mc as model_test
    backend = model_test.mcr
    runtime = backend.DefaultNPURuntime
    pool = PhaseAShard(args.build_dir, backend, pool_only=True)
    independent_pool_cases = list(pool_cases())
    for frame, (name, bits) in enumerate(independent_pool_cases):
        checked_frame(pool, runtime, frame, name, bits, None, bits, args.failure_dir)
    if args.pool_only:
        print(json.dumps(dict(status='PASS', scope='independent pooling', cases=len(independent_pool_cases))))
        return
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_test.load_model()
    trained = model_test.fuse_bn(model.spp9.conv1)  # Retain for baseline identity cache.
    packed = [pack_shard_weights(trained, shard) for shard in range(16)]
    name = 'gemm_t28_ic256_oc128_kb128_p1'
    baseline = backend._get_gemm_handle(name)
    require(baseline is not None, f'missing fresh baseline {name}')
    probe = PhaseAShard(args.build_dir, backend)
    for frame in range(args.frames):
        shard = frame % 16
        case = ('random', 'negative', 'boundaries', 'cancellation', 'zero')[(frame // 16) % 5]
        torch.manual_seed(42 + frame)
        x = torch.randn(20, 20, 256, dtype=torch.bfloat16)
        if case == 'negative':
            x.fill_(-0.5)
        elif case == 'zero':
            x.zero_()
        elif case == 'boundaries':
            x.zero_()
            for index, value in ((0, 1), (15, -1), (16, .5), (27, -.5), (28, 1), (383, -1), (384, .5), (399, -1)):
                x.reshape(400, 256)[index] = value
        elif case == 'cancellation':
            row = values(trained[:32768]).reshape(128, 256)[shard * 8]
            left, right = int(np.abs(row[:128]).argmax()), 128 + int(np.abs(row[128:]).argmax())
            require(row[right] != 0, 'cannot construct weighted K-boundary cancellation input')
            x.zero_()
            x[:, :, left] = 1
            x[:, :, right] = float(-row[left] / row[right])
        # Call the exact baseline primitive, disabling its historical recovery.
        try:
            with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
                with patch.object(backend, '_load_gemm_handle', side_effect=RuntimeError('baseline failed; restart process')):
                    ref = backend._run_gemm_kblocked(baseline, name, name, x, trained, 20, 20, 128, 28, 128, 1)
                counts = metrics.snapshot()
            require(counts['run_calls'] == counts['completed_runs'] == 1 and counts['load_calls'] == 0,
                    f'baseline run/reload count mismatch: {counts}')
            require(torch.isfinite(ref).all().item(), 'baseline produced nonfinite output')
        except Exception as error:
            probe.failed = True
            artifact = None
            if args.failure_dir is not None:
                args.failure_dir.mkdir(parents=True, exist_ok=True)
                artifact = args.failure_dir / f'baseline-{frame:04d}-shard{shard}-{case}.npz'
                with artifact.open('xb') as stream:
                    np.savez(stream, input=x.contiguous().view(torch.uint16).numpy(),
                             weights=trained, error=np.array(str(error)))
            print(json.dumps(dict(status='FAIL', route='baseline', frame=frame, case=case,
                                  error=str(error), failure_artifact=str(artifact) if artifact else None)), flush=True)
            raise
        f0 = ref[:, :, shard * 8:(shard + 1) * 8].contiguous().view(torch.uint16).numpy().reshape(400, 8)
        bits = x.contiguous().view(torch.uint16).numpy().reshape(400, 256)
        checked_frame(probe, runtime, frame, f'shard{shard}_{case}', bits, packed[shard], f0, args.failure_dir)
    print(json.dumps(dict(status='PASS', projection_frames=args.frames, trained_shards=16, pool_cases=len(independent_pool_cases),
                          scope='f0 fresh-baseline equivalence; pools independent exact semantics')))


if __name__ == '__main__':
    main()
