#!/usr/bin/env python3
"""All16 trained shards: fresh production KB128 f0 + independent pools/gather.

Exact finite bf16 and floor metadata gate; no phase-B projection or full SPP.
Each frame rotates whole trained shards, retaining all16. Baseline retries are
disabled. Device input replication is819200B despite one204800B host sync.
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
from sppelan.phase_a_gather_host import PhaseAGather, pack_weights, frame_weights, frame_input, gather_oracle, validate_output, UPLOAD_BYTES, DOWNLOAD_BYTES
from runtime_metrics import RuntimeMetrics


def validate_counts(counts):
    required = dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                    sync_to_calls=2, sync_from_calls=2,
                    sync_to_bytes=UPLOAD_BYTES, sync_from_bytes=DOWNLOAD_BYTES)
    for key, wanted in required.items():
        if counts.get(key) != wanted:
            raise RuntimeError(f'phase-A gather wrong {key}: {counts}')


def save_failure(directory, frame, case, record, error):
    if directory is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'frame-{frame:04d}-{case}.npz'
    with path.open('xb') as stream:
        np.savez(stream, **record, error=np.array(str(error)))
    return str(path)


def checked_frame(probe, bits, packed, expected, record, metrics_class=RuntimeMetrics):
    try:
        runtime = probe.backend.DefaultNPURuntime
        with metrics_class(runtime, runtime._tensor_class) as metrics:
            started = time.perf_counter()
            observed, metadata = probe.run(bits, packed)
            wall_ms = (time.perf_counter() - started) * 1000
            record.update(observed=observed, metadata=metadata)
            counts = metrics.snapshot()
        validate_counts(counts)
        validate_output(observed, expected, metadata)
        return wall_ms, counts
    except Exception:
        probe.failed = True
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--baseline-dir', required=True, help='fresh full-model root containing gemm/')
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--failure-dir', type=Path)
    args = parser.parse_args()
    if args.frames < 6:
        parser.error('at least six frames required for all numerical input cases')
    os.environ['MDV6_BUILD_DIR'] = str(Path(args.baseline_dir).resolve())
    import torch
    import test_full_model_mc as model_test
    backend = model_test.mcr
    runtime = backend.DefaultNPURuntime
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_test.load_model()
    trained = model_test.fuse_bn(model.spp9.conv1)
    # Hold every variant alive: the baseline packed-weight cache uses identity.
    variants = [frame_weights(trained, frame) for frame in range(16)]
    packed_variants = [pack_weights(raw) for raw in variants]
    name = 'gemm_t28_ic256_oc128_kb128_p1'
    baseline = backend._get_gemm_handle(name)
    if baseline is None:
        raise RuntimeError(f'missing fresh production baseline {name}')
    probe = PhaseAGather(args.build_dir, backend)
    for frame in range(args.frames):
        raw, packed = variants[frame % 16], packed_variants[frame % 16]
        case, bits = frame_input(frame, raw, args.seed)
        record = dict(input=bits, raw_weights=raw, packed_weights=packed)
        route = 'baseline'
        try:
            x = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(20, 20, 256)
            with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
                with patch.object(backend, '_load_gemm_handle', side_effect=RuntimeError('baseline failed; restart process')):
                    ref = backend._run_gemm_kblocked(baseline, name, name, x, raw, 20, 20, 128, 28, 128, 1)
                baseline_counts = metrics.snapshot()
            if any(baseline_counts.get(key) != wanted for key, wanted in
                   dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0).items()):
                raise RuntimeError(f'baseline run/reload count mismatch: {baseline_counts}')
            f0 = ref.contiguous().view(torch.uint16).numpy().reshape(400, 128)
            record['reference_f0'] = f0
            if not torch.isfinite(ref).all().item():
                raise RuntimeError('baseline produced nonfinite output')
            expected = gather_oracle(f0)
            record['expected'] = expected
            route = 'integrated'
            wall_ms, counts = checked_frame(probe, bits, packed, expected, record)
            print(json.dumps(dict(status='PASS', frame=frame, case=case, shard_rotation=frame % 16,
                                  trained_shards=16, wall_ms=wall_ms, counts=counts,
                                  baseline_counts=baseline_counts)), flush=True)
        except Exception as error:
            probe.failed = True
            artifact = save_failure(args.failure_dir, frame, case, record, error)
            print(json.dumps(dict(status='FAIL', route=route, frame=frame, case=case,
                                  error=str(error), failure_artifact=artifact)), flush=True)
            raise
    print(json.dumps(dict(status='PASS', frames=args.frames, trained_shards_per_frame=16,
                          comparison='every bf16 bit exact and finite; all metadata floor0/reserved0',
                          scope='production f0 equivalence plus independent CPU pools and resident gather; not full SPP')))


if __name__ == '__main__':
    main()
