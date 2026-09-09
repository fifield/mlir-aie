#!/usr/bin/env python3
"""Trained ELAN2 Conv4: bitwise comparison against four legacy GEMM runs.

Run with python -O too: checks deliberately do not use assert statements.
Timers include host packing, uploads, submission and output materialization.
Artifacts/model initialization is outside timing; no hardware retries.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whole_gemm import WholeGemm
from runtime_metrics import RuntimeMetrics


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--frames', type=int, default=16)
    args = parser.parse_args()
    if args.frames < 8:
        parser.error('use at least eight frames for two weight variants and four input cases')
    import torch
    import test_full_model_mc as model_test
    backend = model_test.mcr
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_test.load_model()
    trained_weights = model_test.fuse_bn(model.elan2.conv4)
    # Keep both arrays alive: the legacy repack cache is identity keyed.
    # Variant swaps output-channel convolution rows, halves BN scales, and
    # negates BN biases, explicitly exercising weight rebinding across frames.
    changed = torch.from_numpy(trained_weights.copy()).view(torch.bfloat16)
    changed[:8192] = changed[:8192].reshape(64, 128).flip(0).reshape(-1)
    changed[8192:8256] *= 0.5
    changed[8256:] *= -1
    weight_variants = (trained_weights, changed.view(torch.uint16).numpy().copy())
    name = 'gemm_t104_ic128_oc64_p2'
    baseline = backend._get_gemm_handle(name)
    require(baseline is not None, f'baseline {name} artifact missing')
    whole = WholeGemm(args.build_dir, backend)
    runtime = backend.DefaultNPURuntime
    for frame in range(args.frames):
        torch.manual_seed(42 + frame)
        x = torch.randn(160, 160, 128, dtype=torch.bfloat16)
        variant = frame % 2
        weights = weight_variants[variant]
        case = (frame // 2) % 4
        if case == 0:
            x.zero_()
        elif case == 1:
            x.fill_(-0.5)
        elif case == 2:
            x.zero_()
            x[0, 0, :] = 1
            x[-1, -1, :] = -1
            # Exercise batch/slot edges in addition to image borders.
            x.reshape(-1, 128)[6655:6657] = 0.5
        outputs = {}
        for route in (('baseline', 'whole') if frame % 2 == 0 else ('whole', 'baseline')):
            with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
                if route == 'baseline':
                    # Legacy helper retries on failure; suppress that recovery
                    # path during validation so a timeout never triggers reuse.
                    with patch.object(backend, '_load_gemm_handle',
                                      side_effect=RuntimeError('baseline failed; restart process')):
                        started = time.perf_counter()
                        out = backend._run_gemm_oc_blocked(baseline, name, name, x, weights,
                                                          160, 160, 64, 104, 64, 2)
                        elapsed = (time.perf_counter() - started) * 1000
                else:
                    started = time.perf_counter()
                    out = whole.run(x, weights)
                    elapsed = (time.perf_counter() - started) * 1000
                counts = metrics.snapshot()
            require(torch.isfinite(out).all().item(), f'nonfinite {route} frame {frame}')
            expected = 4 if route == 'baseline' else 1
            require(counts['run_calls'] == expected and counts['completed_runs'] == expected,
                    f'wrong dispatch count {counts}')
            require(counts['sync_from_calls'] == expected, f'wrong readback count {counts}')
            require(counts['load_calls'] == 0, f'unexpected context reload {counts}')
            outputs[route] = out
            print(json.dumps(dict(frame=frame, seed=42 + frame, weight_variant=variant, case=case, route=route,
                                  wall_ms=elapsed, counts=counts)), flush=True)
        error = (outputs['whole'].float() - outputs['baseline'].float()).abs().max().item()
        require(torch.equal(outputs['whole'], outputs['baseline']),
                f'nonidentical output frame={frame} diff={error}')
        print(json.dumps(dict(frame=frame, max_abs_diff=error, ok=True)), flush=True)
    print(json.dumps(dict(status='PASS', frames=args.frames, comparison='bitwise exact')))


if __name__ == '__main__':
    main()
