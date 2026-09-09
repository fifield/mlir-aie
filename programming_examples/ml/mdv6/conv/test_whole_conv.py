#!/usr/bin/env python3
"""Trained-weight full-operator comparison against the unchanged 4-run kernel.

Requires the prebuilt mc_re8_rn3 artifact under MDV6_BUILD_DIR and a separately
built whole_conv artifact. Calls the inner baseline directly, bypassing opt-in
routing. Cold setup is outside timing; timers include packing/upload/readback.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whole_conv import WholeConv
from runtime_metrics import RuntimeMetrics


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--frames', type=int, default=48)
    args = parser.parse_args()
    if args.frames < 48:
        parser.error('use at least 48 frames to cover all 12 weights and four input cases')
    import torch
    import test_full_model_mc as model_test
    backend = model_test.mcr
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_test.load_model()
    weights = [model_test.fuse_bn(block.conv2)
               for layer in (model.rep_elan8, model.rep_elan21)
               for branch in (layer.conv2[0], layer.conv3[0])
               for block in branch.bottleneck]
    require(len(weights) == 12, 'trained layer inventory changed')
    baseline_handle = backend._get_mc_handle('mc_re8_rn3')
    require(baseline_handle is not None, 'baseline mc_re8_rn3 artifact missing')
    whole = WholeConv(args.build_dir, backend)
    runtime = backend.DefaultNPURuntime
    for frame in range(args.frames):
        torch.manual_seed(42 + frame)
        x = torch.randn(20, 20, 64, dtype=torch.bfloat16)
        case = (frame // 12) % 4
        if case == 0:
            x.zero_()
        elif case == 1:
            x.fill_(-0.5)
        elif case == 2:
            x.zero_()
            x[0, 0, :] = 1
            x[-1, -1, :] = -1
        wt = weights[frame % 12]
        results = {}
        # Alternate order; each route must remain correct across contexts.
        for route in (('baseline', 'whole') if frame % 2 == 0 else ('whole', 'baseline')):
            with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
                started = time.perf_counter()
                if route == 'baseline':
                    out = backend._run_tiled_mc_inner(baseline_handle, x, wt,
                                                      20, 20, 64, 8, 8, 16, 1, 3, 1)
                else:
                    out = whole.run(x, wt)
                elapsed = (time.perf_counter() - started) * 1000
                counts = metrics.snapshot()
            require(torch.isfinite(out).all().item(), f'nonfinite {route} frame {frame}')
            expected = 4 if route == 'baseline' else 1
            require(counts['run_calls'] == expected and counts['completed_runs'] == expected,
                    f'wrong dispatch count {counts}')
            require(counts['sync_from_calls'] == expected, f'wrong readback count {counts}')
            results[route] = out
            print(json.dumps(dict(frame=frame, weight=frame % 12, case=case, route=route,
                                  wall_ms=elapsed, counts=counts)), flush=True)
        error = (results['whole'].float() - results['baseline'].float()).abs().max().item()
        require(torch.equal(results['whole'], results['baseline']), f'nonidentical output frame={frame} diff={error}')
        print(json.dumps(dict(frame=frame, max_abs_diff=error, ok=True)), flush=True)
    print(json.dumps(dict(status='PASS', frames=args.frames, comparison='bitwise exact')))


if __name__ == '__main__':
    main()
