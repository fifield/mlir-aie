#!/usr/bin/env python3
"""Trained re4/re15 Conv4: legacy-equivalence of three runs versus one.

Includes zero, negative, cancellation, random and spatial-batch boundary inputs.
Explicit checks survive python -O. Initialization excluded; packing, uploads,
launch and output materialization included. Any failure stops without retries.
Default PASS means bitwise legacy-equivalence, NOT mathematical correctness.
Use --check-zero-rows to enforce the independent zero-input/zero-bias invariant.
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
from whole_kblocked import WholeKBlocked, pack_weights
from runtime_metrics import RuntimeMetrics


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def zero_row_diagnostic(input_values, output_values, raw_weights):
    """Independent pointwise invariant, using numerical input/output arrays.

    Zero convolution input and zero BN bias imply exactly zero after SiLU.
    Report observed nonzeros even when nonzero bias makes that gate inapplicable.
    Signed bf16 zero is accepted for the raw BN bias fields.
    """
    import numpy as np
    zero_rows = np.all(input_values.reshape(-1, 256) == 0, axis=1)
    output_rows = output_values.reshape(-1, 128)[zero_rows]
    nonzero = output_rows != 0
    bias_zero = bool(np.all((raw_weights[-128:] & 0x7fff) == 0))
    return dict(applicable=bias_zero, zero_input_rows=int(zero_rows.sum()),
                nonzero_output_rows=int(np.any(nonzero, axis=1).sum()),
                nonzero_output_elements=int(nonzero.sum()),
                max_abs_output=float(np.abs(output_rows).max()) if output_rows.size else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--failure-dir', type=Path,
                        help='save input, raw weights and both outputs on mismatch; never overwrite')
    parser.add_argument('--check-zero-rows', action='store_true',
                        help='fail boundary cases if zero-input rows leak with zero BN bias')
    args = parser.parse_args()
    if args.frames < 10:
        parser.error('use at least ten frames for two trained weights and five input cases')
    import numpy as np
    import torch
    import test_full_model_mc as model_test
    backend = model_test.mcr
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_test.load_model()
    # Keep both arrays alive for the baseline's identity-keyed packing cache.
    variants = [('re4', model_test.fuse_bn(model.rep_elan4.conv4)),
                ('re15', model_test.fuse_bn(model.rep_elan15.conv4))]
    for name, weights in variants:
        require(np.array_equal(pack_weights(weights),
                               backend._repack_weights_kblocked(weights, 256, 128, 16)),
                f'host packing differs from baseline for {name}')
    name = 'gemm_t68_ic256_oc128_kb16_p1'
    baseline = backend._get_gemm_handle(name)
    require(baseline is not None, f'baseline {name} artifact missing')
    whole = WholeKBlocked(args.build_dir, backend)
    runtime = backend.DefaultNPURuntime
    for frame in range(args.frames):
        torch.manual_seed(42 + frame)
        variant, weights = variants[frame % 2]
        x = torch.randn(80, 80, 256, dtype=torch.bfloat16)
        case = ('zero', 'negative', 'cancellation', 'random', 'boundaries')[(frame // 2) % 5]
        if case == 'zero':
            x.zero_()
        elif case == 'negative':
            x.fill_(-0.5)
        elif case == 'cancellation':
            # Nearly cancel output channel zero across the first two K blocks,
            # exercising preservation of the baseline's rounded partial sums.
            matrix = torch.from_numpy(weights[:32768].copy()).view(torch.bfloat16).float().reshape(128, 256)
            left = int(matrix[0, :16].abs().argmax())
            right = 16 + int(matrix[0, 16:32].abs().argmax())
            require(abs(matrix[0, right].item()) > 0, 'cannot construct cancellation input')
            x.zero_()
            x[:, :, left] = 1
            x[:, :, right] = -matrix[0, left] / matrix[0, right]
        elif case == 'boundaries':
            x.zero_()
            flat = x.reshape(-1, 256)
            for index, value in ((0, 1), (67, -1), (68, 0.5),
                                 (2175, -0.5), (2176, 1), (4351, -1),
                                 (4352, 0.5), (6391, -0.5), (6392, 1), (6399, -1)):
                flat[index] = value
        outputs = {}
        zero_diagnostics = {}
        for route in (('baseline', 'whole') if frame % 2 == 0 else ('whole', 'baseline')):
            with RuntimeMetrics(runtime, runtime._tensor_class) as metrics:
                if route == 'baseline':
                    with patch.object(backend, '_load_gemm_handle',
                                      side_effect=RuntimeError('baseline failed; restart process')):
                        started = time.perf_counter()
                        out = backend._run_gemm_kblocked(baseline, name, name, x, weights,
                                                        80, 80, 128, 68, 16, 1)
                        elapsed = (time.perf_counter() - started) * 1000
                else:
                    started = time.perf_counter()
                    out = whole.run(x, weights)
                    elapsed = (time.perf_counter() - started) * 1000
                counts = metrics.snapshot()
            require(torch.isfinite(out).all().item(), f'nonfinite {route} frame {frame}')
            expected = 3 if route == 'baseline' else 1
            require(counts['run_calls'] == expected and counts['completed_runs'] == expected,
                    f'wrong dispatch count {counts}')
            require(counts['sync_from_calls'] == expected, f'wrong readback count {counts}')
            require(counts['load_calls'] == 0, f'unexpected context reload {counts}')
            outputs[route] = out
            if case == 'boundaries':
                zero_diagnostics[route] = zero_row_diagnostic(x.float().numpy(), out.float().numpy(), weights)
                print(json.dumps(dict(frame=frame, route=route,
                                      zero_row_diagnostic=zero_diagnostics[route])), flush=True)
            print(json.dumps(dict(frame=frame, seed=42 + frame, weight_variant=variant, case=case,
                                  route=route, wall_ms=elapsed, counts=counts)), flush=True)
        error = (outputs['whole'].float() - outputs['baseline'].float()).abs().max().item()
        equal = torch.equal(outputs['whole'].view(torch.uint16), outputs['baseline'].view(torch.uint16))
        semantic_failure = args.check_zero_rows and any(
            d['applicable'] and d['nonzero_output_elements'] for d in zero_diagnostics.values())
        if not equal or semantic_failure:
            if args.failure_dir is not None:
                args.failure_dir.mkdir(parents=True, exist_ok=True)
                failure_path = args.failure_dir / f'frame-{frame:04d}-{variant}-{case}.npz'
                with failure_path.open('xb') as failure_file:
                    np.savez(failure_file,
                             input=x.contiguous().view(torch.uint16).numpy(),
                             weights=weights,
                             baseline=outputs['baseline'].contiguous().view(torch.uint16).numpy(),
                             whole=outputs['whole'].contiguous().view(torch.uint16).numpy())
                print(json.dumps(dict(frame=frame, failure_artifact=str(failure_path))), flush=True)
            mismatch = outputs['whole'].view(torch.uint16) != outputs['baseline'].view(torch.uint16)
            coords = mismatch.nonzero()[:16].tolist()
            pixels = mismatch.reshape(6400, 128).any(dim=1)
            print(json.dumps(dict(frame=frame, ok=False, max_abs_diff=error,
                                  semantic_failure=bool(semantic_failure), zero_row_diagnostics=zero_diagnostics,
                                  mismatch_elements=int(mismatch.sum()),
                                  mismatch_pixels_by_batch=[int(pixels[a:b].sum())
                                                            for a, b in ((0, 2176), (2176, 4352), (4352, 6400))],
                                  first_mismatch_coordinates=coords,
                                  first_mismatch_values=[dict(baseline=float(outputs['baseline'][tuple(c)]),
                                                              whole=float(outputs['whole'][tuple(c)])) for c in coords])),
                  flush=True)
        require(equal, f'nonidentical output frame={frame} diff={error}')
        require(not semantic_failure, f'zero-input rows leak with zero BN bias frame={frame}: {zero_diagnostics}')
        print(json.dumps(dict(frame=frame, max_abs_diff=error, ok=True)), flush=True)
    print(json.dumps(dict(status='PASS', frames=args.frames, comparison='bitwise exact',
                          scope='legacy-equivalence, not mathematical correctness',
                          zero_row_gate_enabled=args.check_zero_rows)))


if __name__ == '__main__':
    main()
