#!/usr/bin/env python3
"""Independent exact sparse oracle for the production KB8 kernel (one worker).

Exposes raw first-K partials separately from final BN+approximate-SiLU outputs.
Synthetic values have exact expected bf16 results. Every spatial row and input
channel receives positive/negative impulses. Zero second-K weights isolate the
partial accumulator import; zero first-K weights isolate second-K matrix input.
Stops at first failure, no reload/retry, including under python -O.
The historically failing dynamic row-extract compiler diagnostic is reported but
only gated with --check-dynamic-extract. Production and literal-index results
remain independently exact-gated by default.
Compiler context can change whether this diagnostic reproduces; zero reported
failures do not establish that all dynamic extraction codegen is safe.
Each 80-element logical weight chunk occupies 96 bf16 elements so its 64-lane
weight load starts on a 64-byte boundary. This padding corrects an early probe
artifact only; deployed KB16/OC128 weight chunks already meet that alignment.
The independent conversion oracle uses floor, guarded by the observed crRnd
metadata (must equal rnd_floor=0). No production rounding setting is changed.
AMD documents supported bf16 rounding modes in XAPP1406:
https://docs.amd.com/r/en-US/xapp1406-aie-ml-fp-computation/Sum-of-bfloat16-Representation
"""
import argparse
import json
from pathlib import Path
import numpy as np


def bf16_bits(values):
    """Finite float32 -> bf16 toward negative infinity; hardware mode is gated.

    Truncate positive magnitudes; negative values with discarded bits advance
    one representable magnitude. This is a fixed independent oracle, not a fit
    to observed output. Input/weight values in this probe are exactly bf16.
    """
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    increment = ((bits & np.uint32(0x80000000)) != 0) & ((bits & 0xffff) != 0)
    return ((bits >> 16) + increment.astype(np.uint32)).astype(np.uint16)


def bf16_values(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def reference(x, weights):
    partial = bf16_values(bf16_bits(x[:, :8] @ weights[:, :8].T))
    total = bf16_values(bf16_bits(partial + x[:, 8:] @ weights[:, 8:].T))
    final = total * (np.float32(0.5) + total / (np.float32(2) + np.float32(2) * np.abs(total)))
    a = x[:, :8]
    traces = [a.reshape(-1), a.T.reshape(-1)]
    traces.extend(np.repeat(a[:, col], 8) for col in range(8))
    traces.append(partial.reshape(-1))
    traces.append(np.tile(a[:, 0], 8))
    prefix = [bf16_bits(partial).reshape(-1), bf16_bits(final).reshape(-1)] + [bf16_bits(t) for t in traces]
    raw_accum = (x[:, :8] @ weights[:, :8].T).astype(np.float32)
    return np.concatenate(prefix + [bf16_bits(partial).reshape(-1), bf16_bits(partial).reshape(-1),
                                    raw_accum.reshape(-1).view(np.uint16),
                                    bf16_bits(partial).reshape(-1), bf16_bits(partial).reshape(-1),
                                    bf16_bits(partial).reshape(-1), np.zeros(32, np.uint16)])


def mismatch_mask(observed, expected):
    """Mixed ABI: bf16 except raw float32 accumulator at uint16[512:576]."""
    values = bf16_values(observed)
    mask = (values != bf16_values(expected)) | ~np.isfinite(values)
    raw = observed[512:576].view(np.float32)
    reference_raw = expected[512:576].view(np.float32)
    mask[512:576] = np.repeat((raw != reference_raw) | ~np.isfinite(raw), 2)
    mask[672:704] = observed[672:704] != expected[672:704]
    return mask


def cases():
    # Exercise the suspected signed fourth-row broadcast immediately.
    x = np.zeros((4, 16), np.float32)
    x[3, 0] = -1
    weights = np.tile(np.eye(8, dtype=np.float32), (1, 2))
    yield 'signed-fourth-row', x, weights
    x = np.zeros((4, 16), np.float32)
    x[:, 0] = [1, 2, 4, -8]
    yield 'distinct-row-magnitudes', x, weights
    for family in ('both', 'first-only', 'second-only'):
        weights = np.zeros((8, 16), np.float32)
        if family != 'second-only':
            weights[:, :8] = np.eye(8, dtype=np.float32)
        if family != 'first-only':
            weights[:, 8:] = np.eye(8, dtype=np.float32)
        for row in range(4):
            for channel in range(16):
                for sign in (1, -1):
                    x = np.zeros((4, 16), np.float32)
                    x[row, channel] = sign
                    yield f'{family}-r{row}-c{channel}-s{sign}', x, weights.copy()
    # Nonzero first partial + zero second input isolates partial import itself.
    # Both partial roundoff and cancellation must preserve the zero other rows.
    for row in range(4):
        for sign in (1, -1):
            for numerator in (1, 2, 3):
                weights = np.zeros((8, 16), np.float32)
                weights[0, 0] = 1
                # Below, at and above the bf16 midpoint; all weights exactly
                # representable. These distinguish floor from nearest modes.
                weights[0, 1] = numerator / 512
                weights[0, 8] = 1
                x = np.zeros((4, 16), np.float32)
                x[row, 0:2] = sign
                x[row, 8] = -sign
                yield f'rounded-partial-cancellation-r{row}-s{sign}-n{numerator}', x, weights


def pack_weights(weights):
    chunks = np.zeros((2, 96), np.uint16)
    for block in range(2):
        chunks[block, :80] = np.concatenate((bf16_bits(weights[:, block * 8:(block + 1) * 8].T).reshape(-1),
                                           bf16_bits(np.ones(8)), bf16_bits(np.zeros(8))))
    return chunks.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True, type=Path)
    parser.add_argument('--failure-dir', type=Path)
    parser.add_argument('--cpu-only', action='store_true', help='validate case construction without loading hardware')
    parser.add_argument('--check-dynamic-extract', action='store_true',
                        help='also gate dynamic row extraction (may fail with affected compiler codegen)')
    args = parser.parse_args()
    if args.cpu_only:
        # A raw float low half of 0x8000 must not compare as bf16 negative zero.
        floor_inputs = [1 + n / 512 for n in (1, 2, 3)] + [-1 - n / 512 for n in (1, 2, 3)]
        if not np.array_equal(bf16_values(bf16_bits(floor_inputs)), [1, 1, 1, -1.0078125, -1.0078125, -1.0078125]):
            raise RuntimeError('floor conversion oracle failed independent midpoint table')
        scalar = np.float32(-1 / 128)
        scalar *= np.float32(.5) + scalar / (np.float32(2) + np.float32(2) * abs(scalar))
        if bf16_values(bf16_bits([scalar]))[0] != -0.0038909912109375:
            raise RuntimeError('floor scalar SiLU conversion oracle failed')
        raw_a = np.zeros(704, np.uint16)
        raw_b = raw_a.copy()
        raw_b[512] = 0x8000
        if not mismatch_mask(raw_a, raw_b)[512:514].all():
            raise RuntimeError('raw float comparison masked significant low bits')
        count = 0
        for name, x, weights in cases():
            packed = pack_weights(weights).reshape(2, 96)
            if packed.strides[0] % 64 != 0 or np.any(packed[:, 80:] != 0):
                raise RuntimeError('weight chunk alignment or zero padding violated')
            for block in range(2):
                recovered = bf16_values(packed[block, :64]).reshape(8, 8).T
                if not np.array_equal(recovered, weights[:, block * 8:(block + 1) * 8]):
                    raise RuntimeError(f'weight packing mismatch {name} block {block}')
            result = bf16_values(reference(x, weights)[:64]).reshape(2, 4, 8)
            zero_rows = np.all(x == 0, axis=1)
            if np.any(result[:, zero_rows] != 0):
                raise RuntimeError(f'bad CPU oracle {name}')
            if 'rounded-partial' in name:
                expected_final = -0.0038909912109375 if '-s-1-' in name else 0.0
                if result[1].sum() != expected_final:
                    raise RuntimeError(f'bad rounded-partial oracle {name}')
            count += 1
        print(json.dumps(dict(status='PASS', scope='CPU oracle construction only', cases=count)))
        return
    import aie.iron as iron
    from aie.utils import DefaultNPURuntime, NPUKernel
    xclbin = args.build_dir / 'kblocked_sparse_probe.xclbin'
    insts = args.build_dir / 'kblocked_sparse_probe.bin'
    if not xclbin.is_file() or not insts.is_file():
        raise FileNotFoundError(f'missing sparse probe artifacts under {args.build_dir}')
    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts)))
    inp, wt, out = [iron.zeros(n, dtype=np.uint16) for n in (64, 192, 704)]
    count = 0
    dynamic_failed_cases = 0
    for name, x, weights in cases():
        for buf, bits in ((inp, bf16_bits(x).reshape(-1)), (wt, pack_weights(weights))):
            buf.data.reshape(-1)[:] = bits
            buf._sync_to_device()
        result = DefaultNPURuntime.run(handle, [inp, wt, out])
        if not result.is_success():
            raise RuntimeError(f'unsuccessful runtime result {name}; stop and recover')
        observed = out.numpy().copy()
        expected = reference(x, weights)
        rounding_mode = int(observed[672])
        print(json.dumps(dict(case=name, observed_rounding_mode=rounding_mode,
                              required_rounding_mode=0, rounding_name='floor')), flush=True)
        values = bf16_values(observed)
        # Numerical equality accepts signed zero, but permits no nonzero error.
        mismatches = mismatch_mask(observed, expected)
        dynamic_mismatches = np.flatnonzero(mismatches[480:512]).tolist()
        dynamic_failed_cases += bool(dynamic_mismatches)
        print(json.dumps(dict(case=name, dynamic_extract_diagnostic=dict(
            ok=not dynamic_mismatches, mismatch_indices=dynamic_mismatches,
            gated=args.check_dynamic_extract))), flush=True)
        gated_mismatches = mismatches.copy()
        if not args.check_dynamic_extract:
            gated_mismatches[480:512] = False
        good = not gated_mismatches.any()
        if not good:
            if args.failure_dir is not None:
                args.failure_dir.mkdir(parents=True, exist_ok=True)
                path = args.failure_dir / f'{name}.npz'
                with path.open('xb') as stream:
                    np.savez(stream, input=bf16_bits(x), weights=bf16_bits(weights),
                             observed=observed, expected=expected)
            mismatch = np.flatnonzero(mismatches)
            regions = [('partial', 0, 32), ('final', 32, 64), ('assembled_a', 64, 96),
                       ('transposed_a', 96, 128)]
            regions += [(f'broadcast_column_{c}', 128 + c * 32, 160 + c * 32) for c in range(8)]
            regions += [('imported_partial', 384, 416)]
            regions += [('raw_broadcast_column_0', 416, 448)]
            regions += [('direct_mmul_whole_store', 448, 480), ('direct_mmul_row_stores', 480, 512),
                        ('direct_mmul_float_bits', 512, 576), ('direct_mmul_mul', 576, 608),
                        ('native_mul_helper', 608, 640)]
            regions += [('literal_mmul_row_stores', 640, 672)]
            regions += [('rounding_mode_metadata', 672, 704)]
            failed_stages = {stage: (mismatch[(mismatch >= a) & (mismatch < b)] - a).tolist()
                             for stage, a, b in regions if np.any((mismatch >= a) & (mismatch < b))}
            print(json.dumps(dict(status='FAIL', case=name, indices=mismatch.tolist(),
                                  failed_stages=failed_stages,
                                  direct_accum_float=observed[512:576].view(np.float32).tolist(),
                                  observed=values.tolist(), expected=bf16_values(expected).tolist())), flush=True)
            raise RuntimeError(f'sparse exact oracle failed {name}; no retry')
        print(json.dumps(dict(status='PASS', case=name)), flush=True)
        count += 1
    print(json.dumps(dict(status='PASS', cases=count, scope='independent exact sparse oracle',
                          required_rounding_mode='floor (0)',
                          dynamic_extract_gated=args.check_dynamic_extract,
                          dynamic_extract_failed_cases=dynamic_failed_cases)))


if __name__ == '__main__':
    main()
