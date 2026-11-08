"""Repeatable row-batch benchmark for the selected pinned backend. Run with python -m laya_local.benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

LENGTHS = (64, 128, 256, 512)
BATCHES = (1, 2, 4, 8)
from .api import BACKENDS, DEFAULT_BACKEND, laya_npu_runtime
FIXTURES = Path(__file__).resolve().parents[1] / 'laya_npu/inputs/qualification/fixtures.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_case(agent, length, batch, backend=None):
    backend=DEFAULT_BACKEND if backend is None else backend
    laya_npu_runtime()
    from laya_npu_runtime.host import prepare_rows
    # The nearly full laya_npu qualification rows; the same rows serve both arithmetics.
    cases = json.loads(FIXTURES.read_text())['cases']
    rows = next(c['rows'] for c in cases if c['target'] == f'exact/b{batch}_l{length}')
    prepared = prepare_rows(agent, rows, length)
    lengths = prepared['valid_lengths']
    if len(rows) != batch or not all(length - 8 <= n <= length for n in lengths):
        raise ValueError(f"B{batch}/L{length}: fixture is not nearly full: {lengths}")
    return rows, lengths


def measure(model, rows, length, batch, warmup, repeats):
    """Time the complete rows() call, including tokenization and CPU heads."""
    warm, samples, launches = [], [], []
    expected = None
    output_hash = None
    valid_lengths = None
    for i in range(warmup + repeats):
        start = time.perf_counter()
        result = model.rows(rows, bucket=length)
        seconds = time.perf_counter() - start
        # Hash only after stopping the timer; never include diagnostic timings.
        current = digest(result['result'])
        lengths = result['prepared']['valid_lengths']
        if result['prepared']['bucket'] != length or len(lengths) != batch:
            raise RuntimeError('Backend did not execute the requested shape')
        if expected is not None and (current != expected or lengths != valid_lengths):
            raise RuntimeError(f'Nonrepeatable formatted output or lengths for B{batch}/L{length}')
        expected, valid_lengths = current, lengths
        # Check valid-position tensors, matching the accepted padding policy.
        output = result['output']
        h = hashlib.sha256()
        for j, n in enumerate(lengths):
            h.update(output[j, :n].tobytes())
        current_output = h.hexdigest()
        if output_hash is not None and current_output != output_hash:
            raise RuntimeError(f'Nonrepeatable valid-token output for B{batch}/L{length}')
        output_hash = current_output
        (warm if i < warmup else samples).append(seconds)
        if getattr(model, 'incidents', None):
            raise RuntimeError('Runtime incident/fallback occurred; refusing mixed-path timing')
        launches.append(dict(warmup=i < warmup, graph=result.get('local_graph')))
        del result
    median = statistics.median(samples)
    return dict(batch=batch, length=length, valid_lengths=valid_lengths,
                rows=rows, input_sha256=digest(rows), result_sha256=expected,
                valid_output_sha256=output_hash, warmup_seconds=warm, samples_seconds=samples,
                median_seconds=median, min_seconds=min(samples), max_seconds=max(samples),
                p95_seconds=sorted(samples)[math.ceil(.95 * len(samples)) - 1],
                rows_per_second=batch / median, valid_tokens_per_second=sum(valid_lengths) / median,
                outputs_repeatable=True, launch_samples=launches)


def selection(model,lengths,batches,skip_unavailable=False):
    plans=getattr(model,'plans',None)
    unavailable=[f'b{b}_l{l}' for l in lengths for b in batches if plans is not None and f'b{b}_l{l}' not in plans]
    if unavailable and not skip_unavailable:
        raise ValueError('Unavailable benchmark shapes: '+', '.join(unavailable)+'. Use --skip-unavailable explicitly or select another backend.')
    if len(unavailable)==len(lengths)*len(batches):raise ValueError('No requested benchmark shapes are available')
    return unavailable


def run_matrix(model, lengths=LENGTHS, batches=BATCHES, warmup=1, repeats=5,
               case_factory=make_case, on_case=None, skip_unavailable=False):
    if warmup < 1 or repeats < 1:
        raise ValueError('warmup and repeats must be positive')
    if not lengths or not batches or any(x not in LENGTHS for x in lengths) or any(x not in BATCHES for x in batches):
        raise ValueError('Unsupported benchmark shape')
    unavailable=selection(model,lengths,batches,skip_unavailable)
    cases = []
    for length in lengths:
        for batch in batches:
            if f'b{batch}_l{length}' in unavailable:continue
            rows, lens = case_factory(model.agent, length, batch)
            case = measure(model, rows, length, batch, warmup, repeats)
            if case['valid_lengths'] != lens:
                raise RuntimeError('Measured token lengths differ from prepared fixture')
            cases.append(case)
            if on_case:
                on_case(case)
    return cases


def provenance(backend=None):
    backend=DEFAULT_BACKEND if backend is None else backend
    laya_npu_runtime()
    import laya_npu_runtime.common as common
    from laya_npu_runtime.pins import BUNDLE_SHA256
    sources = 'bundle.json'
    ROOT, DATA, file_digest = common.ROOT, common.DATA, common.digest
    return dict(backend=backend, graph_bundle_sha256=BUNDLE_SHA256,
                sources_sha256=file_digest(DATA / sources),
                host_sources_sha256=file_digest(DATA / 'host_sources.json'),
                benchmark_sha256=file_digest(Path(__file__)),
                runtime_sources={name: file_digest(ROOT / name)
                                 for name in ('infer.py', 'runtime.py', 'host.py', 'common.py', 'pins.py')},
                python=platform.python_version(), platform=platform.platform())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', choices=LENGTHS, default=LENGTHS)
    parser.add_argument('--batches', type=int, nargs='+', choices=BATCHES, default=BATCHES)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('laya-benchmark-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'))
    parser.add_argument('--backend',choices=BACKENDS,default=DEFAULT_BACKEND)
    parser.add_argument('--skip-unavailable',action='store_true',help='Explicitly omit unqualified shapes and record them in the report')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error('--warmup and --repeats must be positive')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    from laya_local import Client
    report = dict(schema=1, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  warmup=args.warmup, repeats=args.repeats, lengths=args.lengths, batches=args.batches,
                  timing='wall time of native Inference.rows; tokenization, NPU, CPU heads included; HTTP excluded',
                  order='length-major, batch-minor; one persistent owner; warmup before each shape',
                  cases=[])
    # Reserve the report name before opening the device; never overwrite an earlier run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)

    def save():
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(args.output)

    def completed(case):
        report['cases'].append(case)
        save()
        print(f"B{case['batch']} L{case['length']:3d}  median {case['median_seconds'] * 1000:9.2f} ms"
              f"  p95 {case['p95_seconds'] * 1000:9.2f} ms  {case['rows_per_second']:7.2f} rows/s", flush=True)

    try:
        report['provenance'] = provenance(args.backend)
        start = time.perf_counter()
        with Client(backend=args.backend) as client:
            report['owner_setup_seconds'] = time.perf_counter() - start
            report['availability']=client.availability()
            report['unavailable_shapes']=selection(client._backend,args.lengths,args.batches,args.skip_unavailable)
            report['complete_requested_matrix']=not report['unavailable_shapes']
            report['skip_unavailable']=args.skip_unavailable
            save()
            # Benchmark-only native access: public predict_batch runs states serially.
            run_matrix(client._backend, args.lengths, args.batches, args.warmup, args.repeats, case_factory=lambda agent,length,batch:make_case(agent,length,batch,args.backend), on_case=completed, skip_unavailable=args.skip_unavailable)
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
