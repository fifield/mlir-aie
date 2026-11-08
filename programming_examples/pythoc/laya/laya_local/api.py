"""Application-facing results, without inference tensors or diagnostic traces."""
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time


# laya_npu is built from source in laya_npu/ (see laya_npu/README.md); fast arithmetic is explicit.
BACKENDS = ('exact', 'fast')
DEFAULT_BACKEND = 'exact'
# The staged laya_npu runtime: `python laya_npu/stage_runtime.py --build BUILD --output STAGE`.
LAYA_NPU_STAGE = Path(os.environ.get('LAYA_NPU_STAGE', Path(__file__).resolve().parents[1] / 'laya_npu/staged/current'))


def laya_npu_runtime():
    """Import the staged laya_npu_runtime package."""
    if not (LAYA_NPU_STAGE / 'laya_npu_runtime').is_dir():
        raise RuntimeError(f'no staged laya_npu runtime at {LAYA_NPU_STAGE}; stage one or set LAYA_NPU_STAGE')
    if str(LAYA_NPU_STAGE) not in sys.path:
        sys.path.insert(0, str(LAYA_NPU_STAGE))
    return importlib.import_module('laya_npu_runtime')


def _make_backend(backend=None, **options):
    backend = DEFAULT_BACKEND if backend is None else backend
    if backend in ('exact', 'fast'):
        laya_npu_runtime()
        from laya_npu_runtime.infer import Inference
        return Inference(use_lock=False, arithmetic=backend, **options)
    raise ValueError('backend must be one of ' + ', '.join(BACKENDS))


def _validate(state, questions):
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be text, a JSON object, or a list")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(q, dict):
            raise ValueError("questions must map nonempty string IDs to objects")
        kind = q.get("type")
        if kind not in ("choice", "score", "noul") or "instructions" not in q:
            raise ValueError(f"{qid}: requires type choice/score/noul and instructions")
        criteria = q.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, (dict, list)) or not criteria:
                raise ValueError(f"{qid}: choice criteria must be a nonempty object or list")
            labels = list(criteria)
            if any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
                raise ValueError(f"{qid}: choice labels must be unique nonempty strings")
        if kind == "score" and (not isinstance(criteria, list) or not criteria):
            raise ValueError(f"{qid}: score criteria must be a nonempty list")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"{qid}: noul criteria must be an object")
    try:
        json.dumps([state, questions], allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("state and questions must contain finite JSON values") from exc


class Client:
    """Own the selected backend; calls are serialized. Fast arithmetic is explicit. Use as a context manager.

    predict_batch accepts a list of states with a shared question schema and
    returns results in input order. Each state is evaluated separately; within
    a state, questions are grouped into supported 8/4/2/1-row calls.
    """

    def __init__(self, backend=None, *, cache_entries=2, cache_bytes=2*1024**3,
                 memo_entries=0, memo_bytes=8*1024**2, phase_major=False):
        backend = DEFAULT_BACKEND if backend is None else backend
        if backend not in BACKENDS:
            raise ValueError('backend must be one of ' + ', '.join(BACKENDS))
        self.backend_name = backend
        self.arithmetic = 'fast' if backend == 'fast' else 'exact'

        self._mutex = threading.RLock()
        self._backend = None
        self._closed = False
        self._device_lock = open("/tmp/npu-dev.lock", "a")
        try:
            fcntl.flock(self._device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._device_lock.close()
            raise RuntimeError("NPU is in use (/tmp/npu-dev.lock); stop the other owner first") from None
        except BaseException:
            self._device_lock.close()
            raise
        try:
            # Always dispatch the selected backend, independently of the release default.
            self._backend = _make_backend(backend, cache_entries=cache_entries, cache_bytes=cache_bytes,
                                          memo_entries=memo_entries, memo_bytes=memo_bytes, phase_major=phase_major)
        except BaseException:
            self._device_lock.close()
            raise

    def availability(self):
        return self._backend.availability()

    def predict(self, state, questions):
        """Return model, answers, usage, routing, and elapsed seconds."""
        _validate(state, questions)
        with self._mutex:
            if self._closed:
                raise RuntimeError("Client is closed")
            started = time.perf_counter()
            result = self._backend.answer(state, questions)['result']
            diagnostics = getattr(self._backend, 'last_response', {})
            return self._response(result, time.perf_counter()-started, diagnostics)

    def _response(self, result, elapsed, diagnostics=None):
        routing = {'model':'english', 'backend':self.backend_name, 'device':'npu',
                   'arithmetic':self.arithmetic, 'approximate':self.arithmetic == 'fast'}
        routing.update(diagnostics or {})
        return {'model':'laya-rl-agent', 'answers':result['answers'],
                'usage':{'input_tokens':result['usage']['input_tokens'], 'output_tokens':0},
                'routing':routing, 'elapsed_seconds':elapsed}

    def predict_batch(self, states, questions):
        """Preserve order; optional laya_npu pairing uses only supplied independent states."""
        if not isinstance(states, list) or not states:
            raise ValueError("states must be a nonempty list")
        for state in states:
            _validate(state, questions)
        with self._mutex:
            if self._closed:raise RuntimeError('Client is closed')
            started=time.perf_counter();results=self._backend.answer_batch(states,questions)
            elapsed=time.perf_counter()-started;events=getattr(self._backend,'last_schedule',{}).get('events',[])
            output=[]
            for i,result in enumerate(results):
                event=events[i//2] if i//2<len(events) else {}
                diag={'scheduling':event.get('request_schedules',[event.get('reason','serial')]*2)[i%2], 'paired':event.get('paired',False),
                      'elapsed_scope':'complete_batch', 'memo_hit':event.get('memo_hits',[False,False])[i%2]}
                executions=event.get('request_executions')
                if executions is not None:diag['execution']=executions[i%2]
                output.append(self._response(result['result'],elapsed,diag))
            return output

    def close(self):
        with self._mutex:
            if self._closed:
                return
            # A failed close cannot authorize dropping the device lock.
            self._backend.close()
            self._closed = True
            self._device_lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
