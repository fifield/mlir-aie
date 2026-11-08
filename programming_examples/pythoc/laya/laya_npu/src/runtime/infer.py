"""laya_npu inference: exact by default; fast arithmetic and service features explicit."""

import copy, threading, time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from .common import ROOT, DATA, read, require, digest
from .host import load_agent, prepare, prepare_rows, cpu_heads, format_result
from .runtime import Runtime, contained
from .cache import RuntimeCache
from .memo import BoundedMemo, key_for, complete_key
from .scheduler import pair_eligibility
from .mixed_schedule import eligibility as mixed_eligibility
from .mixed_contract import validate_record as validate_mixed_record


class Inference:
    version = "laya_npu"

    def __init__(
        self,
        cpu_threads=8,
        instrument=False,
        use_lock=True,
        arithmetic="exact",
        cache_entries=2,
        cache_bytes=2 * 1024**3,
        memo_entries=0,
        memo_bytes=8 * 1024**2,
        phase_major=False,
    ):
        require(
            arithmetic in ("exact", "fast"),
            "arithmetic must be exact or explicitly selected fast",
        )
        require(
            isinstance(memo_entries, int) and memo_entries >= 0,
            "memo entries must be nonnegative",
        )
        from .pins import BUNDLE_SHA256

        require(digest(DATA / "bundle.json") == BUNDLE_SHA256, "bundle hash mismatch")
        self.bundle = read(DATA / "bundle.json")
        require(
            self.bundle["version"] == "laya_npu"
            and self.bundle["default_variant"] == "exact",
            "invalid laya_npu default contract",
        )
        require(
            digest(DATA / "host_sources.json") == self.bundle["host_sources_sha256"],
            "host manifest changed",
        )
        for path, expected in read(DATA / "host_sources.json").items():
            require(digest(contained(path)) == expected, "host source changed: " + path)
        validate_mixed_record(
            self.bundle.get("mixed_requests", {}).get("exact", {}).get("b2_b1_l64")
        )
        self.arithmetic = arithmetic
        self.backend_name = arithmetic
        self.plans = self.bundle["variants"][arithmetic]["plans"]
        self.identity = dict(
            bundle_sha256=BUNDLE_SHA256,
            backend=self.backend_name,
            arithmetic=arithmetic,
            host_sources_sha256=self.bundle["host_sources_sha256"],
            model_sources_sha256=self.bundle["sources_sha256"],
        )
        self.cache = RuntimeCache(cache_entries, cache_bytes)
        self.memo = BoundedMemo(memo_entries, memo_bytes) if memo_entries else None
        self.phase_major = bool(phase_major)
        self.lifecycle = deque(maxlen=128)
        self.last_response = {}
        self.last_schedule = {}
        torch.set_num_threads(cpu_threads)
        self.agent = load_agent()
        self.runtime = None
        self.key = None
        self.closed = False
        self.use_lock = use_lock
        self.mutex = threading.RLock()
        self.incidents = []
        self.fallback = None

    def availability(self):
        rejected = self.bundle["variants"][self.arithmetic].get("rejected_shapes", {})
        return dict(
            supported_shapes=list(self.plans), rejected_shapes=copy.deepcopy(rejected)
        )

    def _record(self, shape):
        if shape not in self.plans:
            rejected = (
                self.bundle.get("variants", {})
                .get(self.arithmetic, {})
                .get("rejected_shapes", {})
            )
            item = rejected.get(shape, {})
            reason = (
                item.get("reason", "shape is not qualified for this backend")
                if isinstance(item, dict)
                else str(item)
            )
            raise ValueError(
                f'{self.backend_name if hasattr(self,"backend_name") else self.arithmetic}: unsupported shape {shape}: {reason}. Select exact explicitly for full shape coverage; no automatic fallback.'
            )
        return self.plans[shape]

    def _preflight(self, state, questions):
        # Prepare/discard one chunk at a time; no runtime or memo side effects.
        remaining = list(questions.items())
        while remaining:
            n = next(n for n in (8, 4, 2, 1) if n <= len(remaining))
            chunk, remaining = remaining[:n], remaining[n:]
            p = prepare(self.agent, state, dict(chunk))
            self._record(f"b{n}_l{p['bucket']}")
            del p

    def _owner(self, key, record):
        identity = f'{self.arithmetic}:{key}:{record["sha256"]}'
        if identity in self.cache.items:
            resident, nbytes = self.cache.items[identity]
            owner, event = self.cache.get(identity, nbytes, lambda: resident)
        else:
            path = contained(record["path"])
            require(digest(path) == record["sha256"], "plan hash mismatch")
            plan = read(path)
            owner, event = self.cache.get(
                identity,
                sum(plan["arenas"]),
                lambda: Runtime(plan, use_lock=self.use_lock),
            )
        self.runtime = owner
        self.key = key
        self.lifecycle.append(event)
        return owner

    @torch.no_grad()
    def _inputs(self, p):
        batch = p["batch"]
        embedding = (
            F.embedding(
                batch["input_ids"],
                self.agent.model.encoder.embeddings.tok_embeddings.weight,
            )
            .bfloat16()
            .view(torch.uint16)
            .numpy()
        )
        mask = batch["attention_mask"].bfloat16().view(torch.uint16).numpy()
        return embedding, mask, batch["qtype"].tolist()

    @torch.no_grad()
    def _finish(self, p, output, times, started, preparation, setup, owner, pair=False):
        t = time.perf_counter()
        batch = p["batch"]
        logits, actions, action_input = cpu_heads(
            self.agent.model, torch.from_numpy(output), batch
        )
        formatted = format_result(
            self.agent,
            {r["id"]: r["question"] for r in p["rows"]},
            batch,
            logits,
            actions,
        )
        heads = time.perf_counter() - t
        return dict(
            result=formatted,
            raw=dict(
                marker_logits=logits.numpy(),
                action_logits=actions.numpy(),
                action_input=action_input.numpy(),
            ),
            output=output,
            prepared=p,
            execution="laya_npu " + self.arithmetic + " aiecc full ELF",
            local_graph=dict(
                version="laya_npu",
                arithmetic=self.arithmetic,
                launches=len(times),
                submit_seconds=times,
                packed_bytes=sum(owner.plan["arenas"]),
                shared_pair=pair,
            ),
            timings=dict(
                cpu_preparation=preparation,
                bucket_setup=setup,
                cpu_heads_calibration_format=heads,
                full_api=time.perf_counter() - started,
            ),
        )

    @torch.no_grad()
    def _execute(self, p, started, preparation):
        require(not self.closed, "inference owner is closed")
        key = f"b{len(p['valid_lengths'])}_l{p['bucket']}"
        t = time.perf_counter()
        owner = self._owner(key, self._record(key))
        setup = time.perf_counter() - t
        t = time.perf_counter()
        args = self._inputs(p)
        preparation += time.perf_counter() - t
        try:
            output, times = owner.execute(*args)
        except BaseException as exc:
            self.incidents.append(str(exc))
            self.close()
            raise
        return self._finish(p, output, times, started, preparation, setup, owner)

    def rows(self, rows, bucket=None, capture=False):
        require(not capture, "intermediate capture unsupported")
        with self.mutex:
            start = time.perf_counter()
            p = prepare_rows(self.agent, rows, bucket)
            return self._execute(p, start, time.perf_counter() - start)

    def __call__(self, state, questions, bucket=None, capture=False):
        require(not capture, "intermediate capture unsupported")
        with self.mutex:
            start = time.perf_counter()
            p = prepare(self.agent, state, questions, bucket)
            return self._execute(p, start, time.perf_counter() - start)

    def _request(self, state, questions):
        require(not self.closed, "inference owner is closed")
        remaining = list(questions.items())
        chunks = []
        keys = []
        while remaining:
            n = next(n for n in (8, 4, 2, 1) if n <= len(remaining))
            chunk, remaining = remaining[:n], remaining[n:]
            p = prepare(self.agent, state, dict(chunk))
            chunks.append(p)
            shape = f"b{n}_l{p['bucket']}"
            identity = dict(self.identity, plan_sha256=self._record(shape)["sha256"])
            keys.append(key_for(p, identity, [state, chunk]))
        require(chunks, "questions cannot be empty")
        record, reason = self._mixed_record(chunks)
        schedule = (
            dict(kind="phase_major_b2_b1_l64", plan_sha256=record["sha256"])
            if record is not None
            else dict(kind="request_major")
        )
        identity = dict(self.identity, selected_schedule=schedule)
        key = complete_key(keys, identity, [state, list(questions.items())])
        hit = self.memo.get(key) if self.memo is not None else None
        return dict(
            prepared=chunks,
            key=key,
            hit=hit,
            mixed_record=record,
            schedule=schedule,
            mixed_reason=reason,
        )

    def _mixed_record(self, prepared):
        record = self.bundle.get("mixed_requests", {}).get("exact", {}).get("b2_b1_l64")
        eligible, reason = mixed_eligibility(
            prepared, self.arithmetic, record, self.cache.byte_limit
        )
        return (record if eligible else None), reason

    @torch.no_grad()
    def _mixed_complete(self, request, record):
        prepared = request["prepared"]
        require(
            [len(p["valid_lengths"]) for p in prepared] == [2, 1]
            and all(p["bucket"] == 64 for p in prepared),
            "mixed prepared request changed",
        )
        start = time.perf_counter()
        owner = self._owner("mixed_b2_b1_l64", record)
        setup = time.perf_counter() - start
        inputs = [self._inputs(p) for p in prepared]
        embedding = np.concatenate([x[0] for x in inputs])
        masks = np.concatenate([x[1] for x in inputs])
        types = [q for x in inputs for q in x[2]]
        self.last_execution["attempted_execution_paths"] += 1
        try:
            output, times = owner.execute(embedding, masks, types)
        except BaseException as exc:
            self.last_execution.update(
                failed_request=True, current_request_completion_unknown=True
            )
            self.incidents.append(str(exc))
            self.close()
            raise
        self.last_execution["submit_seconds"].extend(times)
        self.last_execution["launches"] += len(times)
        self.last_execution["completed_device_requests"] += 1
        require(output.shape == (3, 64, 1024), "mixed output shape changed")
        results = [
            self._finish(p, output[a:b], times, start, 0, setup, owner)["result"]
            for p, a, b in [(prepared[0], 0, 2), (prepared[1], 2, 3)]
        ]
        result = self._merge(results)
        if self.memo is not None:
            self.memo.put(request["key"], result)
        return result

    def _merge(self, results):
        merged = copy.deepcopy(results[0])
        merged["answers"] = {}
        merged["usage"] = {"input_tokens": 0, "output_tokens": 0}
        for result in results:
            merged["answers"].update(result["answers"])
            merged["usage"]["input_tokens"] += result["usage"]["input_tokens"]
            merged["usage"]["output_tokens"] += result["usage"].get("output_tokens", 0)
        return merged

    def _complete(self, request):
        self.last_execution = dict(
            submit_seconds=[],
            launches=0,
            attempted_execution_paths=0,
            completed_device_requests=0,
            failed_request=False,
            current_request_completion_unknown=False,
            memo_hit=request["hit"] is not None,
        )
        if request["hit"] is not None:
            return request["hit"]
        if request.get("mixed_record") is not None:
            try:
                return self._mixed_complete(request, request["mixed_record"])
            except BaseException:
                self.last_execution["failed_request"] = True
                raise
        results = []
        for prepared in request["prepared"]:
            self.last_execution["attempted_execution_paths"] += 1
            try:
                out = self._execute(prepared, time.perf_counter(), 0)
            except BaseException:
                self.last_execution.update(
                    failed_request=True, current_request_completion_unknown=True
                )
                raise
            times = out["local_graph"]["submit_seconds"]
            self.last_execution["submit_seconds"].extend(times)
            self.last_execution["launches"] += len(times)
            self.last_execution["completed_device_requests"] += 1
            results.append(out["result"])
        result = self._merge(results)
        if self.memo is not None:
            self.memo.put(request["key"], result)
        return result

    def answer(self, state, questions):
        """Complete ordered response, optionally memoized; no diagnostic timings cached."""
        with self.mutex:
            self.last_response = {}
            self.last_schedule = {}
            self.last_execution = dict(
                submit_seconds=[],
                launches=0,
                attempted_execution_paths=0,
                completed_device_requests=0,
                failed_request=False,
                current_request_completion_unknown=False,
                memo_hit=False,
            )
            try:
                request = self._request(state, questions)
                out = self._complete(request)
                self.last_response = dict(
                    memo_hit=request["hit"] is not None,
                    scheduling=(
                        "memo_hit"
                        if request["hit"] is not None
                        else (
                            "phase_major_b2_b1_l64"
                            if request.get("mixed_record") is not None
                            else "single_request"
                        )
                    ),
                    execution=copy.deepcopy(self.last_execution),
                )
                return dict(result=out)
            except BaseException:
                self.last_execution["failed_request"] = True
                self.last_response = dict(
                    execution=copy.deepcopy(self.last_execution), failed=True
                )
                raise

    @torch.no_grad()
    def _pair(self, requests, record):
        self._record("b1_l64")
        prepared = [r["prepared"][0] for r in requests]
        start = time.perf_counter()
        owner = self._owner("pair_b1_l64", record)
        setup = time.perf_counter() - start
        inputs = [self._inputs(p) for p in prepared]
        embedding = np.concatenate([x[0] for x in inputs])
        masks = np.concatenate([x[1] for x in inputs])
        types = [x[2][0] for x in inputs]
        try:
            output, times = owner.execute(embedding, masks, types)
        except BaseException as exc:
            self.incidents.append(str(exc))
            self.close()
            raise
        require(output.shape == (2, 64, 1024), "pair output shape changed")
        # Preserve each request's original B1 CPU-head arithmetic and row order.
        results = [
            self._finish(p, output[i : i + 1], times, start, 0, setup, owner, True)[
                "result"
            ]
            for i, p in enumerate(prepared)
        ]
        for request, result in zip(requests, results):
            if self.memo is not None:
                self.memo.put(request["key"], result)
        return results

    def answer_batch(self, states, questions):
        """At most two supplied independent requests are held for scheduling."""
        with self.mutex:
            self.last_response = {}
            self.last_schedule = {}
            self.last_execution = dict(
                submit_seconds=[],
                launches=0,
                attempted_execution_paths=0,
                completed_device_requests=0,
                failed_request=False,
                current_request_completion_unknown=False,
                memo_hit=False,
            )
            try:
                require(not self.closed, "inference owner is closed")
                if self.arithmetic == "fast" and self.bundle.get("variants", {}).get(
                    "fast", {}
                ).get("rejected_shapes"):
                    for state in states:
                        self._preflight(state, questions)
                results = []
                events = []
                self.last_schedule = dict(events=events)
                record = (
                    self.bundle.get("phase_pairs", {})
                    .get(self.arithmetic, {})
                    .get("b1_l64")
                )
                for offset in range(0, len(states), 2):
                    self.last_execution = dict(
                        submit_seconds=[],
                        launches=0,
                        attempted_execution_paths=0,
                        completed_device_requests=0,
                        failed_request=False,
                        current_request_completion_unknown=False,
                        memo_hit=False,
                    )
                    requests = [
                        self._request(state, questions)
                        for state in states[offset : offset + 2]
                    ]
                    prepared = (
                        [r["prepared"][0] for r in requests]
                        if all(len(r["prepared"]) == 1 for r in requests)
                        else []
                    )
                    eligible, reason = pair_eligibility(
                        prepared, self.phase_major, record
                    )
                    if self.phase_major and any(
                        len(r["prepared"]) != 1 for r in requests
                    ):
                        eligible = False
                        reason = "requires_one_question_per_request"
                    if any(r["hit"] is not None for r in requests):
                        eligible = False
                        reason = "memo_hit"
                    execution = []
                    events.append(
                        dict(
                            offset=offset,
                            paired=eligible,
                            reason=reason,
                            request_executions=execution,
                            memo_hits=[r["hit"] is not None for r in requests],
                            request_schedules=[
                                (
                                    "memo_hit"
                                    if r["hit"] is not None
                                    else (
                                        "phase_major_b1_b1"
                                        if eligible
                                        else r["schedule"]["kind"]
                                    )
                                )
                                for r in requests
                            ],
                        )
                    )
                    if eligible:
                        self.last_execution = {}
                        results.extend(self._pair(requests, record))
                        execution.extend([None] * len(requests))
                    else:
                        for r in requests:
                            results.append(self._complete(r))
                            execution.append(copy.deepcopy(self.last_execution))
                self.last_schedule = dict(events=events)
                return [dict(result=r) for r in results]
            except BaseException:
                self.last_execution["failed_request"] = True
                self.last_schedule.update(
                    failed=True, execution=copy.deepcopy(self.last_execution)
                )
                raise

    def close(self):
        with self.mutex:
            self.cache.close()
            self.runtime = None
            self.closed = True
            if self.memo is not None:
                self.memo.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
