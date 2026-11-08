"""One ELF context, persistent packed buffers, and bounded XRT submissions."""

import contextlib
import fcntl
import hashlib
import time
import zlib
import numpy as np
from .common import ROOT, DATA, digest, require, read
from .mixed_contract import validate_plan

# Never release buffers/context belonging to a run with no confirmed terminal state.
TERMINAL = {4, 5, 6, 8, 9, 10, 11}


def contained(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT.resolve()), "artifact path escapes laya_npu")
    return path


def blob_sources(plan):
    result = []
    for record in plan.get("blob_roots", []):
        root = contained(record["path"])
        manifest = contained(record["manifest"])
        require(digest(manifest) == record["sha256"], "blob manifest changed")
        result.append((root, read(manifest)))
    return result


def load_blob(key, sources=()):
    if key is None:
        return None
    require(
        len(key) == 64 and all(c in "0123456789abcdef" for c in key),
        "invalid operand identity",
    )
    path = None
    for root, manifest in sources:
        if key in manifest:
            path = contained(root / f"{key}.z")
            require(digest(path) == manifest[key], "compressed operand changed")
            break
    if path is None:
        path = contained(DATA / "blobs" / f"{key}.z")
    data = zlib.decompress(path.read_bytes())
    require(hashlib.sha256(data).hexdigest() == key, "model operand hash mismatch")
    return data


class Runtime:
    def __init__(self, plan, use_lock=True):
        init_started = time.perf_counter()
        self.init_phases = {}
        validate_plan(plan)
        import pyxrt as x

        self.x = x
        self.plan = plan
        self.use_lock = use_lock
        self.device = self.context = self.elf = None
        self.bos = []
        self.hosts = []
        self.kernels = {}
        self.closed = False
        self.failed = False
        self.lock_wait_seconds = 0.0
        self.pending = None
        self.retaining_nonterminal = False
        self.blob_sources = blob_sources(plan)
        try:
            path = contained(plan["elf"])
            require(digest(path) == plan["elf_sha256"], "ELF hash mismatch")
            self.device = x.device(0)
            self.elf = x.elf(str(path))
            require(digest(path) == plan["elf_sha256"], "ELF changed while loading")
            with self.lock():
                self.context = x.hw_context(self.device, self.elf)
            self.init_phases["validate_context"] = time.perf_counter() - init_started
            phase = time.perf_counter()
            self.bos = [x.ext.bo(self.device, n) for n in plan["arenas"]]
            self.hosts = [
                np.frombuffer(b.map(), np.uint8, count=n)
                for b, n in zip(self.bos, plan["arenas"])
            ]
            for host in self.hosts:
                host.fill(0)
            self.init_phases["allocate_map_zero"] = time.perf_counter() - phase
            phase = time.perf_counter()
            for a in plan["allocations"]:
                data = load_blob(a["initial"], self.blob_sources)
                if data is not None:
                    require(
                        len(data) == a["nbytes"], "initial allocation size mismatch"
                    )
                    self.hosts[a["arena"]][a["offset"] : a["offset"] + a["nbytes"]] = (
                        np.frombuffer(data, np.uint8)
                    )
            self.init_phases["decompress_hash_copy"] = time.perf_counter() - phase
            phase = time.perf_counter()
            for b in self.bos:
                b.sync(x.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.init_phases["initial_bo_upload"] = time.perf_counter() - phase
            phase = time.perf_counter()
            self.types = [
                load_blob(k, self.blob_sources) for k in plan["inputs"]["type_blobs"]
            ]
            headers = {
                tuple(d["bindings"][-1])
                for d in plan["dispatches"]
                if d["label"].startswith("eng.")
            }
            self.headers = []
            for binding in headers:
                arena, off, n = self.location(binding)
                self.headers.append(
                    (
                        arena,
                        off,
                        n,
                        hashlib.sha256(self.hosts[arena][off : off + n]).hexdigest(),
                    )
                )
            self.kernels = {
                g["instance"]: x.ext.kernel(self.context, "main:" + g["instance"])
                for g in plan["groups"]
            }
            self.init_phases["types_headers_kernels"] = time.perf_counter() - phase
            self.init_phases["total"] = time.perf_counter() - init_started
        except BaseException:
            self.close()
            raise

    @contextlib.contextmanager
    def lock(self):
        if not self.use_lock:
            yield
            return
        with open("/tmp/npu-dev.lock", "a") as f:
            start = time.perf_counter()
            fcntl.flock(f, fcntl.LOCK_EX)
            self.lock_wait_seconds += time.perf_counter() - start
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def location(self, binding):
        slot, offset, nbytes = binding
        a = self.plan["allocations"][slot]
        require(
            0 <= offset and offset + nbytes <= a["nbytes"], "binding outside allocation"
        )
        return a["arena"], a["offset"] + offset, nbytes

    def upload(self, binding, data):
        arena, offset, nbytes = self.location(binding)
        value = np.frombuffer(data, np.uint8)
        require(value.nbytes == nbytes, "dynamic input size mismatch")
        self.hosts[arena][offset : offset + nbytes] = value
        self.bos[arena].sync(
            self.x.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, offset
        )

    def _wait_retained(self, run):
        self.retaining_nonterminal = True
        # No close/unlock/eviction while the device can still reference our BOs.
        while True:
            try:
                if int(run.state()) in TERMINAL:
                    break
            except BaseException:
                pass
            try:
                time.sleep(0.1)
            except BaseException:
                # Cancellation does not authorize freeing device-owned memory.
                pass
        self.retaining_nonterminal = False

    def submit(self, group):
        require(
            not self.closed and not self.failed, "runtime closed after failed command"
        )
        run = self.x.run(self.kernels[group["instance"]])
        for i, bo in enumerate(self.bos):
            run.set_arg(i, bo)
        started = time.perf_counter()
        state = None
        self.pending = run
        try:
            run.start()
            while time.perf_counter() - started <= 5.0:
                current = int(run.state())
                if current in TERMINAL:
                    state = current
                    break
                time.sleep(20e-6)
            elapsed = time.perf_counter() - started
            require(
                state == 4 and elapsed <= group["deadline"],
                f"NPU command failed: state={state}, seconds={elapsed:.6f}",
            )
            return elapsed
        except BaseException:
            self.failed = True
            raise
        finally:
            if state not in TERMINAL:
                self._wait_retained(run)
            self.pending = None

    def execute(self, embedding, masks, types):
        # Hold one lock over dynamic uploads, all dependent groups and downloads.
        with self.lock():
            return self._execute(embedding, masks, types)

    def _execute(self, embedding, masks, types):
        require(not self.closed and not self.failed, "runtime unavailable")
        inputs = self.plan["inputs"]
        for i, q in enumerate(types):
            require(int(q) in range(3), "unsupported question type")
            self.upload(inputs["embedding"][i], embedding[i].tobytes())
            self.upload(inputs["mask"][i], masks[i].tobytes())
            self.upload(inputs["type"][i], self.types[int(q)])
        times = [self.submit(g) for g in self.plan["groups"]]
        for arena, off, n, expected in self.headers:
            self.bos[arena].sync(
                self.x.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, n, off
            )
            require(
                hashlib.sha256(self.hosts[arena][off : off + n]).hexdigest()
                == expected,
                "engine control header changed on device",
            )
        from ml_dtypes import bfloat16

        output = []
        for binding in self.plan["outputs"]:
            arena, offset, nbytes = self.location(binding)
            self.bos[arena].sync(
                self.x.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, offset
            )
            output.append(
                self.hosts[arena][offset : offset + nbytes]
                .view(bfloat16)
                .astype(np.float32)
                .reshape(self.plan["length"], 1024)
            )
        return np.stack(output), times

    def close(self):
        if self.closed:
            return
        require(
            self.pending is None and not self.retaining_nonterminal,
            "cannot close nonterminal runtime",
        )
        self.closed = True
        self.kernels.clear()
        self.hosts.clear()
        self.bos.clear()
        self.context = self.elf = self.device = None
