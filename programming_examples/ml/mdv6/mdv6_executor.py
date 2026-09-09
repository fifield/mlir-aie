"""Persistent baseline for fusion experiments, NOT a device-resident graph.

The graph is shared with the legacy test. Model/fused weights stay alive; packed
weight caches, instruction handles, and size-based synchronous BO pools remain
owned by run_tiled_mc and are populated on the first frame. Every convolution
still materializes its output on the host. Do not use concurrently or mutate
model parameters. After a hardware/context failure, terminate the process and
follow the documented driver recovery procedure before constructing a new one.
"""
import contextlib
import io
import os
import time


DIAGNOSTIC_MODES = (
    'DEBUG_GEMM_TRAINED', 'DEBUG_GEMM', 'DEBUG_KERNEL', 'DEBUG_LAYERS',
    'DEBUG_CONV1', 'DEBUG_ELAN2', 'DEBUG_ELAN2_SUB',
)


def reject_diagnostics():
    enabled = [key for key in DIAGNOSTIC_MODES if os.environ.get(key)]
    if enabled:
        raise ValueError('inference-only execution requires unsetting ' + ', '.join(enabled))


class MDV6Executor:
    """Single-threaded, immutable-weight executor with explicit host boundaries.

    Input: CPU bf16 contiguous NCHW (1,3,640,640), all elements valid, borrowed
    through run_frame. Internal graph tensors use host HWC/NCHW and legacy patch
    layouts. Output: CPU detection triples owned by the caller. No tensor here
    claims on-chip residency; no asynchronous lifetime allocator is provided.
    """
    route = 'persistent-hybrid-host-materialized'

    def __init__(self, backend=None, verbose=False):
        reject_diagnostics()
        if backend is None:
            import test_full_model_mc as backend
        self.backend = backend
        self.verbose = verbose
        self.failed = False
        self.frame_setup_ms = 0.0
        with self._output():
            self.model = backend.load_model()
            self.conv0_weights = backend.pad_conv0_weights(self.model)

    def _output(self):
        return contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO())

    def run_frame(self, x):
        reject_diagnostics()
        if self.failed:
            raise RuntimeError('executor invalid after failed frame; recover hardware and restart process')
        if tuple(x.shape) != (1, 3, 640, 640):
            raise ValueError('input shape must be (1,3,640,640)')
        if x.device.type != 'cpu' or x.dtype != self.backend.torch.bfloat16 or not x.is_contiguous():
            raise ValueError('input must be contiguous CPU bfloat16 NCHW')
        try:
            with self._output(), self.backend.torch.no_grad():
                result, _ = self.backend.run_hybrid_forward(self.model, x, self.conv0_weights)
            return result
        except Exception:
            self.failed = True
            raise

    def reference(self, x):
        """CPU oracle: explicitly call outside the inference timer."""
        with self.backend.torch.no_grad():
            return self.model(x)


class LegacyGraphExecutor(MDV6Executor):
    """Matched-scope comparator: reconstruct weights before each reference.

    Mirrors legacy main's model lifetime and per-frame stem-weight packing;
    deliberately excludes model construction and CPU reference from frame time.
    Use a separate process per route to avoid shared runtime-cache contamination.
    """
    route = 'legacy-reconstructed-hybrid-host-materialized'

    def reference(self, x):
        started = time.perf_counter()
        with self._output():
            self.model = self.backend.load_model()
        self.conv0_weights = None
        self.frame_setup_ms = (time.perf_counter() - started) * 1000
        return super().reference(x)
