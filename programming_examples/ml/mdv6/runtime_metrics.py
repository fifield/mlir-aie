"""Scoped, single-threaded observation of the installed synchronous XRT API.

These are Python API counts, not hardware traces. Tensor sync bytes are whole
BO sizes, including padding/instructions when they use the tensor API. Direct
pyxrt sync calls, device DMA, context switches/evictions inside the driver and
device cycles are not observable here. Do not nest this with other patchers.
"""
from collections import Counter
from pathlib import Path


class RuntimeMetrics:
    _active = False

    def __init__(self, runtime, tensor_class):
        self.runtime = runtime
        self.tensor_class = tensor_class
        self.counts = Counter()
        self._patches = []

    def _patch(self, obj, name, replacement):
        own = vars(obj).get(name)
        had_own = name in vars(obj)
        self._patches.append((obj, name, had_own, own))
        setattr(obj, name, replacement)

    def __enter__(self):
        if RuntimeMetrics._active:
            raise RuntimeError("RuntimeMetrics is process-global; nesting is unsupported")
        RuntimeMetrics._active = True
        self.counts.clear()
        try:
            run = self.runtime.run

            def observed_run(*args, **kwargs):
                self.counts['run_calls'] += 1
                result = run(*args, **kwargs)
                self.counts['returned_runs'] += 1
                if result.is_success():
                    self.counts['completed_runs'] += 1
                return result

            self._patch(self.runtime, 'run', observed_run)
            load = self.runtime.load

            def observed_load(kernel, *args, **kwargs):
                self.counts['load_calls'] += 1
                cache = getattr(self.runtime, '_context_cache', None)
                if cache is not None:
                    path = Path(kernel.xclbin_path).resolve()
                    key = (str(path), path.stat().st_mtime)
                    if key not in cache:
                        self.counts['context_cache_misses'] += 1
                return load(kernel, *args, **kwargs)

            self._patch(self.runtime, 'load', observed_load)
            if hasattr(self.runtime, '_evict'):
                evict = self.runtime._evict

                def observed_evict(*args, **kwargs):
                    self.counts['eviction_calls'] += 1
                    return evict(*args, **kwargs)

                self._patch(self.runtime, '_evict', observed_evict)
            for direction in ('to', 'from'):
                name = f'_sync_{direction}_device'
                original = getattr(self.tensor_class, name)

                def observed_sync(tensor, *args, _original=original,
                                  _direction=direction, **kwargs):
                    self.counts[f'sync_{_direction}_calls'] += 1
                    # buffer_object()/size() do not materialize host data.
                    size = tensor.buffer_object().size()
                    result = _original(tensor, *args, **kwargs)
                    self.counts[f'sync_{_direction}_bytes'] += int(size)
                    return result

                self._patch(self.tensor_class, name, observed_sync)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def snapshot(self):
        keys = ('run_calls', 'returned_runs', 'completed_runs', 'load_calls',
                'context_cache_misses', 'eviction_calls', 'sync_to_calls',
                'sync_from_calls', 'sync_to_bytes', 'sync_from_bytes')
        result = {key: self.counts[key] for key in keys}
        if not hasattr(self.runtime, '_context_cache'):
            result['context_cache_misses'] = None
        if not hasattr(self.runtime, '_evict'):
            result['eviction_calls'] = None
        result.update(device_dma_bytes=None, device_cycles=None,
                      driver_context_switches=None, host_wait_calls=None)
        return result

    def __exit__(self, *exc):
        for obj, name, had_own, original in reversed(self._patches):
            if had_own:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self._patches.clear()
        RuntimeMetrics._active = False
        return False
