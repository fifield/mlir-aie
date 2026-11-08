"""Optional complete-response memo; ordered prepared tensors and immutable identity."""

import hashlib, json
from collections import OrderedDict


class BoundedMemo:
    def __init__(self, max_entries, max_bytes):
        if (
            not isinstance(max_entries, int)
            or not isinstance(max_bytes, int)
            or min(max_entries, max_bytes) < 1
        ):
            raise ValueError("Positive memo bounds required")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.values = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self.values:
            self.misses += 1
            return None
        value = self.values.pop(key)
        self.values[key] = value
        self.hits += 1
        return json.loads(value)

    def put(self, key, value):
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        size = len(key.encode()) + len(encoded)
        if size > self.max_bytes:
            return False
        if key in self.values:
            self.bytes -= len(key.encode()) + len(self.values.pop(key))
        while self.values and (
            len(self.values) >= self.max_entries or self.bytes + size > self.max_bytes
        ):
            old_key, old = self.values.popitem(last=False)
            self.bytes -= len(old_key.encode()) + len(old)
            self.evictions += 1
        self.values[key] = encoded
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return True

    def clear(self):
        self.values.clear()
        self.bytes = 0


def key_for(prepared, identity, request):
    metadata = dict(
        identity=identity,
        batch=len(prepared["valid_lengths"]),
        bucket=prepared["bucket"],
        layout="whole-batch-native-ordered-rows",
        valid_lengths=prepared["valid_lengths"],
        ordered_request=request,
    )
    h = hashlib.sha256(
        json.dumps(
            metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    )
    for field in ("input_ids", "attention_mask", "qtype", "marker_pos", "marker_mask"):
        x = prepared["batch"][field].numpy()
        h.update(field.encode())
        h.update(str(x.dtype).encode())
        h.update(json.dumps(list(x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def complete_key(parts, identity, request):
    return hashlib.sha256(
        json.dumps(
            dict(parts=parts, identity=identity, request=request),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
