"""Arena-byte/context bounded LRU copied from the validated lifecycle screen."""

from collections import OrderedDict


class RuntimeCache:
    def __init__(self, entries=2, byte_limit=2 * 1024**3):
        if (
            not isinstance(entries, int)
            or not isinstance(byte_limit, int)
            or entries < 1
            or byte_limit < 1
        ):
            raise ValueError("Cache bounds must be positive integers")
        self.entries, self.byte_limit = entries, byte_limit
        self.items = OrderedDict()
        self.peak_bytes = 0

    @property
    def resident_bytes(self):
        return sum(size for _, size in self.items.values())

    def get(self, key, nbytes, factory):
        event = dict(
            key=key, hit=key in self.items, evicted=[], requested_arena_bytes=nbytes
        )
        if key in self.items:
            runtime, size = self.items[key]
            if size != nbytes:
                raise ValueError("Cached arena extent changed for immutable identity")
            self.items.move_to_end(key)
        else:
            if nbytes > self.byte_limit:
                raise ValueError(
                    f"{key}: arena bytes {nbytes} exceed cache bound {self.byte_limit}"
                )
            while self.items and (
                len(self.items) >= self.entries
                or self.resident_bytes + nbytes > self.byte_limit
            ):
                old_key, (old, _) = next(iter(self.items.items()))
                old.close()  # Never drop ownership if close cannot confirm terminal safety.
                del self.items[old_key]
                event["evicted"].append(old_key)
            runtime, size = factory(), nbytes
            self.items[key] = (runtime, size)
            event["init_seconds"] = getattr(runtime, "init_phases", {})
        event.update(
            resident_arena_bytes=self.resident_bytes, resident_contexts=len(self.items)
        )
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        return runtime, event

    def close(self):
        for key in list(self.items):
            self.items[key][0].close()
            del self.items[key]
