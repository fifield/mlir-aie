"""Explicit synchronous external-BO contracts for experimental MDV6 islands.

No allocator, implicit layout conversion, or context-surviving on-chip storage
is promised here. Element strides describe storage, not DMA descriptor strides.
bf16 values are transported as uint16 bit patterns (never numerically cast).
"""
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class TensorContract:
    name: str
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    strides: tuple[int, ...]
    layout: str
    owner: str
    producer: str
    consumers: tuple[str, ...]
    last_use: int
    valid_origin: tuple[int, ...]
    valid_shape: tuple[int, ...]
    dtype: str = "bf16"
    storage: str = "external_bo"

    def __post_init__(self):
        n = len(self.physical_shape)
        if not self.name or not self.owner or not self.layout:
            raise ValueError("name, owner and layout are required")
        if not n or len(self.logical_shape) != n or any(
            len(x) != n for x in (self.strides, self.valid_origin, self.valid_shape)
        ):
            raise ValueError("shape, strides and valid region ranks must agree")
        if any(d <= 0 for d in self.logical_shape + self.physical_shape + self.strides):
            raise ValueError("dimensions and strides must be positive")
        if any(o < 0 or s <= 0 or o + s > p for o, s, p in zip(
            self.valid_origin, self.valid_shape, self.physical_shape
        )) or self.valid_shape != self.logical_shape:
            raise ValueError("valid region must contain exactly the logical tensor")
        if self.dtype != "bf16" or self.storage != "external_bo" or self.last_use < 0:
            raise ValueError("only bf16 external BOs with explicit last use are supported")
        # This initial contract deliberately rejects views/overlapping layouts.
        expected = tuple(prod(self.physical_shape[i + 1:]) for i in range(n))
        if self.strides != expected:
            raise ValueError("only contiguous physical storage is currently supported")

    @property
    def nbytes(self):
        return 2 * prod(self.physical_shape)

    def require_compatible(self, consumer):
        fields = ("logical_shape", "physical_shape", "strides", "layout", "dtype",
                  "storage", "valid_origin", "valid_shape")
        if any(getattr(self, f) != getattr(consumer, f) for f in fields):
            raise ValueError("producer/consumer physical layouts differ; explicit repacker required")


class DeviceBuffer:
    """Own a named tensor without permitting accidental host materialization.

    Runtime tensor ownership remains with this object/caller. Sequential run()
    completion is required before mark_written(); asynchronous reuse is not
    supported. A reset invalidates contents; it does not recover an XRT context.
    """
    def __init__(self, contract, tensor):
        if tensor.nbytes != contract.nbytes:
            raise ValueError("BO size does not match physical contract")
        # np.uint16 (class) and np.dtype('uint16') are both used by runtimes;
        # avoid importing NumPy merely to validate metadata in CPU-only users.
        dtype_name = getattr(tensor.dtype, "__name__", str(tensor.dtype))
        if dtype_name != "uint16":
            raise ValueError("bf16 external buffers require uint16 bit-pattern storage")
        self.contract = contract
        self._tensor = tensor
        self._valid = False

    def destination(self):
        if self._tensor.device != "npu":
            raise RuntimeError("output BO must already be NPU accessible")
        return self._tensor

    def mark_written(self):
        self.destination()
        self._valid = True

    def for_consumer(self, contract):
        self.contract.require_compatible(contract)
        if not self._valid:
            raise RuntimeError("device contents are not valid")
        return self.destination()

    def invalidate(self):
        self._valid = False

    def download(self):
        if not self._valid:
            raise RuntimeError("device contents are not valid")
        return self._tensor.numpy().copy()

    def __repr__(self):
        return f"DeviceBuffer({self.contract.name!r}, valid={self._valid})"

    def __array__(self, *args, **kwargs):
        raise TypeError("use explicit download() at a declared host boundary")
