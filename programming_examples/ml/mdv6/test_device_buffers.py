import unittest
from dataclasses import replace
from device_buffers import TensorContract, DeviceBuffer


def contract():
    return TensorContract("inter", (3, 8, 8, 16), (3, 8, 8, 16),
                          (1024, 128, 16, 1), "tile-HWC", "proof", "stage1",
                          ("stage2",), 1, (0, 0, 0, 0), (3, 8, 8, 16))


class FakeTensor:
    nbytes = 6144
    device = "npu"
    dtype = "uint16"
    def numpy(self):
        raise AssertionError("unexpected host sync")


class BufferTests(unittest.TestCase):
    def test_reuse_preserves_identity_without_sync(self):
        t = FakeTensor()
        b = DeviceBuffer(contract(), t)
        b.mark_written()
        self.assertIs(b.for_consumer(replace(contract(), name="stage2_input")), t)
        self.assertIn("inter", repr(b))

    def test_invalid_contents(self):
        b = DeviceBuffer(contract(), FakeTensor())
        with self.assertRaises(RuntimeError):
            b.for_consumer(contract())
        b.mark_written()
        b.invalidate()
        with self.assertRaises(RuntimeError):
            b.download()

    def test_layout_mismatch(self):
        with self.assertRaises(ValueError):
            contract().require_compatible(replace(contract(), layout="CHW"))

    def test_invalid_metadata(self):
        for kwargs in ({"strides": (1, 1, 1, 1)}, {"valid_origin": (0, 0, 0, 1)},
                       {"dtype": "fp32"}, {"last_use": -1}, {"owner": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(contract(), **kwargs)

    def test_no_implicit_host_conversion(self):
        b = DeviceBuffer(contract(), FakeTensor())
        with self.assertRaises(TypeError):
            b.__array__()

    def test_reject_wrong_transport_dtype(self):
        t = FakeTensor()
        t.dtype = "float32"
        with self.assertRaises(ValueError):
            DeviceBuffer(contract(), t)

    def test_reject_wrong_size_or_cpu_residency(self):
        t = FakeTensor()
        t.nbytes = 1
        with self.assertRaises(ValueError):
            DeviceBuffer(contract(), t)
        t.nbytes = 6144
        b = DeviceBuffer(contract(), t)
        t.device = "cpu"
        with self.assertRaises(RuntimeError):
            b.mark_written()


if __name__ == "__main__":
    unittest.main()
