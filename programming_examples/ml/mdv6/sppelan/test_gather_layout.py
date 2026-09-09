"""CPU-only independent gather mapping and persistent fail-stop host checks."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from sppelan.gather_host import (GatherProbe, frame_input, transpose_oracle, segment_oracle,
                                 INPUT_SHAPE, OUTPUT_SHAPE, INPUT_ELEMENTS, OUTPUT_ELEMENTS)
from sppelan.test_gather_probe import save_failure


class LayoutTests(unittest.TestCase):
    def test_source_4d_dma_pattern_covers_each_element_once(self):
        # Literal authored DMA dimensions: [(25,128),(4,12800),
        # (4,3200),(128,1)]. Enumerate the actual stream order, independently
        # of TensorAccessPattern and gather_segments implementations.
        stripe, row, level, pixel_lane = np.indices((25, 4, 4, 128))
        addresses = stripe * 128 + row * 12800 + level * 3200 + pixel_lane
        np.testing.assert_array_equal(np.sort(addresses.reshape(-1)), np.arange(51200))
        pixel = stripe * 16 + pixel_lane // 8
        lane = pixel_lane % 8
        semantic_addresses = ((row * 4 + level) * 400 + pixel) * 8 + lane
        np.testing.assert_array_equal(addresses, semantic_addresses)

    def test_destination_4d_dma_pattern_partitions_stripe_once(self):
        # One source stream is ordered [row,level,pixel,lane]. The four
        # receivers use [(4,8),(4,128),(16,512),(8,1)] + 32*source.
        row, level, pixel, lane = np.indices((4, 4, 16, 8))
        relative = row * 8 + level * 128 + pixel * 512 + lane
        all_destinations = []
        for source in range(4):
            addresses = relative + 32 * source
            semantic_addresses = pixel * 512 + level * 128 + (source * 4 + row) * 8 + lane
            np.testing.assert_array_equal(addresses, semantic_addresses)
            self.assertEqual(np.unique(addresses).size, 2048)
            all_destinations.append(addresses.reshape(-1))
        np.testing.assert_array_equal(np.sort(np.concatenate(all_destinations)), np.arange(8192))

    def test_composed_dma_stream_matches_full_semantic_concat(self):
        # uint32 identities avoid either uint16 sentinel's individual aliasing.
        identities = np.arange(INPUT_ELEMENTS, dtype=np.uint32).reshape(INPUT_SHAPE)
        stripe, row, level, pixel_lane = np.indices((25, 4, 4, 128))
        source_addresses = (stripe * 128 + row * 12800 + level * 3200 + pixel_lane).reshape(25, 2048)
        row, level, pixel, lane = np.indices((4, 4, 16, 8))
        destination_relative = (row * 8 + level * 128 + pixel * 512 + lane).reshape(-1)
        result = np.empty((25, 8192), np.uint32)
        for source in range(4):
            stream = identities[source].reshape(-1)[source_addresses]
            result[:, destination_relative + source * 32] = stream
        expected = identities.transpose(3, 2, 0, 1, 4).reshape(25, 8192)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(np.unique(result).size, INPUT_ELEMENTS)

    def test_oracles_agree_all_patterns(self):
        for frame in range(5):
            with self.subTest(frame=frame):
                _, bits = frame_input(frame)
                np.testing.assert_array_equal(transpose_oracle(bits), segment_oracle(bits))

    def test_sentinel_pair_is_unique_not_either_half_alone(self):
        _, low = frame_input(0)
        _, high = frame_input(1)
        self.assertLess(np.unique(low).size, INPUT_ELEMENTS)
        self.assertLess(np.unique(high).size, INPUT_ELEMENTS)
        reconstructed = low.astype(np.uint32) | (high.astype(np.uint32) << 16)
        np.testing.assert_array_equal(reconstructed.reshape(-1), np.arange(INPUT_ELEMENTS))

    def test_explicit_coordinate_mapping_and_all_destinations(self):
        _, bits = frame_input(2)
        output = transpose_oracle(bits)
        for dest in range(4):
            for stripe, pixel, level, source, row, lane in ((0, 0, 0, 0, 0, 0),
                    (24, 15, 3, 3, 3, 7), (12, 7, 2, 1, 3, 4), (1, 0, 1, 2, 2, 2)):
                k = level * 128 + source * 32 + row * 8 + lane
                self.assertEqual(output[dest, stripe, pixel, k],
                                 bits[source, row, level, stripe * 16 + pixel, lane])
            np.testing.assert_array_equal(output[0], output[dest])
        self.assertFalse(np.shares_memory(output, bits))

    def test_random_changes_between_cycles(self):
        self.assertFalse(np.array_equal(frame_input(2)[1], frame_input(7)[1]))

    def test_contracts(self):
        for bits in (np.zeros(INPUT_SHAPE, np.float32), np.zeros(INPUT_ELEMENTS, np.uint16)):
            for oracle in (transpose_oracle, segment_oracle):
                with self.assertRaises(ValueError):
                    oracle(bits)
        with self.assertRaises(ValueError):
            frame_input(-1)

    def make_probe(self):
        backend = Mock()
        def buffer(n, dtype):
            result = Mock(data=np.zeros(n, dtype))
            result.numpy.return_value = result.data
            return result
        backend.iron.zeros.side_effect = buffer
        backend.DefaultNPURuntime.run.return_value.is_success.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            for name in ('gather_probe.xclbin', 'gather_probe.bin'):
                Path(directory, name).touch()
            probe = GatherProbe(directory, backend)
        return probe, backend

    def test_persistent_arenas_one_load_and_sync_per_run(self):
        probe, backend = self.make_probe()
        for frame in range(2):
            _, bits = frame_input(frame)
            output = probe.run(bits)
            np.testing.assert_array_equal(probe.input.data, bits.reshape(-1))
            self.assertEqual(output.shape, OUTPUT_SHAPE)
            self.assertFalse(np.shares_memory(output, probe.output.data))
        backend.DefaultNPURuntime.load.assert_called_once()
        self.assertEqual(backend.iron.zeros.call_count, 2)
        self.assertEqual(probe.input._sync_to_device.call_count, 2)
        self.assertEqual(probe.output.numpy.call_count, 2)
        backend.DefaultNPURuntime.run.assert_called_with(probe.handle, [probe.input, probe.output])

    def test_each_runtime_failure_invalidates_without_retry(self):
        for stage in ('upload', 'run', 'result', 'readback', 'output_contract'):
            with self.subTest(stage=stage):
                probe, backend = self.make_probe()
                if stage == 'upload':
                    probe.input._sync_to_device.side_effect = RuntimeError('upload failed')
                elif stage == 'run':
                    backend.DefaultNPURuntime.run.side_effect = RuntimeError('run failed')
                elif stage == 'result':
                    backend.DefaultNPURuntime.run.return_value.is_success.return_value = False
                elif stage == 'readback':
                    probe.output.numpy.side_effect = RuntimeError('readback failed')
                else:
                    probe.output.numpy.return_value = np.zeros(1, np.uint16)
                with self.assertRaises(RuntimeError):
                    probe.run(frame_input(0)[1])
                with self.assertRaisesRegex(RuntimeError, 'invalid after failure'):
                    probe.run(frame_input(1)[1])
                self.assertEqual(backend.DefaultNPURuntime.run.call_count, 0 if stage == 'upload' else 1)
                backend.DefaultNPURuntime.load.assert_called_once()

    def test_missing_artifacts_never_load(self):
        backend = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                GatherProbe(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def test_exclusive_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            _, bits = frame_input(0)
            expected = transpose_oracle(bits)
            path = save_failure(directory, 0, 'low', bits, expected, None, 'failed')
            with np.load(path) as data:
                np.testing.assert_array_equal(data['input'], bits)
                self.assertNotIn('observed', data.files)
            with self.assertRaises(FileExistsError):
                save_failure(directory, 0, 'low', bits, expected, expected, 'later')


if __name__ == '__main__':
    unittest.main()
