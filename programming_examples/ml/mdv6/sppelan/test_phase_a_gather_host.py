"""Independent packing/pooling/gather and persistent numerical host CPU gates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sppelan.phase_a_gather_host import (PhaseAGather, INPUT_SHAPE, WEIGHT_SHAPE,
    OUTPUT_SHAPE, METADATA_SHAPE, pack_weights, frame_weights, frame_input,
    gather_oracle, validate_output)
from sppelan.phase_a_host import values, exact_bits
from sppelan.test_phase_a_gather import validate_counts, checked_frame, save_failure


class PhaseAGatherHostTests(unittest.TestCase):
    def raw(self):
        return np.arange(33024, dtype=np.uint16)

    def counts(self):
        return dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                    sync_to_calls=2, sync_from_calls=2,
                    sync_to_bytes=272384, sync_from_bytes=1639424)

    def test_all16_weight_slices_independent_coordinate_packing(self):
        raw = self.raw()
        packed = pack_weights(raw)
        self.assertEqual(packed.shape, (4, 4, 2, 1056))
        for col in range(4):
            for worker in range(4):
                for block in range(2):
                    chunk = packed[col, worker, block]
                    for k in range(128):
                        for lane in range(8):
                            output = 32 * col + 8 * worker + lane
                            self.assertEqual(chunk[k * 8 + lane], raw[output * 256 + block * 128 + k])
                    np.testing.assert_array_equal(chunk[1024:1032], raw[32768 + 32*col + 8*worker:32776 + 32*col + 8*worker])
                    np.testing.assert_array_equal(chunk[1032:1040], raw[32896 + 32*col + 8*worker:32904 + 32*col + 8*worker])
                    self.assertFalse(chunk[1040:].any())
                    self.assertEqual(((4*col + worker)*2112 + block*1056)*2 % 64, 0)

    def test_rotated_weights_keep_all_trained_shards_and_bn(self):
        raw = self.raw()
        for frame in range(16):
            rotated = frame_weights(raw, frame)
            for channel in range(128):
                original = (channel - frame * 8) % 128
                np.testing.assert_array_equal(rotated[channel*256:(channel+1)*256], raw[original*256:(original+1)*256])
                self.assertEqual(rotated[32768+channel], raw[32768+original])
                self.assertEqual(rotated[32896+channel], raw[32896+original])
        np.testing.assert_array_equal(raw, self.raw())

    def test_fullwidth_pool_and_gather_independent_clipped_windows(self):
        row, col, channel = np.indices((20, 20, 128))
        floats = (-((row * 3 + col * 7 + channel) % 19) - 1).astype(np.float32)
        floats[0, 0, 0], floats[19, 19, 127] = 1, 2
        floats[:, :, 1] = 0
        floats.view(np.uint32)[::2, ::2, 1] = 0x80000000
        f0 = exact_bits(floats).reshape(400, 128)
        independent = [f0]
        current = values(f0).reshape(20, 20, 128)
        for _ in range(3):
            following = np.empty_like(current)
            for y in range(20):
                for x in range(20):
                    window = current[max(0,y-2):min(20,y+3), max(0,x-2):min(20,x+3)].reshape(-1,128)
                    following[y,x] = window[window.argmax(axis=0), np.arange(128)]
            independent.append(exact_bits(following).reshape(400,128))
            current = following
        expected = np.stack(independent).transpose(1,0,2).reshape(25,16,512)
        observed = gather_oracle(f0)
        for destination in range(4):
            np.testing.assert_array_equal(observed[destination], expected)

    def test_gather_channel_identity_every_destination(self):
        f0 = exact_bits(np.broadcast_to(np.arange(128, dtype=np.float32), (400,128)))
        observed = gather_oracle(f0)
        for level in range(4):
            for source in range(4):
                for worker in range(4):
                    k = level*128 + source*32 + worker*8
                    np.testing.assert_array_equal(observed[:,:,:,k:k+8],
                        np.broadcast_to(f0[0,source*32+worker*8:source*32+worker*8+8], (4,25,16,8)))

    def test_axis_mutations_stale_destination_and_nonfinite_rejected(self):
        f0 = exact_bits(np.random.default_rng(5).integers(-8,9,(400,128)).astype(np.float32))
        expected = gather_oracle(f0)
        mutations = [np.roll(expected,32,axis=3), np.roll(expected,128,axis=3),
                     np.roll(expected,8,axis=3), np.roll(expected,1,axis=1)]
        stale = expected.copy()
        stale[3] = 0
        mutations.append(stale)
        nonfinite = expected.copy()
        nonfinite[0,0,0,0] = 0x7f80
        mutations.append(nonfinite)
        for wrong in mutations:
            with self.assertRaises(RuntimeError):
                validate_output(wrong, expected, np.zeros(METADATA_SHAPE,np.uint16))

    def test_every_metadata_word_is_strict(self):
        expected = np.zeros(OUTPUT_SHAPE,np.uint16)
        for word in range(512):
            metadata = np.zeros(METADATA_SHAPE,np.uint16)
            metadata.reshape(-1)[word] = 1
            with self.assertRaisesRegex(RuntimeError,'floor0'):
                validate_output(expected, expected, metadata)

    def test_input_patterns_finite_changing_and_preserved(self):
        raw = exact_bits(np.linspace(.125, 1, 33024, dtype=np.float32))
        for frame in range(6):
            name,bits = frame_input(frame,raw)
            self.assertEqual(bits.shape,INPUT_SHAPE)
            self.assertTrue(np.isfinite(values(bits)).all())
        self.assertFalse(np.array_equal(frame_input(0,raw)[1],frame_input(6,raw)[1]))
        self.assertIn(0x8000,frame_input(5,raw)[1])

    def make_probe(self):
        backend=Mock()
        def buffer(n,dtype):
            result=Mock(data=np.zeros(n,dtype))
            result.numpy.return_value=result.data
            return result
        backend.iron.zeros.side_effect=buffer
        backend.DefaultNPURuntime.run.return_value.is_success.return_value=True
        with tempfile.TemporaryDirectory() as directory:
            for name in ('spp_phase_a_gather.xclbin','spp_phase_a_gather.bin'):
                Path(directory,name).touch()
            probe=PhaseAGather(directory,backend)
        return probe,backend

    def test_persistent_native_fourbo_bindings_and_input_preservation(self):
        probe,backend=self.make_probe()
        bits=np.zeros(INPUT_SHAPE,np.uint16)
        weights=pack_weights(self.raw())
        before=weights.copy()
        for _ in range(2):
            out,metadata=probe.run(bits,weights)
            self.assertFalse(np.shares_memory(out,probe.output.data))
            self.assertFalse(np.shares_memory(metadata,probe.metadata.data))
        np.testing.assert_array_equal(weights,before)
        np.testing.assert_array_equal(probe.weights.data,weights.reshape(-1))
        np.testing.assert_array_equal(probe.input.data,bits.reshape(-1))
        backend.DefaultNPURuntime.load.assert_called_once()
        self.assertEqual(backend.iron.zeros.call_count,4)
        self.assertEqual(backend.DefaultNPURuntime.run.call_count,2)
        for buffer in (probe.input,probe.weights):
            self.assertEqual(buffer._sync_to_device.call_count,2)
        for buffer in (probe.output,probe.metadata):
            self.assertEqual(buffer.numpy.call_count,2)
        backend.DefaultNPURuntime.run.assert_called_with(probe.handle,[probe.input,probe.weights,probe.output,probe.metadata])

    def test_runtime_failure_poison_no_retry(self):
        for stage in ('input','weight','run','result','output','metadata','contract'):
            probe,backend=self.make_probe()
            if stage in ('input','weight'):
                getattr(probe,'input' if stage=='input' else 'weights')._sync_to_device.side_effect=RuntimeError('upload')
            elif stage=='run':
                backend.DefaultNPURuntime.run.side_effect=RuntimeError('run')
            elif stage=='result':
                backend.DefaultNPURuntime.run.return_value.is_success.return_value=False
            elif stage in ('output','metadata'):
                getattr(probe,stage).numpy.side_effect=RuntimeError('download')
            else:
                probe.metadata.numpy.return_value=np.zeros(1,np.uint16)
            with self.assertRaises(RuntimeError):
                probe.run(np.zeros(INPUT_SHAPE,np.uint16),pack_weights(self.raw()))
            with self.assertRaisesRegex(RuntimeError,'invalid after failure'):
                probe.run(np.zeros(INPUT_SHAPE,np.uint16),pack_weights(self.raw()))
            self.assertLessEqual(backend.DefaultNPURuntime.run.call_count,1)

    def test_semantic_metadata_count_failures_poison_and_save_observations(self):
        for kind in ('semantic','metadata','counts'):
            probe,backend=self.make_probe()
            counts=self.counts()
            if kind=='semantic': probe.output.data[0]=1
            if kind=='metadata': probe.metadata.data[0]=1
            if kind=='counts': counts['run_calls']=2
            metrics=Mock()
            metrics.return_value.__enter__=Mock(return_value=Mock(snapshot=Mock(return_value=counts)))
            metrics.return_value.__exit__=Mock(return_value=False)
            record={}
            with self.assertRaises(RuntimeError):
                checked_frame(probe,np.zeros(INPUT_SHAPE,np.uint16),pack_weights(self.raw()),
                              np.zeros(OUTPUT_SHAPE,np.uint16),record,metrics_class=metrics)
            self.assertTrue(probe.failed)
            self.assertIn('observed',record)
            self.assertIn('metadata',record)

    def test_counts_and_exclusive_evidence(self):
        counts=self.counts()
        validate_counts(counts)
        for key in counts:
            bad=dict(counts)
            bad[key]+=1
            with self.assertRaises(RuntimeError): validate_counts(bad)
            del bad[key]
            with self.assertRaises(RuntimeError): validate_counts(bad)
        record=dict(input=np.zeros(INPUT_SHAPE,np.uint16),raw_weights=self.raw(),
                    metadata=np.zeros(METADATA_SHAPE,np.uint16))
        with tempfile.TemporaryDirectory() as directory:
            path=save_failure(directory,0,'test',record,'failure')
            with np.load(path) as saved:
                for key,value in record.items(): np.testing.assert_array_equal(saved[key],value)
            with self.assertRaises(FileExistsError): save_failure(directory,0,'test',record,'later')

    def test_missing_artifacts_and_bad_contracts(self):
        backend=Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError): PhaseAGather(directory,backend)
        backend.DefaultNPURuntime.load.assert_not_called()
        with self.assertRaises(ValueError): pack_weights(np.zeros(1,np.uint16))
        with self.assertRaises(ValueError): gather_oracle(np.zeros((400,8),np.uint16))
        with self.assertRaises(ValueError): frame_weights(self.raw(),-1)
        with self.assertRaises(ValueError): frame_input(-1,self.raw())


if __name__=='__main__':
    unittest.main()
