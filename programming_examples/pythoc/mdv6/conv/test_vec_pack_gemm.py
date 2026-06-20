#!/usr/bin/env python3
"""Offline bit-exact check of the vectorized GEMM (1x1) pack/unpack helpers
against the old per-slot reference loops. No NPU required.

The merged GEMM dispatch (_run_gemm_oc_blocked_merged / _kblocked / _pair)
slices input_flat (M, IC) into tile_m-row slots packed into total_slots, pads
trailing slots with slot-0, runs the kernel, then reassembles only the active
pixels back into output_flat (M, out_ch).

Covers:
  - M divisible by pixels_per_call (clean, full active slots)
  - M not divisible (partial last active slot + slot-0 padded trailing slots)
  - multi-batch (M > pixels_per_call so the batch loop runs >1)
  - K-blocked variant (uses the same input/output packing -> same helpers)
  - pair path (two outputs share one input BO -> two reassembles)
"""
import os, sys
import numpy as np
import torch

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)
import importlib.util
spec = importlib.util.spec_from_file_location(
    'ett', os.path.join(_base, '_full_model_helpers', 'elan_test_tiled.py'))
ett = importlib.util.module_from_spec(spec); spec.loader.exec_module(ett)

bf16_to_uint16 = ett.bf16_to_uint16
uint16_to_bf16 = ett.uint16_to_bf16
pack_gemm_input_batch_u16 = ett.pack_gemm_input_batch_u16
reassemble_gemm_output = ett.reassemble_gemm_output

N_CORES = 32


def ref_pack(input_flat, batch_start, batch_end, total_slots, tile_m, input_size):
    """Old per-slot input packing loop (verbatim from run_tiled_mc.py)."""
    batch_pixels = batch_end - batch_start
    host_in = np.zeros(total_slots * input_size, dtype=np.uint16)
    n_active_slots = (batch_pixels + tile_m - 1) // tile_m
    for s in range(n_active_slots):
        pix_start = batch_start + s * tile_m
        pix_end = min(pix_start + tile_m, batch_end)
        active_u16 = bf16_to_uint16(input_flat[pix_start:pix_end].flatten())
        dst = s * input_size
        host_in[dst:dst + len(active_u16)] = active_u16
    slot0 = host_in[:input_size]
    for s in range(n_active_slots, total_slots):
        host_in[s * input_size:(s + 1) * input_size] = slot0
    return host_in


def ref_reassemble(out_data, batch_start, batch_end, total_slots, tile_m,
                   out_ch, output_size, output_flat):
    """Old per-slot output reassembly loop (verbatim from run_tiled_mc.py)."""
    batch_pixels = batch_end - batch_start
    n_active_slots = (batch_pixels + tile_m - 1) // tile_m
    for s in range(min(n_active_slots, total_slots)):
        pix_start = batch_start + s * tile_m
        pix_end = min(pix_start + tile_m, batch_end)
        if pix_start >= batch_end:
            break
        n_pix = pix_end - pix_start
        start = s * output_size
        tile_out = uint16_to_bf16(out_data[start:start + n_pix * out_ch])
        tile_out = tile_out.reshape(n_pix, out_ch)
        output_flat[pix_start:pix_end, :] = tile_out.to(torch.bfloat16)


def run_case(name, M, IC, out_ch, tile_m, ppc, n_outputs=1):
    torch.manual_seed(hash(name) & 0xffff)
    input_flat = torch.randn(M, IC, dtype=torch.bfloat16)
    input_flat_u16 = bf16_to_uint16(input_flat.contiguous())

    input_size = tile_m * IC
    output_size = tile_m * out_ch
    pixels_per_call = N_CORES * tile_m * ppc
    total_slots = N_CORES * ppc

    n_batches = 0
    for batch_start in range(0, M, pixels_per_call):
        batch_end = min(batch_start + pixels_per_call, M)
        batch_pixels = batch_end - batch_start
        n_batches += 1

        # --- input pack: new vs old, bitwise uint16 ---
        ref_buf = ref_pack(input_flat, batch_start, batch_end,
                           total_slots, tile_m, input_size)
        vec_buf = pack_gemm_input_batch_u16(
            input_flat_u16, batch_start, batch_pixels,
            total_slots, tile_m, input_size)
        assert vec_buf.shape == ref_buf.shape, \
            f"{name}: pack shape {vec_buf.shape} != {ref_buf.shape}"
        assert np.array_equal(vec_buf, ref_buf), \
            f"{name}: input pack mismatch at batch_start {batch_start}"

        # --- output reassembly: new vs old, bitwise bf16 ---
        for _o in range(n_outputs):
            out_data = np.random.randint(
                0, 65536, size=total_slots * output_size, dtype=np.uint16)
            ref_out = torch.zeros(M, out_ch, dtype=torch.bfloat16)
            vec_out = torch.zeros(M, out_ch, dtype=torch.bfloat16)
            ref_reassemble(out_data, batch_start, batch_end, total_slots,
                          tile_m, out_ch, output_size, ref_out)
            reassemble_gemm_output(out_data, batch_start, batch_pixels,
                                   total_slots, tile_m, out_ch, output_size,
                                   vec_out)
            ref_u = bf16_to_uint16(ref_out.contiguous())
            vec_u = bf16_to_uint16(vec_out.contiguous())
            assert np.array_equal(ref_u, vec_u), \
                f"{name}: output reassembly mismatch at batch_start {batch_start}"

    print(f"  [PASS] {name}: M={M} IC={IC} OC={out_ch} tile_m={tile_m} "
          f"ppc={ppc} -> {n_batches} batch(es), {total_slots} slots, "
          f"{n_outputs} output(s)")


def main():
    print("Vectorized GEMM (1x1) pack/unpack bit-exact self-check:")
    # name, M, IC, out_ch, tile_m, ppc[, n_outputs]
    cases = [
        # M divisible by pixels_per_call: M == N_CORES*tile_m*ppc
        ("gemm_divisible",     N_CORES * 8 * 1, 64, 64, 8, 1),
        # M not divisible: partial last active slot + slot-0 padded trailing
        ("gemm_partial_slot",  N_CORES * 8 * 1 - 3, 64, 64, 8, 1),
        # M smaller than one full slot count (few active slots)
        ("gemm_few_slots",     5 * 8 + 2, 128, 32, 8, 1),
        # multi-batch: M > pixels_per_call (batch loop runs >1), partial tail
        ("gemm_multibatch",    N_CORES * 8 * 1 * 3 + 17, 64, 96, 8, 1),
        # multi-batch with ppc>1
        ("gemm_ppc2_multi",    N_CORES * 4 * 2 * 2 + 5, 256, 64, 4, 2),
        # K-blocked variant: identical input/output packing, different shape
        ("gemm_kblocked",      N_CORES * 16 * 1 + 100, 512, 128, 16, 1),
        # pair path: two outputs share one input pack
        ("gemm_pair",          N_CORES * 8 * 1 + 40, 64, 64, 8, 1, 2),
        # pair path multi-batch
        ("gemm_pair_multi",    N_CORES * 8 * 1 * 2 + 11, 96, 48, 8, 1, 2),
    ]
    for c in cases:
        run_case(*c)
    print("ALL GEMM CASES BIT-EXACT.")


if __name__ == "__main__":
    main()
