//===- kernel.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// High-performance GEMV kernel (M=1) with multi-row cascade reduction.
// Ported from pacifica-apps/kernels/llama2/llama_4096_512/kernel.cc
// (gemv_reduce_K + cascade_north_south), with FFN/SiLU/quantize stripped
// and the weight source switched from stream intrinsics to L1 pointers
// so the design fits the standard aie4 example object_fifo flow.
//
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;
using namespace me_primitive;

// Problem geometry (npu3, up to 3 cols x 4 compute rows = 12 cores).
// Total problem: M=1, K=K_TOTAL (default 4096), N=cols*64, INT8 x INT8 -> INT32.
// K is split evenly across the 4 compute rows, then tiled into tile_K chunks,
// so K_TOTAL must be a multiple of N_rows*tile_K (=256).  Pass -DK_TOTAL=<K>
// (from the Makefile `K` variable) to build a different contraction length;
// this must match the --ktotal passed to gemv_placed.py.
#ifndef K_TOTAL
#define K_TOTAL 4096
#endif
constexpr unsigned tile_K          = 64;
constexpr unsigned tile_N          = 64;
constexpr unsigned N_rows          = 4;                   // COMPUTE_ROWS count
static_assert(K_TOTAL % (N_rows * tile_K) == 0,
              "K_TOTAL must be a multiple of N_rows*tile_K (=256)");
constexpr unsigned K_per_row       = K_TOTAL / N_rows;    // 1024 @ K=4096
constexpr unsigned K_tiles_per_row = K_per_row / tile_K;  // 16   @ K=4096
constexpr unsigned N_per_col       = tile_N;              // 64
constexpr unsigned A_slice_bytes   = K_per_row;           // 1024 @ K=4096

// Per-K-tile staging count: half from each weight pointer.
// v128int8 carries 128 INT8 (= 32 cols of 4-byte words). Two weight
// pointers feed 16 v128 each per K-tile, exactly matching the source
// design's two-stream layout and filling the 64x64 staging matrix.
constexpr unsigned regs_per_stream = 16;
constexpr unsigned reg_offset      = 16;

// gemv_reduce_K: reduce along K (M=1, tile_N=64), accumulate into a
// single v64acc32 across K_tiles_per_row 64x64 INT8 tiles, write the
// resulting 64 INT32 column partial to the local result buffer.
//
// Weights arrive on TWO L1 buffers fed by two parallel shim MM2S
// channels (mirroring the source design's two weight streams).  W_lo
// carries the "low half" of every K-tile (slots 0..15 -> N cols 0..31)
// and W_hi carries the "high half" (slots 16..31 -> N cols 32..63).
// Each buffer is K_tiles_per_row * regs_per_stream * sizeof(v128int8)
// = 32 KB per core.  The pointers advance side by side per K-tile.
static inline void gemv_reduce_K(int8_t *restrict A,
                                 int8_t *restrict W_lo,
                                 int8_t *restrict W_hi,
                                 int32_t *restrict R) {
  v64int8  *pA   = reinterpret_cast<v64int8  *>(A);
  v128int8 *pW_lo = reinterpret_cast<v128int8 *>(W_lo);
  v128int8 *pW_hi = reinterpret_cast<v128int8 *>(W_hi);

  v64acc32 acc      = undef_v64acc32();
  uint2_t  mul_mode = MMAC_INT8;
  bool     zero_acc = 1;  // first iteration overwrites the accumulator

  // Prologue: stage K-tile 0.
  #pragma unroll
  for (size_t reg_idx = 0; reg_idx < regs_per_stream; reg_idx++)
    chess_prepare_for_pipelining {
      v128int8 lo = pW_lo[reg_idx];
      v128int8 hi = pW_hi[reg_idx];
      staging_m64x64int8 = upd(staging_m64x64int8, lo, reg_idx, mul_mode);
      staging_m64x64int8 =
          upd(staging_m64x64int8, hi, reg_idx + reg_offset, mul_mode);
    }
  pW_lo += regs_per_stream;
  pW_hi += regs_per_stream;

  // Pipelined body: compute K-tile k while staging K-tile k+1.
  for (unsigned k = 0; k < K_tiles_per_row - 1; k++)
    chess_prepare_for_pipelining chess_loop_range(3, 3) {
      staging_to_matrix_m64x64int8();

      v64int8 Xbuff = *pA++;
      acc = mac_conf(Xbuff, 1, 1, acc, zero_acc);
      zero_acc = 0;

      #pragma unroll
      for (size_t reg_idx = 0; reg_idx < regs_per_stream; reg_idx++)
        chess_prepare_for_pipelining {
          v128int8 lo = pW_lo[reg_idx];
          v128int8 hi = pW_hi[reg_idx];
          staging_m64x64int8 = upd(staging_m64x64int8, lo, reg_idx, mul_mode);
          staging_m64x64int8 =
              upd(staging_m64x64int8, hi, reg_idx + reg_offset, mul_mode);
        }
      pW_lo += regs_per_stream;
      pW_hi += regs_per_stream;
    }

  // Epilogue: final MMAC for K-tile (K_tiles_per_row - 1).
  staging_to_matrix_m64x64int8();
  v64int8 Xbuff = *pA++;
  acc = mac_conf(Xbuff, 1, 1, acc, zero_acc);

  v64acc32 *pOut = reinterpret_cast<v64acc32 *>(R);
  *pOut = acc;
}

// vec_size = 16 INT32 lanes per cascade transfer, so a 64-element
// partial requires 4 cascade chunks.
constexpr unsigned cascade_vec   = 16;
constexpr unsigned cascade_chunks = N_per_col / cascade_vec;  // 4

// Cascade roots (hardware row 5): forward partial as the column root.
static inline void cascade_root_send(int32_t *restrict partial) {
  v16int32 *pP = reinterpret_cast<v16int32 *>(partial);
  #pragma unroll
  for (unsigned i = 0; i < cascade_chunks; i++)
    put_mcd(pP[i]);
}

// Cascade middle (hardware rows 3, 4): receive + add + forward.
static inline void cascade_middle(int32_t *restrict partial) {
  v16int32 *pP = reinterpret_cast<v16int32 *>(partial);
  #pragma unroll
  for (unsigned i = 0; i < cascade_chunks; i++) {
    v16int32 in = get_scd_v16int32();
    v16int32 sum = add(in, pP[i]);
    put_mcd(sum);
  }
}

// Cascade sink (hardware row 2): receive + add + write to column output C.
static inline void cascade_sink_write(int32_t *restrict partial,
                                      int32_t *restrict C) {
  v16int32 *pP = reinterpret_cast<v16int32 *>(partial);
  v16int32 *pC = reinterpret_cast<v16int32 *>(C);
  #pragma unroll
  for (unsigned i = 0; i < cascade_chunks; i++) {
    v16int32 in = get_scd_v16int32();
    pC[i] = add(in, pP[i]);
  }
}

extern "C" {

// A compute tile has only 2 DMA input channels, so the design packs the
// per-row activation slice (1 KB) at the FRONT of this core's W_lo
// buffer.  Each core consumes two FIFOs total:
//   AW_lo: [A_slice 1024 B][W_lo half-slab 32768 B] = 33792 B
//   W_hi : 32768 B (high half of every K-tile, N cols 32..63)
// AW_lo's lo-half data starts at offset A_slice_bytes inside the buffer.

// Cascade root entry: GEMV + cascade send.
void gemv_core_send(int8_t *restrict AW_lo, int8_t *restrict W_hi) {
  int8_t *A    = AW_lo;
  int8_t *W_lo = AW_lo + A_slice_bytes;
  alignas(32) int32_t partial[N_per_col];
  gemv_reduce_K(A, W_lo, W_hi, partial);
  cascade_root_send(partial);
}

// Cascade middle entry: GEMV + cascade pass-through.
void gemv_core_pass(int8_t *restrict AW_lo, int8_t *restrict W_hi) {
  int8_t *A    = AW_lo;
  int8_t *W_lo = AW_lo + A_slice_bytes;
  alignas(32) int32_t partial[N_per_col];
  gemv_reduce_K(A, W_lo, W_hi, partial);
  cascade_middle(partial);
}

// Cascade sink entry: GEMV + cascade reduce + write C.
// C: 64 INT32 = 256 bytes (this column's final partial output).
void gemv_core_sink(int8_t *restrict AW_lo, int8_t *restrict W_hi,
                    int32_t *restrict C) {
  int8_t *A    = AW_lo;
  int8_t *W_lo = AW_lo + A_slice_bytes;
  alignas(32) int32_t partial[N_per_col];
  gemv_reduce_K(A, W_lo, W_hi, partial);
  cascade_sink_write(partial, C);
}

}  // extern "C"
