//===- kernel.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// INT8 GEMV kernel (M=1) with chunked L1 inputs and cascade reduction.
// See README.md for configuration and host layout.
//
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;
using namespace me_primitive;

// Problem geometry (npu3, up to 3 cols x N_ROWS compute rows).
// Total problem: M=1, K=K_TOTAL (default 4096), N=cols*64, INT8 x INT8 ->
// INT32. K is split evenly across N_ROWS compute rows, then tiled into tile_K
// chunks, so K_TOTAL must be a multiple of N_ROWS*tile_K.  Pass -DK_TOTAL=<K>
// and -DN_ROWS=<rows> (from the Makefile K/ROWS variables) to match the
// --ktotal/--rows passed to gemv_placed.py.
#ifndef K_TOTAL
#define K_TOTAL 4096
#endif
#ifndef N_ROWS
#define N_ROWS 4
#endif
constexpr unsigned tile_K = 64;
constexpr unsigned tile_N = 64;
constexpr unsigned N_rows = N_ROWS; // active COMPUTE_ROWS count
static_assert(N_rows >= 1 && N_rows <= 4, "N_ROWS must be 1..4");
static_assert(K_TOTAL % (N_rows * tile_K) == 0,
              "K_TOTAL must be a multiple of N_rows*tile_K");
constexpr unsigned K_per_row = K_TOTAL / N_rows;
constexpr unsigned K_tiles_per_row = K_per_row / tile_K;
constexpr unsigned N_per_col = tile_N;

// K-tiles per FIFO item; zero selects the whole row. Must match --kchunk
// in gemv_placed.py (the Makefile passes KCHUNK to both).
#ifndef K_CHUNK_TILES_RAW
#define K_CHUNK_TILES_RAW 0
#endif
constexpr unsigned K_chunk_tiles =
    (K_CHUNK_TILES_RAW == 0) ? K_tiles_per_row : K_CHUNK_TILES_RAW;
static_assert(K_tiles_per_row % K_chunk_tiles == 0,
              "K_tiles_per_row must be a multiple of K_chunk_tiles (KCHUNK)");
constexpr unsigned num_chunks = K_tiles_per_row / K_chunk_tiles;

// Per-K-tile staging count: half from each weight pointer.
// Each pointer supplies 16 v128int8 loads, together filling the 64x64 matrix.
constexpr unsigned regs_per_stream = 16;
constexpr unsigned reg_offset = 16;

// Per-chunk byte sizes.  One K-chunk's AW_lo sub-buffer is
// [A_chunk (K_chunk_tiles*tile_K bytes) | W_lo_chunk (same K range, N cols
// 0..31)]; W_hi_chunk carries the same K range's N cols 32..63.  The host
// layout interleaves chunks this way per row (see test.cpp's pack_*_byte),
// so each per-chunk object FIFO item is a self-contained, independently
// DMA-able unit.
constexpr unsigned chunk_A_bytes = K_chunk_tiles * tile_K;
constexpr unsigned chunk_Wlo_bytes = K_chunk_tiles * tile_K * tile_N / 2;

// Reduce K_chunk_tiles matrices into 64 INT32 partial sums for this chunk.
// chunk_out receives the raw v64acc32 bit pattern.
static inline void gemv_reduce_chunk(int8_t *restrict A, int8_t *restrict W_lo,
                                     int8_t *restrict W_hi,
                                     int32_t *restrict chunk_out) {
  v64int8 *pA = reinterpret_cast<v64int8 *>(A);
  v128int8 *pW_lo = reinterpret_cast<v128int8 *>(W_lo);
  v128int8 *pW_hi = reinterpret_cast<v128int8 *>(W_hi);

  v64acc32 acc = undef_v64acc32();
  uint2_t mul_mode = MMAC_INT8;
  bool zero_acc = 1; // first iteration overwrites the accumulator

// Prologue: stage tile 0 of this chunk.
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

  // Pipelined body: compute tile k while staging tile k+1 (within chunk).
  // chess_loop_range is a trip-count contract, not a hint. Its bounds must
  // match K_chunk_tiles - 1, including zero, to keep loads within the chunk.
  for (unsigned k = 0; k < K_chunk_tiles - 1; k++)
    chess_prepare_for_pipelining chess_loop_range(K_chunk_tiles - 1,
                                                  K_chunk_tiles - 1) {
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

  // Epilogue: final MMAC for the chunk's last tile.
  staging_to_matrix_m64x64int8();
  v64int8 Xbuff = *pA++;
  acc = mac_conf(Xbuff, 1, 1, acc, zero_acc);

  v64acc32 *pOut = reinterpret_cast<v64acc32 *>(chunk_out);
  *pOut = acc;
}

// vec_size = 16 INT32 lanes per cascade transfer, so a 64-element
// partial requires 4 cascade chunks.
constexpr unsigned cascade_vec = 16;
constexpr unsigned cascade_chunks = N_per_col / cascade_vec;

// Cascade roots (topmost active row): forward partial as the column root.
static inline void cascade_root_send(int32_t *restrict partial) {
  v16int32 *pP = reinterpret_cast<v16int32 *>(partial);
#pragma unroll
  for (unsigned i = 0; i < cascade_chunks; i++)
    put_mcd(pP[i]);
}

// Cascade middle (interior active rows): receive + add + forward.
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

// No-cascade write (used only when N_ROWS==1: this core is both root and
// sink, with no upstream/downstream neighbor to exchange cascade data with).
static inline void write_partial(int32_t *restrict partial,
                                 int32_t *restrict C) {
  v16int32 *pP = reinterpret_cast<v16int32 *>(partial);
  v16int32 *pC = reinterpret_cast<v16int32 *>(C);
#pragma unroll
  for (unsigned i = 0; i < cascade_chunks; i++)
    pC[i] = pP[i];
}

extern "C" {

// Zero the per-core running partial-sum buffer before accumulating chunks.
void gemv_zero_partial(int32_t *restrict partial) {
#pragma unroll
  for (unsigned i = 0; i < N_per_col; i++)
    partial[i] = 0;
}

// Accumulate one K-chunk's worth of MACs into the running `partial` buffer.
// AW_lo_chunk: [A_chunk (chunk_A_bytes) | W_lo_chunk (chunk_Wlo_bytes)].
// Called num_chunks times per core per GEMV iteration.
void gemv_chunk_accumulate(int8_t *restrict AW_lo_chunk,
                           int8_t *restrict W_hi_chunk,
                           int32_t *restrict partial) {
  int8_t *A_chunk = AW_lo_chunk;
  int8_t *Wlo_chunk = AW_lo_chunk + chunk_A_bytes;
  alignas(32) int32_t chunk_out[N_per_col];
  gemv_reduce_chunk(A_chunk, Wlo_chunk, W_hi_chunk, chunk_out);

  v16int32 *pP = reinterpret_cast<v16int32 *>(partial);
  v16int32 *pO = reinterpret_cast<v16int32 *>(chunk_out);
#pragma unroll
  for (unsigned i = 0; i < cascade_chunks; i++)
    pP[i] = add(pP[i], pO[i]);
}

// Finish-and-cascade entries: call once per core per GEMV iteration, after
// all num_chunks calls to gemv_chunk_accumulate have filled `partial`.
void gemv_finish_send(int32_t *restrict partial) { cascade_root_send(partial); }

void gemv_finish_pass(int32_t *restrict partial) { cascade_middle(partial); }

void gemv_finish_sink(int32_t *restrict partial, int32_t *restrict C) {
  cascade_sink_write(partial, C);
}

// N_ROWS==1 entry: this core is both root and sink with no cascade peer.
void gemv_finish_solo(int32_t *restrict partial, int32_t *restrict C) {
  write_partial(partial, C);
}

} // extern "C"
