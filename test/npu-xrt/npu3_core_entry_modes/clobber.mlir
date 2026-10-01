//===- clobber.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Same core as entry.mlir, but a linked call sets crVecRnd = crRnd = 0
// (floor) again after the entry writes. The host expects every inexact
// result to floor: the modes are live, and later writes by calls are
// clobbers that the compiler does not undo (D19).
module {
  aie.device(npu3) {
    func.func private @get_vec_rnd() -> i32 attributes {link_with = "modes.o"}
    func.func private @get_rnd() -> i32 attributes {link_with = "modes.o"}
    func.func private @set_modes(i32, i32) attributes {link_with = "modes.o"}
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_0_2 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<64xf32>>
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<128xf32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c96 = arith.constant 96 : index
      %c97 = arith.constant 97 : index
      %pad = arith.constant 0.0 : f32
      %bias = arith.constant dense<895.001> : vector<32xf32>
      %vr = func.call @get_vec_rnd() : () -> i32
      %r = func.call @get_rnd() : () -> i32
      // Clobber both controls after entry, as an external call may.
      %z = arith.constant 0 : i32
      func.call @set_modes(%z, %z) : (i32, i32) -> ()
      %in = aie.objectfifo.acquire @in(Consume, 1) : memref<64xf32>
      %out = aie.objectfifo.acquire @out(Produce, 1) : memref<128xf32>
      // Vector float add (VADD.f, rounds with crVecRnd).
      %x = vector.transfer_read %in[%c0], %pad {in_bounds = [true]} : memref<64xf32>, vector<32xf32>
      %s = arith.addf %x, %bias : vector<32xf32>
      vector.transfer_write %s, %out[%c0] {in_bounds = [true]} : vector<32xf32>, memref<128xf32>
      // Scalar f32 -> bf16 (VCONV.bf16.fp32, rounds with crRnd), then a
      // scalar bf16 add of the result.
      scf.for %i = %c0 to %c32 step %c1 {
        %j = arith.addi %i, %c32 : index
        %f = memref.load %in[%j] : memref<64xf32>
        %h = arith.truncf %f : f32 to bf16
        %g = arith.extf %h : bf16 to f32
        %k = arith.addi %i, %c64 : index
        memref.store %g, %out[%k] : memref<128xf32>
        // Scalar bf16 add: an f32 add, then VCONV (rounds with crRnd).
        %bb = arith.constant 5.859375e-03 : bf16
        %sb = arith.addf %h, %bb : bf16
        %sg = arith.extf %sb : bf16 to f32
        memref.store %sg, %out[%j] : memref<128xf32>
      }
      %vrf = arith.sitofp %vr : i32 to f32
      %rf = arith.sitofp %r : i32 to f32
      memref.store %vrf, %out[%c96] : memref<128xf32>
      memref.store %rf, %out[%c97] : memref<128xf32>
      aie.objectfifo.release @in(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
    aie.runtime_sequence @sequence(%a : memref<64xf32>, %c : memref<128xf32>) {
      %t1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%c : memref<128xf32> offset = 0 len = 128)
        aie.end
      } {issue_token = true}
      %t0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<64xf32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
