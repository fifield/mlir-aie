// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -split-input-file -verify-diagnostics

// Maskable operations without a tail-safe AIE4 lowering are diagnosed.

func.func @masked_axpy(%a: vector<32xf32>, %b: f32, %acc: vector<32xf32>, %n: index) -> vector<32xf32> {
  %mask = vector.create_mask %n : vector<32xi1>
  // expected-error @+1 {{AIE4: masked 'vector.outerproduct' has no tail-safe lowering}}
  %r = vector.mask %mask { vector.outerproduct %a, %b, %acc {kind = #vector.kind<add>} : vector<32xf32>, f32 } : vector<32xi1> -> vector<32xf32>
  return %r : vector<32xf32>
}

// -----

func.func @gather(%m: memref<1000xf32>, %idx: vector<32xi32>, %mask: vector<32xi1>, %pt: vector<32xf32>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{AIE4: 'vector.gather' has no tail-safe lowering}}
  %r = vector.gather %m[%c0] [%idx], %mask, %pt : memref<1000xf32>, vector<32xi32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
  return %r : vector<32xf32>
}

// -----

func.func @read_transposed(%m: memref<32x32xf32>, %n: index) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %n : vector<32xi1>
  // expected-error @+1 {{AIE4: masked vector.transfer_read requires a minor identity permutation map}}
  %v = vector.mask %mask { vector.transfer_read %m[%c0, %c0], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<32x32xf32>, vector<32xf32> } : vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----

func.func @read_128(%m: memref<256xi8>, %i: index, %n: index) -> vector<128xi8> {
  %pad = arith.constant 0 : i8
  %mask = vector.create_mask %n : vector<128xi1>
  // expected-error @+1 {{AIE4: masks must be fixed 1-D with at most 64 lanes}}
  %v = vector.mask %mask { vector.transfer_read %m[%i], %pad {in_bounds = [true]} : memref<256xi8>, vector<128xi8> } : vector<128xi1> -> vector<128xi8>
  return %v : vector<128xi8>
}

// -----

func.func @read_tensor(%t: tensor<1000xf32>, %i: index, %n: index) -> vector<32xf32> {
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %n : vector<32xi1>
  // expected-error @+1 {{AIE4: masked vector.transfer_read on a tensor has no tail-safe lowering}}
  %v = vector.mask %mask { vector.transfer_read %t[%i], %pad {in_bounds = [true]} : tensor<1000xf32>, vector<32xf32> } : vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----

func.func @maskedload_2d(%m: memref<8x32xf32>, %mask: vector<2x32xi1>, %pt: vector<2x32xf32>) -> vector<2x32xf32> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{AIE4: vector.maskedload requires a fixed 1-D vector of at most 64 lanes}}
  %v = vector.maskedload %m[%c0, %c0], %mask, %pt : memref<8x32xf32>, vector<2x32xi1>, vector<2x32xf32> into vector<2x32xf32>
  return %v : vector<2x32xf32>
}

// -----

func.func @scatter(%m: memref<1000xf32>, %idx: vector<32xi32>, %mask: vector<32xi1>, %v: vector<32xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{AIE4: 'vector.scatter' has no tail-safe lowering}}
  vector.scatter %m[%c0] [%idx], %mask, %v : memref<1000xf32>, vector<32xi32>, vector<32xi1>, vector<32xf32>
  return
}

// -----

// A 2-D mask is diagnosed before canonicalization can flatten the masked
// write inside its region.
func.func @write_2d(%m: memref<4x32xf32>, %v: vector<2x32xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %c2, %n : vector<2x32xi1>
  // expected-error @+1 {{AIE4: masks must be fixed 1-D with at most 64 lanes}}
  vector.mask %mask { vector.transfer_write %v, %m[%c0, %c0] {in_bounds = [true, true]} : vector<2x32xf32>, memref<4x32xf32> } : vector<2x32xi1>
  return
}

// -----

// minnumf/maxnumf need a NaN identity, which Peano cannot parse as a constant.
func.func @reduce_maxnumf(%v: vector<32xf32>, %n: index) -> f32 {
  %mask = vector.create_mask %n : vector<32xi1>
  // expected-error @+1 {{AIE4: masked maxnumf reduction has no supported identity for 'f32'}}
  %r = vector.mask %mask { vector.reduction <maxnumf>, %v : vector<32xf32> into f32 } : vector<32xi1> -> f32
  return %r : f32
}

// -----

func.func @reduce_minnumf(%v: vector<32xf32>, %n: index) -> f32 {
  %mask = vector.create_mask %n : vector<32xi1>
  // expected-error @+1 {{AIE4: masked minnumf reduction has no supported identity for 'f32'}}
  %r = vector.mask %mask { vector.reduction <minnumf>, %v : vector<32xf32> into f32 } : vector<32xi1> -> f32
  return %r : f32
}

// -----

func.func @reduce_index(%v: vector<32xindex>, %n: index) -> index {
  %mask = vector.create_mask %n : vector<32xi1>
  // expected-error @+1 {{AIE4: masked add reduction has no supported identity for 'index'}}
  %r = vector.mask %mask { vector.reduction <add>, %v : vector<32xindex> into index } : vector<32xi1> -> index
  return %r : index
}
