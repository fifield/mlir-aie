//===- aie4-mmac-fp8-cse-guard.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check workaround scoping after per-core and unified lowering: fp8
// staging disables MachineCSE; all dense staging disables the staging-load
// and indexed auto-increment combines. An ordinary core receives no flags.
//
// REQUIRES: peano
//
// RUN: %aiecc -v --no-unified --get-core-elfs --tmpdir %t.p.prj --output-dir %t.p.out %s 2>&1 | FileCheck %s
// RUN: %aiecc -v --unified --get-core-elfs --tmpdir %t.u.prj --output-dir %t.u.out %s 2>&1 | FileCheck %s
//
// CHECK-DAG: exec: {{.*}}/llc {{.*}}main_core_0_2.ll -O2 --march=aie4 --mcpu=aie4b -disable-machine-cse -aie4postlegalizergenericcombiner-disable-rule=combine_indexed_to_autoinc -aie4postlegalizercustomcombiner-disable-rule=combine_staging_upd_load,combine_indexed_to_autoinc --function-sections
// CHECK-DAG: exec: {{.*}}/llc {{.*}}main_core_1_2.ll -O2 --march=aie4 --mcpu=aie4b -aie4postlegalizergenericcombiner-disable-rule=combine_indexed_to_autoinc -aie4postlegalizercustomcombiner-disable-rule=combine_staging_upd_load,combine_indexed_to_autoinc --function-sections

// CHECK-DAG: exec: {{.*}}/llc {{.*}}main_core_2_2.ll -O2 --march=aie4 --mcpu=aie4b --function-sections

module {
  aie.device(npu3) {
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %t22 = aie.tile(2, 2)
    %a8 = aie.buffer(%t02) {sym_name = "a8"} : memref<64xi8>
    %c8 = aie.buffer(%t02) {sym_name = "c8"} : memref<64xf32>
    %ai = aie.buffer(%t12) {sym_name = "ai"} : memref<64xi8>
    %ci = aie.buffer(%t12) {sym_name = "ci"} : memref<64xi32>

    %plain = aie.buffer(%t22) {sym_name = "plain"} : memref<64xi32>

    // fp8: both workarounds.
    %core_0_2 = aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %araw = vector.load %a8[%c0] : memref<64xi8>, vector<64xi8>
      %a = vector.bitcast %araw : vector<64xi8> to vector<64xf8E4M3FN>
      %s = memref.load %a8[%c0] : memref<64xi8>
      %sb = arith.bitcast %s : i8 to f8E4M3FN
      %b = vector.broadcast %sb : f8E4M3FN to vector<64x64xf8E4M3FN>
      %r = aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> -> vector<64xf32> {
        %x = aievec.aie4.mmac fp8 %a : vector<64xf8E4M3FN> -> vector<64xf32>
        aievec.aie4.matrix_yield %x : vector<64xf32>
      }
      vector.store %r, %c8[%c0] : memref<64xf32>, vector<64xf32>
      aie.end
    }

    // int8: the staging-load workaround only.
    %core_1_2 = aie.core(%t12) {
      %c0 = arith.constant 0 : index
      %a = vector.load %ai[%c0] : memref<64xi8>, vector<64xi8>
      %s = memref.load %ai[%c0] : memref<64xi8>
      %b = vector.broadcast %s : i8 to vector<64x64xi8>
      %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
        %x = aievec.aie4.mmac int8 %a {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
        aievec.aie4.matrix_yield %x : vector<64xi32>
      }
      vector.store %r, %ci[%c0] : memref<64xi32>, vector<64xi32>
      aie.end
    }
    // Ordinary scalar arithmetic: neither workaround.
    %core_2_2 = aie.core(%t22) {
      %c0 = arith.constant 0 : index
      %v = memref.load %plain[%c0] : memref<64xi32>
      %r = arith.addi %v, %v : i32
      memref.store %r, %plain[%c0] : memref<64xi32>
      aie.end
    }
  }
}
