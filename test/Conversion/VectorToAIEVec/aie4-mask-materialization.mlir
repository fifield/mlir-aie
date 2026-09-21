// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" --convert-aievec-to-llvm="aie-target=aie4b" --convert-vector-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | aie-translate --mlir-to-llvmir | FileCheck %s

// Exercise both mask directions through LLVM translation, including unused
// carrier bits, generic mask producers/consumers, and scalar conditions.

// CHECK-LABEL: define <16 x i1> @compare_16(
// CHECK: %[[CMP:[0-9]+]] = call { <16 x i32>, i32 } @llvm.aie4.vadd32.512(
// CHECK: %[[MASK:[0-9]+]] = extractvalue { <16 x i32>, i32 } %[[CMP]], 1
// CHECK: %[[LOW:[0-9]+]] = trunc i32 %[[MASK]] to i16
// CHECK: %[[V:[0-9]+]] = insertelement <1 x i16> poison, i16 %[[LOW]], i32 0
// CHECK: %[[BOOL:[0-9]+]] = bitcast <1 x i16> %[[V]] to <16 x i1>
// CHECK: ret <16 x i1> %[[BOOL]]
func.func @compare_16(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi1> {
  %m = arith.cmpi slt, %a, %b : vector<16xi32>
  return %m : vector<16xi1>
}

// CHECK-LABEL: define <16 x i32> @select_16(
// CHECK-SAME: <16 x i1> %[[M:[0-9]+]], <16 x i32> %[[T:[0-9]+]], <16 x i32> %[[F:[0-9]+]])
// CHECK: %[[V:[0-9]+]] = bitcast <16 x i1> %[[M]] to <1 x i16>
// CHECK: %[[LOW:[0-9]+]] = extractelement <1 x i16> %[[V]], i64 0
// CHECK: %[[MASK:[0-9]+]] = zext i16 %[[LOW]] to i32
// CHECK: call <16 x i32> @llvm.aie4.vadd32.mask.512(<16 x i32> %[[F]], <16 x i32> %[[T]], i32 0, i32 2, i32 %[[MASK]], i32 0)
func.func @select_16(%m: vector<16xi1>, %t: vector<16xi32>, %f: vector<16xi32>) -> vector<16xi32> {
  %r = arith.select %m, %t, %f : vector<16xi1>, vector<16xi32>
  return %r : vector<16xi32>
}

// CHECK-LABEL: define <32 x i1> @compare_32(
// CHECK: %[[CMP:[0-9]+]] = call { <32 x i16>, {{.*}} } @llvm.aie4.vadd16.512(
// CHECK: %[[MASK:[0-9]+]] = extractvalue {{.*}} %[[CMP]], 1
// CHECK: %[[V:[0-9]+]] = insertelement <1 x i32> poison, i32 %[[MASK]], i32 0
// CHECK: %[[BOOL:[0-9]+]] = bitcast <1 x i32> %[[V]] to <32 x i1>
// CHECK: ret <32 x i1> %[[BOOL]]
func.func @compare_32(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi1> {
  %m = arith.cmpi slt, %a, %b : vector<32xi16>
  return %m : vector<32xi1>
}

// CHECK-LABEL: define <32 x i16> @select_32(
// CHECK-SAME: <32 x i1> %[[M:[0-9]+]], <32 x i16> %[[T:[0-9]+]], <32 x i16> %[[F:[0-9]+]])
// CHECK: %[[V:[0-9]+]] = bitcast <32 x i1> %[[M]] to <1 x i32>
// CHECK: %[[MASK:[0-9]+]] = extractelement <1 x i32> %[[V]], i64 0
// CHECK: call <32 x i16> @llvm.aie4.vadd16.mask.512(<32 x i16> %[[F]], <32 x i16> %[[T]], i32 0, i32 2, i32 %[[MASK]], i32 0)
func.func @select_32(%m: vector<32xi1>, %t: vector<32xi16>, %f: vector<32xi16>) -> vector<32xi16> {
  %r = arith.select %m, %t, %f : vector<32xi1>, vector<32xi16>
  return %r : vector<32xi16>
}

// CHECK-LABEL: define <64 x i1> @compare_64(
// CHECK: %[[CMP:[0-9]+]] = call { <64 x i8>, {{.*}} } @llvm.aie4.vadd8.512(
// CHECK: %[[MASK:[0-9]+]] = extractvalue {{.*}} %[[CMP]], 1
// CHECK: %[[SCALAR:[0-9]+]] = bitcast <2 x i32> %[[MASK]] to i64
// CHECK: %[[V:[0-9]+]] = insertelement <1 x i64> poison, i64 %[[SCALAR]], i32 0
// CHECK: %[[BOOL:[0-9]+]] = bitcast <1 x i64> %[[V]] to <64 x i1>
// CHECK: ret <64 x i1> %[[BOOL]]
func.func @compare_64(%a: vector<64xi8>, %b: vector<64xi8>) -> vector<64xi1> {
  %m = arith.cmpi slt, %a, %b : vector<64xi8>
  return %m : vector<64xi1>
}

// CHECK-LABEL: define <64 x i8> @select_64(
// CHECK-SAME: <64 x i1> %[[M:[0-9]+]], <64 x i8> %[[T:[0-9]+]], <64 x i8> %[[F:[0-9]+]])
// CHECK: %[[V:[0-9]+]] = bitcast <64 x i1> %[[M]] to <1 x i64>
// CHECK: %[[MASK:[0-9]+]] = extractelement <1 x i64> %[[V]], i64 0
// CHECK: %[[PAIR:[0-9]+]] = bitcast i64 %[[MASK]] to <2 x i32>
// CHECK: call <64 x i8> @llvm.aie4.vadd8.mask.512(<64 x i8> %[[F]], <64 x i8> %[[T]], i32 0, i32 2, <2 x i32> %[[PAIR]], i32 0)
func.func @select_64(%m: vector<64xi1>, %t: vector<64xi8>, %f: vector<64xi8>) -> vector<64xi8> {
  %r = arith.select %m, %t, %f : vector<64xi1>, vector<64xi8>
  return %r : vector<64xi8>
}

// CHECK-LABEL: define i1 @extract_last_lane(
// CHECK: %[[MASK:[0-9]+]] = extractvalue { <16 x i32>, i32 } {{.*}}, 1
// CHECK: %[[LOW:[0-9]+]] = trunc i32 %[[MASK]] to i16
// CHECK: %[[WIDE:[0-9]+]] = zext i16 %[[LOW]] to i32
// CHECK: %[[SHIFT:[0-9]+]] = lshr i32 %[[WIDE]], 15
// CHECK: %[[BIT:[0-9]+]] = trunc i32 %[[SHIFT]] to i1
// CHECK: ret i1 %[[BIT]]
func.func @extract_last_lane(%a: vector<16xi32>, %b: vector<16xi32>) -> i1 {
  %m = arith.cmpi slt, %a, %b : vector<16xi32>
  %r = vector.extract %m[15] : i1 from vector<16xi1>
  return %r : i1
}

// CHECK-LABEL: define <16 x i32> @equal_select(
// CHECK: %[[EQ:[0-9]+]] = icmp eq <16 x i32>
// CHECK: %[[V:[0-9]+]] = bitcast <16 x i1> %[[EQ]] to <1 x i16>
// CHECK: %[[LOW:[0-9]+]] = extractelement <1 x i16> %[[V]], i64 0
// CHECK: %[[MASK:[0-9]+]] = zext i16 %[[LOW]] to i32
// CHECK: call <16 x i32> @llvm.aie4.vadd32.mask.512({{.*}}i32 %[[MASK]], i32 0)
func.func @equal_select(%a: vector<16xi32>, %b: vector<16xi32>, %t: vector<16xi32>, %f: vector<16xi32>) -> vector<16xi32> {
  %m = arith.cmpi eq, %a, %b : vector<16xi32>
  %r = arith.select %m, %t, %f : vector<16xi1>, vector<16xi32>
  return %r : vector<16xi32>
}

// CHECK-LABEL: define <32 x i16> @mask_logic(
// CHECK: %[[M:[0-9]+]] = bitcast <1 x i32> {{.*}} to <32 x i1>
// CHECK: %[[N:[0-9]+]] = bitcast <1 x i32> {{.*}} to <32 x i1>
// CHECK: %[[XOR:[0-9]+]] = xor <32 x i1> %[[M]], %[[N]]
// CHECK: %[[V:[0-9]+]] = bitcast <32 x i1> %[[XOR]] to <1 x i32>
// CHECK: %[[MASK:[0-9]+]] = extractelement <1 x i32> %[[V]], i64 0
// CHECK: call <32 x i16> @llvm.aie4.vadd16.mask.512({{.*}}i32 %[[MASK]], i32 0)
func.func @mask_logic(%a: vector<32xi16>, %b: vector<32xi16>, %c: vector<32xi16>, %d: vector<32xi16>) -> vector<32xi16> {
  %m = arith.cmpi slt, %a, %b : vector<32xi16>
  %n = arith.cmpi uge, %c, %d : vector<32xi16>
  %mask = arith.xori %m, %n : vector<32xi1>
  %r = arith.select %mask, %a, %b : vector<32xi1>, vector<32xi16>
  return %r : vector<32xi16>
}

// CHECK-LABEL: define <16 x i32> @scalar_select_16(
// CHECK-SAME: i1 %[[M:[0-9]+]], <16 x i32> %[[T:[0-9]+]], <16 x i32> %[[F:[0-9]+]])
// CHECK: %[[MASK:[0-9]+]] = sext i1 %[[M]] to i32
// CHECK: call <16 x i32> @llvm.aie4.vadd32.mask.512(<16 x i32> %[[F]], <16 x i32> %[[T]], i32 0, i32 2, i32 %[[MASK]], i32 0)
func.func @scalar_select_16(%m: i1, %t: vector<16xi32>, %f: vector<16xi32>) -> vector<16xi32> {
  %r = arith.select %m, %t, %f : vector<16xi32>
  return %r : vector<16xi32>
}

// CHECK-LABEL: define <64 x i8> @scalar_select_64(
// CHECK-SAME: i1 %[[M:[0-9]+]], <64 x i8> %[[T:[0-9]+]], <64 x i8> %[[F:[0-9]+]])
// CHECK: %[[MASK:[0-9]+]] = sext i1 %[[M]] to i64
// CHECK: %[[PAIR:[0-9]+]] = bitcast i64 %[[MASK]] to <2 x i32>
// CHECK: call <64 x i8> @llvm.aie4.vadd8.mask.512(<64 x i8> %[[F]], <64 x i8> %[[T]], i32 0, i32 2, <2 x i32> %[[PAIR]], i32 0)
func.func @scalar_select_64(%m: i1, %t: vector<64xi8>, %f: vector<64xi8>) -> vector<64xi8> {
  %r = arith.select %m, %t, %f : vector<64xi8>
  return %r : vector<64xi8>
}

// CHECK-LABEL: define <16 x float> @float_select(
// CHECK-NOT: @llvm.aie4
// CHECK: %[[R:[0-9]+]] = select <16 x i1> {{.*}}, <16 x float> {{.*}}, <16 x float>
// CHECK-NOT: @llvm.aie4
// CHECK: ret <16 x float> %[[R]]
func.func @float_select(%m: vector<16xi1>, %t: vector<16xf32>, %f: vector<16xf32>) -> vector<16xf32> {
  %r = arith.select %m, %t, %f : vector<16xi1>, vector<16xf32>
  return %r : vector<16xf32>
}

// CHECK-LABEL: define i1 @dynamic_mask_extract(
// CHECK-SAME: <64 x i8> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i64 %[[INDEX:[0-9]+]])
// CHECK: %[[INDEX32:[0-9]+]] = trunc i64 %[[INDEX]] to i32
// CHECK: %[[WORDS:[0-9]+]] = bitcast <1 x i64> {{.*}} to <2 x i32>
// CHECK: %[[LOW:[0-9]+]] = extractelement <2 x i32> %[[WORDS]], i64 0
// CHECK: %[[HIGH:[0-9]+]] = extractelement <2 x i32> %[[WORDS]], i64 1
// CHECK: %[[UPPER:[0-9]+]] = icmp uge i32 %[[INDEX32]], 32
// CHECK: %[[WORD:[0-9]+]] = select i1 %[[UPPER]], i32 %[[HIGH]], i32 %[[LOW]]
// CHECK: %[[OFFSET:[0-9]+]] = and i32 %[[INDEX32]], 31
// CHECK: %[[SHIFT:[0-9]+]] = lshr i32 %[[WORD]], %[[OFFSET]]
// CHECK: %[[BIT:[0-9]+]] = trunc i32 %[[SHIFT]] to i1
// CHECK: ret i1 %[[BIT]]
func.func @dynamic_mask_extract(%a: vector<64xi8>, %b: vector<64xi8>, %index: index) -> i1 {
  %mask = arith.cmpi slt, %a, %b : vector<64xi8>
  %r = vector.extract %mask[%index] : i1 from vector<64xi1>
  return %r : i1
}

// CHECK-LABEL: define { i1, <64 x i16> } @multiple_mask_consumers(
// CHECK-SAME: <64 x i16> %[[A:[0-9]+]], <64 x i16> %[[B:[0-9]+]], <64 x i16> %[[T:[0-9]+]], <64 x i16> %[[F:[0-9]+]])
// CHECK: %[[CMP:[0-9]+]] = call { <64 x i16>, <2 x i32> } @llvm.aie4.vadd16.1024(<64 x i16> %[[A]], <64 x i16> %[[B]], i32 1, i32 15, i32 0)
// CHECK: %[[PAIR:[0-9]+]] = extractvalue { <64 x i16>, <2 x i32> } %[[CMP]], 1
// CHECK: %[[MASK:[0-9]+]] = bitcast <2 x i32> %[[PAIR]] to i64
// CHECK: %[[V:[0-9]+]] = insertelement <1 x i64> poison, i64 %[[MASK]], i32 0
// CHECK: %[[WORDS:[0-9]+]] = bitcast <1 x i64> %[[V]] to <2 x i32>
// CHECK: %[[HIGH:[0-9]+]] = extractelement <2 x i32> %[[WORDS]], i64 1
// CHECK: %[[SHIFT:[0-9]+]] = lshr i32 %[[HIGH]], 31
// CHECK: %[[BIT:[0-9]+]] = trunc i32 %[[SHIFT]] to i1
// CHECK: %[[SELECT:[0-9]+]] = call <64 x i16> @llvm.aie4.vadd16.mask.1024(<64 x i16> %[[F]], <64 x i16> %[[T]], i32 0, i32 2, <2 x i32> %[[PAIR]], i32 0)
// CHECK: %[[RESULT0:[0-9]+]] = insertvalue { i1, <64 x i16> } poison, i1 %[[BIT]], 0
// CHECK: %[[RESULT:[0-9]+]] = insertvalue { i1, <64 x i16> } %[[RESULT0]], <64 x i16> %[[SELECT]], 1
// CHECK: ret { i1, <64 x i16> } %[[RESULT]]
func.func @multiple_mask_consumers(%a: vector<64xi16>, %b: vector<64xi16>, %t: vector<64xi16>, %f: vector<64xi16>) -> (i1, vector<64xi16>) {
  %mask = arith.cmpi slt, %a, %b : vector<64xi16>
  %bit = vector.extract %mask[63] : i1 from vector<64xi1>
  %selected = arith.select %mask, %t, %f : vector<64xi1>, vector<64xi16>
  return %bit, %selected : i1, vector<64xi16>
}

// CHECK-LABEL: define <16 x i32> @signed_select(
// CHECK-SAME: <16 x i1> %[[M:[0-9]+]], <16 x i32> %[[T:[0-9]+]], <16 x i32> %[[F:[0-9]+]])
// CHECK-NOT: @llvm.aie4
// CHECK: %[[R:[0-9]+]] = select <16 x i1> %[[M]], <16 x i32> %[[T]], <16 x i32> %[[F]]
// CHECK-NOT: @llvm.aie4
// CHECK: ret <16 x i32> %[[R]]
func.func @signed_select(%m: vector<16xi1>, %t: vector<16xsi32>, %f: vector<16xsi32>) -> vector<16xsi32> {
  %r = arith.select %m, %t, %f : vector<16xi1>, vector<16xsi32>
  return %r : vector<16xsi32>
}

// CHECK-LABEL: define <16 x i32> @unsigned_select(
// CHECK-SAME: <16 x i1> %[[M:[0-9]+]], <16 x i32> %[[T:[0-9]+]], <16 x i32> %[[F:[0-9]+]])
// CHECK-NOT: @llvm.aie4
// CHECK: %[[R:[0-9]+]] = select <16 x i1> %[[M]], <16 x i32> %[[T]], <16 x i32> %[[F]]
// CHECK-NOT: @llvm.aie4
// CHECK: ret <16 x i32> %[[R]]
func.func @unsigned_select(%m: vector<16xi1>, %t: vector<16xui32>, %f: vector<16xui32>) -> vector<16xui32> {
  %r = arith.select %m, %t, %f : vector<16xi1>, vector<16xui32>
  return %r : vector<16xui32>
}

// CHECK-LABEL: define i1 @extract_mask_logic(
// CHECK-SAME: <64 x i1> %[[M:[0-9]+]], <64 x i1> %[[N:[0-9]+]], i64 %[[INDEX:[0-9]+]])
// CHECK: %[[INDEX32:[0-9]+]] = trunc i64 %[[INDEX]] to i32
// CHECK: %[[MV:[0-9]+]] = bitcast <64 x i1> %[[M]] to <2 x i32>
// CHECK: %[[ML:[0-9]+]] = extractelement <2 x i32> %[[MV]], i64 0
// CHECK: %[[MH:[0-9]+]] = extractelement <2 x i32> %[[MV]], i64 1
// CHECK: %[[UPPER:[0-9]+]] = icmp uge i32 %[[INDEX32]], 32
// CHECK: %[[MW:[0-9]+]] = select i1 %[[UPPER]], i32 %[[MH]], i32 %[[ML]]
// CHECK: %[[OFFSET:[0-9]+]] = and i32 %[[INDEX32]], 31
// CHECK: %[[MS:[0-9]+]] = lshr i32 %[[MW]], %[[OFFSET]]
// CHECK: %[[MB:[0-9]+]] = trunc i32 %[[MS]] to i1
// CHECK: %[[NV:[0-9]+]] = bitcast <64 x i1> %[[N]] to <2 x i32>
// CHECK: %[[NL:[0-9]+]] = extractelement <2 x i32> %[[NV]], i64 0
// CHECK: %[[NH:[0-9]+]] = extractelement <2 x i32> %[[NV]], i64 1
// CHECK: %[[NW:[0-9]+]] = select i1 %[[UPPER]], i32 %[[NH]], i32 %[[NL]]
// CHECK: %[[NS:[0-9]+]] = lshr i32 %[[NW]], %[[OFFSET]]
// CHECK: %[[NB:[0-9]+]] = trunc i32 %[[NS]] to i1
// CHECK: %[[XOR:[0-9]+]] = xor i1 %[[MB]], %[[NB]]
// CHECK: ret i1 %[[XOR]]
func.func @extract_mask_logic(%m: vector<64xi1>, %n: vector<64xi1>, %index: index) -> i1 {
  %mask = arith.xori %m, %n : vector<64xi1>
  %r = vector.extract %mask[%index] : i1 from vector<64xi1>
  return %r : i1
}

// CHECK-LABEL: define i1 @extract_nested_mask_logic(
// CHECK-SAME: <64 x i1> %{{[0-9]+}}, <64 x i1> %{{[0-9]+}}, <64 x i1> %[[P:[0-9]+]], <64 x i1> %[[Q:[0-9]+]], i64 %[[INDEX:[0-9]+]])
// CHECK: %[[INDEX32:[0-9]+]] = trunc i64 %[[INDEX]] to i32
// CHECK: %[[UPPER:[0-9]+]] = icmp uge i32 %[[INDEX32]], 32
// CHECK: %[[OFFSET:[0-9]+]] = and i32 %[[INDEX32]], 31
// CHECK: %[[XOR:[0-9]+]] = xor i1
// CHECK: %[[PV:[0-9]+]] = bitcast <64 x i1> %[[P]] to <2 x i32>
// CHECK: %[[PL:[0-9]+]] = extractelement <2 x i32> %[[PV]], i64 0
// CHECK: %[[PH:[0-9]+]] = extractelement <2 x i32> %[[PV]], i64 1
// CHECK: %[[PW:[0-9]+]] = select i1 %[[UPPER]], i32 %[[PH]], i32 %[[PL]]
// CHECK: %[[PS:[0-9]+]] = lshr i32 %[[PW]], %[[OFFSET]]
// CHECK: %[[PB:[0-9]+]] = trunc i32 %[[PS]] to i1
// CHECK: %[[AND:[0-9]+]] = and i1 %[[XOR]], %[[PB]]
// CHECK: %[[QV:[0-9]+]] = bitcast <64 x i1> %[[Q]] to <2 x i32>
// CHECK: %[[QL:[0-9]+]] = extractelement <2 x i32> %[[QV]], i64 0
// CHECK: %[[QH:[0-9]+]] = extractelement <2 x i32> %[[QV]], i64 1
// CHECK: %[[QW:[0-9]+]] = select i1 %[[UPPER]], i32 %[[QH]], i32 %[[QL]]
// CHECK: %[[QS:[0-9]+]] = lshr i32 %[[QW]], %[[OFFSET]]
// CHECK: %[[QB:[0-9]+]] = trunc i32 %[[QS]] to i1
// CHECK: %[[OR:[0-9]+]] = or i1 %[[AND]], %[[QB]]
// CHECK: ret i1 %[[OR]]
func.func @extract_nested_mask_logic(%m: vector<64xi1>, %n: vector<64xi1>, %p: vector<64xi1>, %q: vector<64xi1>, %index: index) -> i1 {
  %xor = arith.xori %m, %n : vector<64xi1>
  %and = arith.andi %xor, %p : vector<64xi1>
  %or = arith.ori %and, %q : vector<64xi1>
  %r = vector.extract %or[%index] : i1 from vector<64xi1>
  return %r : i1
}
