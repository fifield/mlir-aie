//===- register_database.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"

#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace xilinx::AIE;
using namespace mlir;

void test_load_database() {
  std::cout << "Test: Load AIE2 Register Database\n";
  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load AIE2 register database");
  }
  std::cout << "  ✓ Database loaded successfully\n";
}

void test_lookup_core_events() {
  std::cout << "Test: Lookup Core Events\n";
  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }
  // Test INSTR_EVENT_0 in core module
  auto instr0 = db->lookupEvent("INSTR_EVENT_0", "core");
  if (!instr0) {
    throw std::runtime_error("Failed to find INSTR_EVENT_0 in core module");
  }
  std::cout << "  ✓ INSTR_EVENT_0 = " << *instr0 << "\n";
}

void test_lookup_pl_events() {
  std::cout << "Test: Lookup PL (Shim) Events\n";
  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Test DMA events in pl module
  auto dma_s2mm_0_start = db->lookupEvent("DMA_S2MM_0_START_TASK", "pl");
  if (!dma_s2mm_0_start) {
    throw std::runtime_error(
        "Failed to find DMA_S2MM_0_START_TASK in pl module");
  }
  std::cout << "  ✓ DMA_S2MM_0_START_TASK = " << *dma_s2mm_0_start << "\n";
}

void test_nonexistent_event() {
  std::cout << "Test: Lookup Non-existent Event\n";
  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }
  // Should return nullopt for non-existent event
  auto result = db->lookupEvent("NONEXISTENT_EVENT", "core");
  if (result.has_value()) {
    throw std::runtime_error("Expected nullopt for non-existent event");
  }
  std::cout << "  ✓ Correctly returns nullopt for non-existent event\n";
}

void test_wrong_module() {
  std::cout << "Test: Lookup Event in Wrong Module\n";
  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // INSTR_EVENT_0 exists in core, but not in pl
  auto result = db->lookupEvent("INSTR_EVENT_0", "pl");
  if (result.has_value()) {
    throw std::runtime_error(
        "Expected nullopt when looking up core event in pl module");
  }
  std::cout << "  ✓ Correctly returns nullopt when module is wrong\n";
}

void test_lookup_register() {
  std::cout << "Test: Lookup Registers\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Test Core_Control register in core module
  auto coreControl = db->lookupRegister("Core_Control", "core");
  if (!coreControl) {
    throw std::runtime_error("Failed to find Core_Control in core module");
  }
  std::cout << "  ✓ Core_Control found\n";
  std::cout << "    - offset: 0x" << std::hex << coreControl->offset << std::dec
            << "\n";
  std::cout << "    - width: " << coreControl->width << " bits\n";

  // Verify offset is 0x32000
  if (coreControl->offset != 0x32000) {
    throw std::runtime_error("Core_Control offset mismatch");
  }
  std::cout << "  ✓ Core_Control offset is correct (0x32000)\n";
}

void test_lookup_register_wrong_module() {
  std::cout << "Test: Lookup Register in Wrong Module\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Core_Control exists in core, but not in pl
  auto result = db->lookupRegister("Core_Control", "pl");
  if (result != nullptr) {
    throw std::runtime_error(
        "Expected nullptr when looking up core register in pl module");
  }
  std::cout << "  ✓ Correctly returns nullptr when module is wrong\n";
}

void test_lookup_nonexistent_register() {
  std::cout << "Test: Lookup Non-existent Register\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  auto result = db->lookupRegister("NONEXISTENT_REGISTER", "core");
  if (result != nullptr) {
    throw std::runtime_error("Expected nullptr for non-existent register");
  }
  std::cout << "  ✓ Correctly returns nullptr for non-existent register\n";
}

void test_register_bit_fields() {
  std::cout << "Test: Register Bit Fields\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Get Core_Control register which has Enable and Reset fields
  auto coreControl = db->lookupRegister("Core_Control", "core");
  if (!coreControl) {
    throw std::runtime_error("Failed to find Core_Control");
  }

  // Check that bit_fields are populated
  if (coreControl->bit_fields.empty()) {
    throw std::runtime_error("Core_Control has no bit fields");
  }
  std::cout << "  ✓ Core_Control has " << coreControl->bit_fields.size()
            << " bit fields\n";

  // Look for the Enable field (bit 0)
  const BitFieldInfo *enableField = coreControl->getField("Enable");
  if (!enableField) {
    throw std::runtime_error("Failed to find Enable bit field");
  }
  std::cout << "  ✓ Enable field found at bits [" << enableField->bit_start
            << ":" << enableField->bit_end << "]\n";

  // Look for the Reset field (bit 1)
  const BitFieldInfo *resetField = coreControl->getField("Reset");
  if (!resetField) {
    throw std::runtime_error("Failed to find Reset bit field");
  }
  std::cout << "  ✓ Reset field found at bits [" << resetField->bit_start << ":"
            << resetField->bit_end << "]\n";
}

void test_encode_field_value() {
  std::cout << "Test: Encode Field Values\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Get Performance_Control0 which has multi-bit fields
  auto perfControl = db->lookupRegister("Performance_Control0", "core");
  if (!perfControl) {
    throw std::runtime_error("Failed to find Performance_Control0");
  }

  // Test encoding Cnt0_Start_Event (bits 6:0)
  // Value 0x25 (event 37) should stay at bits 0-6
  const BitFieldInfo *cnt0Start = perfControl->getField("Cnt0_Start_Event");
  if (!cnt0Start) {
    throw std::runtime_error("Failed to find Cnt0_Start_Event field");
  }

  auto encoded0 = db->encodeFieldValue(*cnt0Start, 0x25);
  if (!encoded0) {
    throw std::runtime_error("Cnt0_Start_Event encoding unexpectedly failed");
  }
  if (*encoded0 != 0x25) {
    std::cerr << "Expected 0x25, got 0x" << std::hex << *encoded0 << std::dec
              << "\n";
    throw std::runtime_error("Cnt0_Start_Event encoding failed");
  }
  std::cout << "  ✓ Cnt0_Start_Event(0x25) = 0x" << std::hex << *encoded0
            << std::dec << "\n";

  // Test encoding Cnt0_Stop_Event (bits 14:8)
  // Value 0x21 should be shifted to bits 8-14, resulting in 0x2100
  const BitFieldInfo *cnt0Stop = perfControl->getField("Cnt0_Stop_Event");
  if (!cnt0Stop) {
    throw std::runtime_error("Failed to find Cnt0_Stop_Event field");
  }

  auto encoded1 = db->encodeFieldValue(*cnt0Stop, 0x21);
  if (!encoded1) {
    throw std::runtime_error("Cnt0_Stop_Event encoding unexpectedly failed");
  }
  if (*encoded1 != 0x2100) {
    std::cerr << "Expected 0x2100, got 0x" << std::hex << *encoded1 << std::dec
              << "\n";
    throw std::runtime_error("Cnt0_Stop_Event encoding failed");
  }
  std::cout << "  ✓ Cnt0_Stop_Event(0x21) = 0x" << std::hex << *encoded1
            << std::dec << "\n";
}

void test_encode_single_bit_field() {
  std::cout << "Test: Encode Single-Bit Field Values\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Get Core_Control which has single-bit fields
  auto coreControl = db->lookupRegister("Core_Control", "core");
  if (!coreControl) {
    throw std::runtime_error("Failed to find Core_Control");
  }

  // Test Enable field (bit 0)
  const BitFieldInfo *enableField = coreControl->getField("Enable");
  if (!enableField) {
    throw std::runtime_error("Failed to find Enable field");
  }

  auto enableEncoded = db->encodeFieldValue(*enableField, 1);
  if (!enableEncoded) {
    throw std::runtime_error("Enable encoding unexpectedly failed");
  }
  if (*enableEncoded != 0x1) {
    std::cerr << "Expected 0x1, got 0x" << std::hex << *enableEncoded
              << std::dec << "\n";
    throw std::runtime_error("Enable encoding failed");
  }
  std::cout << "  ✓ Enable(1) = 0x" << std::hex << *enableEncoded << std::dec
            << "\n";

  // Test Reset field (bit 1)
  const BitFieldInfo *resetField = coreControl->getField("Reset");
  if (!resetField) {
    throw std::runtime_error("Failed to find Reset field");
  }

  auto resetEncoded = db->encodeFieldValue(*resetField, 1);
  if (!resetEncoded) {
    throw std::runtime_error("Reset encoding unexpectedly failed");
  }
  if (*resetEncoded != 0x2) {
    std::cerr << "Expected 0x2, got 0x" << std::hex << *resetEncoded << std::dec
              << "\n";
    throw std::runtime_error("Reset encoding failed");
  }
  std::cout << "  ✓ Reset(1) = 0x" << std::hex << *resetEncoded << std::dec
            << "\n";
}

void test_encode_field_value_overflow() {
  std::cout << "Test: Encode Field Value Overflow\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Core_Control's single-bit Enable field cannot hold a value above 1.
  auto coreControl = db->lookupRegister("Core_Control", "core");
  if (!coreControl) {
    throw std::runtime_error("Failed to find Core_Control");
  }
  const BitFieldInfo *enableField = coreControl->getField("Enable");
  if (!enableField) {
    throw std::runtime_error("Failed to find Enable field");
  }

  auto overflowed = db->encodeFieldValue(*enableField, 2);
  if (overflowed.has_value()) {
    throw std::runtime_error(
        "Expected nullopt when a value overflows its field width");
  }
  std::cout << "  ✓ Correctly returns nullopt when a value overflows its "
               "field width\n";
}

// ======== AIE4 Tests ========

void test_aie4_load_database() {
  std::cout << "Test: Load AIE4 Register Database\n";

  auto db = RegisterDatabase::loadAIE4();
  if (!db) {
    throw std::runtime_error("Failed to load AIE4 register database");
  }

  std::cout << "  ✓ Database loaded successfully\n";
}

void test_aie4_lookup_register() {
  std::cout << "Test: AIE4 Lookup Registers\n";

  auto db = RegisterDatabase::loadAIE4();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Trace0_Control0 in core module, offset 0xA00D0
  auto traceCtrl = db->lookupRegister("Trace0_Control0", "core");
  if (!traceCtrl) {
    throw std::runtime_error("Failed to find Trace0_Control0 in core");
  }
  std::cout << "  ✓ Trace0_Control0 found\n";
  std::cout << "    - offset: 0x" << std::hex << traceCtrl->offset << std::dec
            << "\n";

  if (traceCtrl->offset != 0xA00D0) {
    throw std::runtime_error("Trace0_Control0 offset mismatch");
  }
  std::cout << "  ✓ Trace0_Control0 offset is correct (0xA00D0)\n";
}

void test_aie4_bitfield_parsing() {
  std::cout << "Test: AIE4 Bit Field Parsing\n";

  auto db = RegisterDatabase::loadAIE4();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  auto traceCtrl = db->lookupRegister("Trace0_Control0", "core");
  if (!traceCtrl) {
    throw std::runtime_error("Failed to find Trace0_Control0");
  }

  if (traceCtrl->bit_fields.empty()) {
    throw std::runtime_error("Trace0_Control0 has no bit fields");
  }
  std::cout << "  ✓ Trace0_Control0 has " << traceCtrl->bit_fields.size()
            << " bit fields\n";

  // Trace_Stop_Event: bit_range [24, 31]
  const BitFieldInfo *stopEvent = traceCtrl->getField("Trace_Stop_Event");
  if (!stopEvent) {
    throw std::runtime_error("Failed to find Trace_Stop_Event field");
  }
  if (stopEvent->bit_start != 24 || stopEvent->bit_end != 31) {
    std::cerr << "Expected [24:31], got [" << stopEvent->bit_start << ":"
              << stopEvent->bit_end << "]\n";
    throw std::runtime_error("Trace_Stop_Event bit range mismatch");
  }
  std::cout << "  ✓ Trace_Stop_Event at bits [" << stopEvent->bit_start << ":"
            << stopEvent->bit_end << "], width=" << stopEvent->getWidth()
            << "\n";

  // Mode: bit_range [0, 1]
  const BitFieldInfo *mode = traceCtrl->getField("Mode");
  if (!mode) {
    throw std::runtime_error("Failed to find Mode field");
  }
  if (mode->bit_start != 0 || mode->bit_end != 1) {
    throw std::runtime_error("Mode bit range mismatch");
  }
  std::cout << "  ✓ Mode at bits [" << mode->bit_start << ":" << mode->bit_end
            << "], width=" << mode->getWidth() << "\n";
}

void test_aie4_encode_field_value() {
  std::cout << "Test: AIE4 Encode Field Values\n";

  auto db = RegisterDatabase::loadAIE4();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Use Performance_Start_Stop_0_1 from memory_tile module
  auto perfCtrl =
      db->lookupRegister("Performance_Start_Stop_0_1", "memory_tile");
  if (!perfCtrl) {
    throw std::runtime_error(
        "Failed to find Performance_Start_Stop_0_1 in memory_tile");
  }

  // Cnt0_Start_Event (bits 7:0) - value 0x25 should stay at bits 0-7
  const BitFieldInfo *cnt0Start = perfCtrl->getField("Cnt0_Start_Event");
  if (!cnt0Start) {
    throw std::runtime_error("Failed to find Cnt0_Start_Event field");
  }

  uint32_t encoded0 = db->encodeFieldValue(*cnt0Start, 0x25).value();
  if (encoded0 != 0x25) {
    std::cerr << "Expected 0x25, got 0x" << std::hex << encoded0 << std::dec
              << "\n";
    throw std::runtime_error("Cnt0_Start_Event encoding failed");
  }
  std::cout << "  ✓ Cnt0_Start_Event(0x25) = 0x" << std::hex << encoded0
            << std::dec << "\n";

  // Cnt0_Stop_Event (bits 15:8) - value 0x21 -> 0x2100
  const BitFieldInfo *cnt0Stop = perfCtrl->getField("Cnt0_Stop_Event");
  if (!cnt0Stop) {
    throw std::runtime_error("Failed to find Cnt0_Stop_Event field");
  }

  uint32_t encoded1 = db->encodeFieldValue(*cnt0Stop, 0x21).value();
  if (encoded1 != 0x2100) {
    std::cerr << "Expected 0x2100, got 0x" << std::hex << encoded1 << std::dec
              << "\n";
    throw std::runtime_error("Cnt0_Stop_Event encoding failed");
  }
  std::cout << "  ✓ Cnt0_Stop_Event(0x21) = 0x" << std::hex << encoded1
            << std::dec << "\n";

  // Cnt1_Start_Event (bits 23:16) - value 0x10 -> 0x100000
  const BitFieldInfo *cnt1Start = perfCtrl->getField("Cnt1_Start_Event");
  if (!cnt1Start) {
    throw std::runtime_error("Failed to find Cnt1_Start_Event field");
  }

  uint32_t encoded2 = db->encodeFieldValue(*cnt1Start, 0x10).value();
  if (encoded2 != 0x100000) {
    std::cerr << "Expected 0x100000, got 0x" << std::hex << encoded2 << std::dec
              << "\n";
    throw std::runtime_error("Cnt1_Start_Event encoding failed");
  }
  std::cout << "  ✓ Cnt1_Start_Event(0x10) = 0x" << std::hex << encoded2
            << std::dec << "\n";
}

void test_aie4_cross_module() {
  std::cout << "Test: AIE4 Cross-Module Register Lookup\n";

  auto db = RegisterDatabase::loadAIE4();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Performance_Start_Stop_0_1 exists in both memory_tile and shim modules
  // with different offsets
  auto memTileReg =
      db->lookupRegister("Performance_Start_Stop_0_1", "memory_tile");
  auto plReg = db->lookupRegister("Performance_Start_Stop_0_1", "shim");

  if (!memTileReg) {
    throw std::runtime_error(
        "Failed to find Performance_Start_Stop_0_1 in memory_tile");
  }
  if (!plReg) {
    throw std::runtime_error(
        "Failed to find Performance_Start_Stop_0_1 in shim");
  }

  // memory_tile offset=0x00000000, shim offset=0x00004000
  if (memTileReg->offset != 0x0) {
    throw std::runtime_error("memory_tile offset mismatch");
  }
  if (plReg->offset != 0x4000) {
    throw std::runtime_error("shim offset mismatch");
  }
  std::cout << "  ✓ memory_tile offset=0x" << std::hex << memTileReg->offset
            << ", shim offset=0x" << plReg->offset << std::dec << "\n";
}

void test_aie4_wrong_module() {
  std::cout << "Test: AIE4 Lookup Register in Wrong Module\n";

  auto db = RegisterDatabase::loadAIE4();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Trace0_Control0 exists in core module, not in memory module
  auto result = db->lookupRegister("Trace0_Control0", "memory");
  if (result != nullptr) {
    throw std::runtime_error(
        "Expected nullptr when looking up core register in memory");
  }
  std::cout << "  ✓ Correctly returns nullptr when module is wrong\n";
}

void test_aie4_nonexistent_register() {
  std::cout << "Test: AIE4 Lookup Non-existent Register\n";

  auto db = RegisterDatabase::loadAIE4();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  auto result = db->lookupRegister("NONEXISTENT_REGISTER", "core");
  if (result != nullptr) {
    throw std::runtime_error("Expected nullptr for non-existent register");
  }
  std::cout << "  ✓ Correctly returns nullptr for non-existent register\n";
}

int main() {
  try {
    std::cout << "==============================================\n";
    std::cout << "Register Database Unit Tests\n";
    std::cout << "==============================================\n\n";
    test_load_database();
    test_lookup_core_events();
    test_lookup_pl_events();
    test_nonexistent_event();
    test_wrong_module();
    test_lookup_register();
    test_lookup_register_wrong_module();
    test_lookup_nonexistent_register();
    test_register_bit_fields();
    test_encode_field_value();
    test_encode_single_bit_field();
    test_encode_field_value_overflow();

    // AIE4 tests
    test_aie4_load_database();
    test_aie4_lookup_register();
    test_aie4_bitfield_parsing();
    test_aie4_encode_field_value();
    test_aie4_cross_module();
    test_aie4_wrong_module();
    test_aie4_nonexistent_register();

    std::cout << "\n==============================================\n";
    std::cout << "All tests passed! ✓\n";
    std::cout << "==============================================\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed: " << e.what() << "\n";
    return 1;
  }
}
