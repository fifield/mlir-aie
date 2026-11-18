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

// AIE4 values are from aie-codegen's xaie4gbl_params.h.
static void expectRegister(const RegisterDatabase &db, llvm::StringRef module,
                           llvm::StringRef name, uint32_t offset) {
  const RegisterInfo *reg = db.lookupRegister(name, module);
  if (!reg)
    throw std::runtime_error("AIE4: missing " + module.str() + " register " +
                             name.str());
  if (reg->offset != offset)
    throw std::runtime_error("AIE4: wrong offset for " + module.str() +
                             "::" + name.str());
  if (db.lookupRegisterByOffset(offset, module) == nullptr)
    throw std::runtime_error("AIE4: no offset lookup for " + module.str() +
                             "::" + name.str());
  std::cout << "  ✓ " << module.str() << "::" << name.str() << " at 0x"
            << std::hex << offset << std::dec << "\n";
}

static void expectField(const RegisterDatabase &db, llvm::StringRef module,
                        llvm::StringRef name, llvm::StringRef field,
                        uint32_t lsb, uint32_t msb) {
  const RegisterInfo *reg = db.lookupRegister(name, module);
  const BitFieldInfo *f = reg ? reg->getField(field) : nullptr;
  if (!f)
    throw std::runtime_error("AIE4: missing field " + name.str() + "." +
                             field.str());
  if (f->bit_start != lsb || f->bit_end != msb)
    throw std::runtime_error("AIE4: wrong bits for " + name.str() + "." +
                             field.str());
  std::cout << "  ✓ " << name.str() << "." << field.str() << " [" << msb << ":"
            << lsb << "]\n";
}

void test_aie4_registers() {
  std::cout << "Test: AIE4 Register Lookup\n";
  auto db = RegisterDatabase::loadAIE4();
  if (!db)
    throw std::runtime_error("Failed to load AIE4 register database");

  expectRegister(*db, "core", "Core_Control", 0x000A3000);
  expectRegister(*db, "core", "Timer_Control", 0x000A0110);
  expectRegister(*db, "core", "Trace0_Control0", 0x000A00D0);
  expectRegister(*db, "core", "Event_Generate", 0x000A0008);
  expectRegister(*db, "memory", "DMA_BD0_0", 0x000AC000);
  expectRegister(*db, "memory", "DMA_S2MM_0_Ctrl", 0x000ACE00);
  expectRegister(*db, "memory", "DMA_MM2S_0_Ctrl", 0x000ACE10);
  // Memtile: BDs are per channel; registers below the 1MB data memory
  expectRegister(*db, "memory_tile", "DMA_S2MM_0_BD0_0", 0x00008000);
  expectRegister(*db, "memory_tile", "DMA_S2MM_0_Ctrl", 0x00009E00);
  expectRegister(*db, "memory_tile", "Timer_A_Control", 0x00000130);
  expectRegister(*db, "shim", "DMA_BD0_0", 0x00018000);
  expectRegister(*db, "shim", "DMA_S2MM_0_Ctrl", 0x00018540);
  expectRegister(*db, "shim", "DMA_S2MM_0_Task_Queue", 0x00018544);
  expectRegister(*db, "shim", "Timer_A_Control", 0x00004090);

  expectField(*db, "core", "Core_Control", "Reset", 1, 1);
  expectField(*db, "core", "Core_Control", "Enable", 0, 0);
  expectField(*db, "core", "Timer_Control", "Reset_Event", 8, 15);
  expectField(*db, "memory", "DMA_S2MM_0_Ctrl", "Reset", 1, 1);
  expectField(*db, "memory", "DMA_S2MM_0_Ctrl", "Controller_ID", 8, 15);
  expectField(*db, "shim", "DMA_S2MM_0_Task_Queue", "Start_BD_ID", 0, 3);
  expectField(*db, "shim", "DMA_S2MM_0_Task_Queue", "Enable_Token_Issue", 31,
              31);

  // Register and field lookups fold case.
  const RegisterInfo *timerControl =
      db->lookupRegister("TIMER_CONTROL", "core");
  if (!timerControl || timerControl->name != "TIMER_CONTROL")
    throw std::runtime_error("AIE4: Timer_Control is not named TIMER_CONTROL");
  if (timerControl->getField("RESET_EVENT") !=
          timerControl->getField("Reset_Event") ||
      !timerControl->getField("reset_event"))
    throw std::runtime_error("AIE4: field lookup is case-sensitive");
  std::cout << "  ✓ Register and field lookups fold case\n";

  const BitFieldInfo *resetEvent =
      db->lookupRegister("Timer_Control", "core")->getField("Reset_Event");
  auto encoded = db->encodeFieldValue(*resetEvent, 0x25);
  if (!encoded || *encoded != 0x2500)
    throw std::runtime_error("AIE4: wrong Reset_Event encoding");
  std::cout << "  ✓ Timer_Control.Reset_Event(0x25) = 0x2500\n";

  if (db->lookupRegister("Trace0_Control0", "memory_tile"))
    throw std::runtime_error("AIE4: Trace0_Control0 found in memory_tile");
  std::cout << "  ✓ Module separation\n";
}

void test_aie4_events() {
  std::cout << "Test: AIE4 Event Lookup\n";
  auto db = RegisterDatabase::loadAIE4();
  if (!db)
    throw std::runtime_error("Failed to load AIE4 register database");
  // xaie_events_aie4.h
  auto startBd = db->lookupEvent("DMA_S2MM_0_START_BD", "core");
  if (!startBd || *startBd != 151)
    throw std::runtime_error("AIE4: wrong core DMA_S2MM_0_START_BD event");
  std::cout << "  ✓ core DMA_S2MM_0_START_BD = " << *startBd << "\n";
  auto user0 = db->lookupEvent("USER_EVENT_0", "pl");
  if (!user0)
    throw std::runtime_error("AIE4: missing pl USER_EVENT_0 event");
  std::cout << "  ✓ pl USER_EVENT_0 = " << *user0 << "\n";
}

void test_aie4_target_model() {
  std::cout << "Test: AIE4 Target Model Register Lookup\n";
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu3);
  struct Expected {
    TileID tile;
    bool isMem;
    const char *name;
    uint32_t offset;
  };
  for (const Expected &e : {
           Expected{{0, 2}, false, "Timer_Control", 0x000A0110},
           Expected{{0, 2}, false, "Core_Control", 0x000A3000},
           Expected{{0, 2}, true, "DMA_S2MM_0_Ctrl", 0x000ACE00},
           Expected{{0, 1}, true, "DMA_S2MM_0_Ctrl", 0x00009E00},
           Expected{{0, 0}, true, "DMA_S2MM_0_Ctrl", 0x00018540},
       }) {
    const RegisterInfo *reg = tm.lookupRegister(e.name, e.tile, e.isMem);
    if (!reg || reg->offset != e.offset)
      throw std::runtime_error(std::string("npu3 lookup failed: ") + e.name);
    std::cout << "  ✓ tile(" << e.tile.col << ", " << e.tile.row << ") "
              << e.name << "\n";
  }
  if (tm.lookupRegister("DMA_S2MM_0_Ctrl", {0, 2}, /*isMem=*/false))
    throw std::runtime_error("npu3: core module has no DMA_S2MM_0_Ctrl");

  // Compute-tile memory events are numbered in the core module's space
  auto coreEvent = tm.lookupEvent("DMA_S2MM_0_START_BD", {0, 2}, false);
  auto memEvent = tm.lookupEvent("DMA_S2MM_0_START_BD", {0, 2}, true);
  if (!coreEvent || !memEvent || *coreEvent != *memEvent)
    throw std::runtime_error("npu3: memory-module event lookup");
  std::cout << "  ✓ memory-module events resolve in the core module\n";

  // npu1/npu2 still use the AIE2 database
  const RegisterInfo *aie2Ctrl =
      getTargetModel(AIEDevice::npu2).lookupRegister("Core_Control", {0, 2});
  if (!aie2Ctrl || aie2Ctrl->offset != 0x00032000)
    throw std::runtime_error("npu2: Core_Control lookup changed");
  std::cout << "  ✓ npu2 Core_Control unchanged\n";
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
    test_aie4_registers();
    test_aie4_events();
    test_aie4_target_model();

    std::cout << "\n==============================================\n";
    std::cout << "All tests passed! ✓\n";
    std::cout << "==============================================\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed: " << e.what() << "\n";
    return 1;
  }
}
