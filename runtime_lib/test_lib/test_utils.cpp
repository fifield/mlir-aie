//===- test_utils.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the generic host code

#include "test_utils.h"
#ifndef TEST_UTILS_XRT_ONLY
#include "memory_allocator.h"
#endif

#include <cassert>
#include <filesystem>

#ifdef TEST_UTILS_USE_XRT
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#endif

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

void test_utils::check_arg_file_exists(const cxxopts::ParseResult &result,
                                       std::string name) {
  if (!result.count(name)) {
    throw std::runtime_error("Missing required argument: " + name);
  }
  std::string path = result[name].as<std::string>();
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("File does not exist: " + path);
  }
}

void test_utils::add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "verify", "whether to verify the AIE computed output",
      cxxopts::value<bool>()->default_value("true"))(
      "iters", "number of iterations",
      cxxopts::value<int>()->default_value("1"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("0"))(
      "trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))(
      "trace_file", "where to store trace output",
      cxxopts::value<std::string>()->default_value("trace.txt"));
}

void test_utils::parse_options(int argc, const char *argv[],
                               cxxopts::Options &options,
                               cxxopts::ParseResult &vm) {
  try {
    vm = options.parse(argc, argv);

    if (vm.count("help")) {
      std::cout << options.help() << "\n";
      std::exit(1);
    }
  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << "\n";
    std::exit(1);
  }

  try {
    check_arg_file_exists(vm, "xclbin");
    check_arg_file_exists(vm, "instr");
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
  }
}

// --------------------------------------------------------------------------
// AIE Specifics
// --------------------------------------------------------------------------

std::vector<uint32_t> test_utils::load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

template <typename T>
std::vector<T> test_utils::load_binary(std::string binary_path) {
  // Open file in binary mode
  std::ifstream bin_file(binary_path, std::ios::binary);
  if (!bin_file.is_open()) {
    throw std::runtime_error("Unable to open binary file\n");
  }

  // Get the size of the file
  bin_file.seekg(0, std::ios::end);
  std::streamsize size = bin_file.tellg();
  bin_file.seekg(0, std::ios::beg);

  // Check that the file size is a multiple of data size
  if (size % sizeof(T) != 0) {
    throw std::runtime_error("File size is not a multiple of data size\n");
  }

  // Allocate vector and read the binary data
  std::vector<T> bin_v(size / sizeof(T));
  if (!bin_file.read(reinterpret_cast<char *>(bin_v.data()), size)) {
    throw std::runtime_error("Failed to read instruction file\n");
  }
  return bin_v;
}

std::vector<uint8_t> test_utils::load_binary(std::string binary_path) {
  return load_binary<uint8_t>(binary_path);
}

std::vector<uint32_t> test_utils::load_instr_binary(std::string instr_path) {
  return load_binary<uint32_t>(instr_path);
}

#ifdef TEST_UTILS_USE_XRT
// --------------------------------------------------------------------------
// XRT
// --------------------------------------------------------------------------
std::string test_utils::get_kernel_name(const xrt::xclbin &xclbin,
                                        const std::string &name_prefix,
                                        int verbosity) {
  std::string available;
  for (auto &k : xclbin.get_kernels()) {
    auto name = k.get_name();
    if (verbosity >= 1)
      std::cout << "Name: " << name << std::endl;
    if (name.rfind(name_prefix, 0) == 0)
      return name;
    if (!available.empty())
      available += ", ";
    available += name;
  }
  throw std::runtime_error(
      "No kernel in the xclbin starts with '" + name_prefix +
      "'. Available kernels: " + (available.empty() ? "(none)" : available));
}

void test_utils::init_xrt_load_kernel(xrt::device &device, xrt::kernel &kernel,
                                      int verbosity, std::string xclbinFileName,
                                      std::string kernelNameInXclbin) {
  // Get a device handle
  unsigned int device_index = 0;
  device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << xclbinFileName << "\n";
  auto xclbin = xrt::xclbin(xclbinFileName);

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << kernelNameInXclbin << "\n";

  // Get the kernel from the xclbin
  auto kernelName = get_kernel_name(xclbin, kernelNameInXclbin, verbosity);

  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << xclbinFileName << "\n";

  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  kernel = xrt::kernel(context, kernelName);

  return;
}

#endif // TEST_UTILS_USE_XRT

// --------------------------------------------------------------------------
// Matrix / Float / Math
// --------------------------------------------------------------------------

// nearly_equal function adapted from Stack Overflow, License CC BY-SA 4.0
// Original author: P-Gn
// Source: https://stackoverflow.com/a/32334103
bool test_utils::nearly_equal(float a, float b, float epsilon, float abs_th)
// those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<float>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b)
    return true;

  auto diff = std::abs(a - b);
  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  // or even faster: std::min(std::abs(a + b),
  // std::numeric_limits<float>::max()); keeping this commented out until I
  // update figures below
  return diff < std::max(abs_th, epsilon * norm);
}

// --------------------------------------------------------------------------
// Tracing
// --------------------------------------------------------------------------
void test_utils::write_out_trace(char *traceOutPtr, size_t trace_size,
                                 std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (int i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

void test_utils::print_test_banner(const char *test_name, const char *message) {

  // Get current time
  time_t now = time(0);
  tm *ltm = localtime(&now);

  // Print banner
  std::cout << "Running " << test_name << " at " << 1900 + ltm->tm_year << "-"
            << 1 + ltm->tm_mon << "-" << ltm->tm_mday << " " << std::setw(2)
            << std::setfill('0') << ltm->tm_hour << ":" << std::setw(2)
            << std::setfill('0') << ltm->tm_min << ":" << std::setw(2)
            << std::setfill('0') << ltm->tm_sec << std::endl;
  std::cout << message << std::endl;
}

// The following functions use mlir_aie_sim_read32/write32 from
// memory_allocator.h and are only available in simulation builds.
#ifndef TEST_UTILS_XRT_ONLY

// TODO: Add a timeout
void test_utils::wait_on_lock(uint64_t lock_addr, uint32_t lock_value) {
  while (1) {
    uint32_t l = mlir_aie_sim_read32(lock_addr);
    if (l >= lock_value)
      break;
  }
}

// Function to print each field in a 32-bit register
void test_utils::print_dma_status_register_fields(const char *reg_name,
                                                  uint32_t reg_value) {
  printf("\n%s register Value: 0x%08X\n\n", reg_name, reg_value);

  // Extract fields using bit masks and shifts
  uint8_t cur_bd = (reg_value >> 25) & 0xF;                    // 28:25 - 4 bits
  uint8_t task_queue_size = (reg_value >> 20) & 0x1F;          // 24:20 - 5 bits
  uint8_t channel_running = (reg_value >> 19) & 0x1;           // Bit 19 - 1 bit
  uint8_t task_queue_overflow = (reg_value >> 18) & 0x1;       // Bit 18 - 1 bit
  uint8_t error_fot_bds_per_task = (reg_value >> 13) & 0x1;    // Bit 13 - 1 bit
  uint8_t error_fot_length_exceeded = (reg_value >> 12) & 0x1; // Bit 12 - 1 bit
  uint8_t stalled_tct_fifo_full = (reg_value >> 5) & 0x1;      // Bit 5 - 1 bit
  uint8_t stalled_stream_starvation = (reg_value >> 4) & 0x1;  // Bit 4 - 1 bit
  uint8_t stalled_lock_rel = (reg_value >> 3) & 0x1;           // Bit 3 - 1 bit
  uint8_t stalled_lock_acq = (reg_value >> 2) & 0x1;           // Bit 2 - 1 bit
  uint8_t status = reg_value & 0x3;                            // 1:0 - 2 bits

  // Print fields
  printf("Field Breakdown:\n");
  printf("Cur_BD (28:25):                   0x%X\n", cur_bd);
  printf("Task_Queue_Size (24:20):          0x%X\n", task_queue_size);
  printf("Channel_Running (19):             %u\n", channel_running);
  printf("Task_Queue_Overflow (18):         %u\n", task_queue_overflow);
  printf("Error_FoT_BDs_per_Task (13):      %u\n", error_fot_bds_per_task);
  printf("Error_FoT_Length_Exceeded (12):   %u\n", error_fot_length_exceeded);
  printf("Stalled_TCT_or_Count_FIFO_Full(5):%u\n", stalled_tct_fifo_full);
  printf("Stalled_Stream_Starvation (4):    %u\n", stalled_stream_starvation);
  printf("Stalled_Lock_Rel (3):             %u\n", stalled_lock_rel);
  printf("Stalled_Lock_Acq (2):             %u\n", stalled_lock_acq);
  printf("Status (1:0):                     0x%X ", status);

  switch (status) {
  case 0:
    printf("IDLE\n");
    break;
  case 1:
    printf("STARTING\n");
    break;
  case 2:
    printf("RUNNING\n");
    break;
  default:
    printf("UNKNOWN\n");
    break;
  }
  printf("\n");
}

void test_utils::print_shim_tile_status(uint64_t col) {
  uint64_t row = 0;
  uint64_t shim_tile_s2mm_status_0 =
      ((col << 25) | (row << 20)) + 0x18570; // 0x00009ED0;
  uint64_t shim_tile_s2mm_status_1 =
      ((col << 25) | (row << 20)) + 0x58579; // 0x00009ED4;
  uint64_t shim_tile_mm2s_status_0 =
      ((col << 25) | (row << 20)) + 0x18580; // 0x00009F00;
  uint64_t shim_tile_mm2s_status_1 =
      ((col << 25) | (row << 20)) + 0x18584; // 0x00009F04;

  uint32_t s2mm_status_0 = mlir_aie_sim_read32(shim_tile_s2mm_status_0);
  uint32_t s2mm_status_1 = mlir_aie_sim_read32(shim_tile_s2mm_status_1);

  uint32_t mm2s_status_0 = mlir_aie_sim_read32(shim_tile_mm2s_status_0);
  uint32_t mm2s_status_1 = mlir_aie_sim_read32(shim_tile_mm2s_status_1);

  test_utils::print_dma_status_register_fields("Shim tile S2MM0",
                                               s2mm_status_0);
  test_utils::print_dma_status_register_fields("Shim tile MM2S0",
                                               mm2s_status_0);
  printf("s2mm_status_0 = 0x%08x, s2mm_status_1 = 0x%08x, mm2s_status_0 = "
         "0x%08x, mm2s_status_1 = 0x%08x\n",
         s2mm_status_0, s2mm_status_1, mm2s_status_0, mm2s_status_1);

  for (uint32_t i = 0; i < 6; i++) {
    uint64_t shim_tile_lock_addr =
        ((col << 25) | (row << 20)) + 0x0010000 + (i * 0x0010);
    uint32_t lock = mlir_aie_sim_read32(shim_tile_lock_addr);
    printf("lock[%d] = 0x%08x\n", i, lock);
  }
}

void test_utils::print_core_tile_status(
    uint64_t col, uint64_t row, uint32_t num_locks, uint32_t addr_offset_0,
    uint32_t addr_len_0, uint32_t addr_offset_1, uint32_t addr_len_1) {

  uint64_t core_tile_status = ((col << 25) | ((row + 4) << 20)) + 0x000A3004;
  uint32_t reg_value = mlir_aie_sim_read32(core_tile_status);
  printf("Core [%ld, %ld] status = 0x%08x\n", col, row, reg_value);

  uint64_t timer_low_addr = ((col << 25) | ((row + 4) << 20)) + 0x000A0120;
  printf("Core [%ld, %ld] timer_lo = 0x%08x\n", col, row,
         mlir_aie_sim_read32(timer_low_addr));
  // Extract fields using bit masks and shifts
  uint8_t core_processor_bus_stall = (reg_value >> 21) & 0x1; // 21 - 1 bit
  uint8_t core_done = (reg_value >> 20) & 0x1;                // 20 - 1 bit
  uint8_t error_halt = (reg_value >> 19) & 0x1;               // 19 - 1 bit
  uint8_t ecc_scrubbing_stall = (reg_value >> 18) & 0x1;      // 18 - 1 bit
  uint8_t ecc_error_stall = (reg_value >> 17) & 0x1;          // 17 - 1 bit
  uint8_t debug_halt = (reg_value >> 16) & 0x1;               // 16 - 1 bit
  uint8_t cascade_stall_mcd = (reg_value >> 15) & 0x1;        // 15 - 1 bit
  uint8_t cascade_stall_scd = (reg_value >> 14) & 0x1;        // 14 - 1 bit
  uint8_t stream_stall_ms0 = (reg_value >> 12) & 0x1;         // 12 - 1 bit
  uint8_t stream_stall_ss1 = (reg_value >> 11) & 0x1;         // 11 - 1 bit
  uint8_t stream_stall_ss0 = (reg_value >> 10) & 0x1;         // 10 - 1 bit
  uint8_t lock_stall_e = (reg_value >> 9) & 0x1;              // 9 - 1 bit
  uint8_t lock_stall_n = (reg_value >> 8) & 0x1;              // 8 - 1 bit
  uint8_t lock_stall_w = (reg_value >> 7) & 0x1;              // 7 - 1 bit
  uint8_t lock_stall_s = (reg_value >> 6) & 0x1;              // 6 - 1 bit
  uint8_t memory_stall_e = (reg_value >> 5) & 0x1;            // 5 - 1 bit
  uint8_t memory_stall_n = (reg_value >> 4) & 0x1;            // 4 - 1 bit
  uint8_t memory_stall_w = (reg_value >> 3) & 0x1;            // 3 - 1 bit
  uint8_t memory_stall_s = (reg_value >> 2) & 0x1;            // 2 - 1 bit
  uint8_t reset = (reg_value >> 1) & 0x1;                     // 1 - 1 bit
  uint8_t enable = reg_value & 0x1;                           // 0 - 1 bit

  // Print non zero fields
  printf("None-zero field breakdown:\n");
  if (core_processor_bus_stall)
    printf("Core_Processor_Bus_Stall (21): %u\n", core_processor_bus_stall);
  if (core_done)
    printf("Core_Done (20):                %u\n", core_done);
  if (error_halt)
    printf("Error_Halt (19):               %u\n", error_halt);
  if (ecc_scrubbing_stall)
    printf("ECC_Scrubbing_Stall (18):      %u\n", ecc_scrubbing_stall);
  if (ecc_error_stall)
    printf("ECC_Error_Stall (17):          %u\n", ecc_error_stall);
  if (debug_halt)
    printf("Debug_Halt (16):               %u\n", debug_halt);
  if (cascade_stall_mcd)
    printf("Cascade_Stall_MCD (15):        %u\n", cascade_stall_mcd);
  if (cascade_stall_scd)
    printf("Cascade_Stall_SCD (14):        %u\n", cascade_stall_scd);
  if (stream_stall_ms0)
    printf("Stream_Stall_MS0 (12):         %u\n", stream_stall_ms0);
  if (stream_stall_ss1)
    printf("Stream_Stall_SS1 (11):         %u\n", stream_stall_ss1);
  if (stream_stall_ss0)
    printf("Stream_Stall_SS0 (10):         %u\n", stream_stall_ss0);
  if (lock_stall_e)
    printf("Lock_Stall_E (9):              %u\n", lock_stall_e);
  if (lock_stall_n)
    printf("Lock_Stall_N (8):              %u\n", lock_stall_n);
  if (lock_stall_w)
    printf("Lock_Stall_W (7):              %u\n", lock_stall_w);
  if (lock_stall_s)
    printf("Lock_Stall_S (6):              %u\n", lock_stall_s);
  if (memory_stall_e)
    printf("Memory_Stall_E (5):            %u\n", memory_stall_e);
  if (reset)
    printf("Reset (1):                     %u\n", reset);
  if (enable)
    printf("Enable (0):                    %u\n", enable);
  printf("\n");

  for (int i = 0; i < num_locks; i++) {
    uint64_t core_tile_lock_addr =
        ((col << 25) | ((row + 4) << 20)) + 0x00AE000 + (i * 0x0010);
    printf("Lock[%d] = 0x%08x\n", i, mlir_aie_sim_read32(core_tile_lock_addr));
  }

  if (addr_len_0) {
    for (int i = 0; i < addr_len_0; i++) {
      uint64_t addr =
          ((col << 25) | ((row + 4) << 20)) + addr_offset_0 + (i * 4);
      printf("0x%08x: 0x%08x\n", addr, mlir_aie_sim_read32(addr));
    }
  }

  if (addr_len_1) {
    for (int i = 0; i < addr_len_1; i++) {
      uint64_t addr =
          ((col << 25) | ((row + 4) << 20)) + addr_offset_1 + (i * 4);
      printf("0x%08x: 0x%08x\n", addr, mlir_aie_sim_read32(addr));
    }
  }

  uint64_t core_tile_s2mm_status_0 =
      ((col << 25) | ((row + 4) << 20)) + 0x000ACF00;
  uint64_t core_tile_s2mm_status_1 =
      ((col << 25) | ((row + 4) << 20)) + 0x000ACF04;
  uint64_t core_tile_mm2s_status_0 =
      ((col << 25) | ((row + 4) << 20)) + 0x000ACF10;

  uint32_t s2mm_status_0 = mlir_aie_sim_read32(core_tile_s2mm_status_0);
  uint32_t s2mm_status_1 = mlir_aie_sim_read32(core_tile_s2mm_status_1);

  uint32_t mm2s_status_0 = mlir_aie_sim_read32(core_tile_mm2s_status_0);

  test_utils::print_dma_status_register_fields("Core tile S2MM0",
                                               s2mm_status_0);
  test_utils::print_dma_status_register_fields("Core tile S2MM1",
                                               s2mm_status_1);
  test_utils::print_dma_status_register_fields("Core tile MM2S0",
                                               mm2s_status_0);
  printf("s2mm_status_0 = 0x%08x, s2mm_status_1 = 0x%08x, mm2s_status_0 = "
         "0x%08x\n",
         s2mm_status_0, s2mm_status_1, mm2s_status_0);
  printf("Core [%ld, %ld] timer_lo = 0x%08x\n", col, row,
         mlir_aie_sim_read32(timer_low_addr));
}

void test_utils::wait_for_cycles(uint64_t col, uint64_t row, uint32_t cycles) {
  uint64_t timer_lo_addr = ((col << 25) | ((row + 4) << 20)) + 0x00A0120;
  uint32_t start = mlir_aie_sim_read32(timer_lo_addr);

  uint32_t end = mlir_aie_sim_read32(timer_lo_addr);

  while ((end - start) < cycles) {
    end = mlir_aie_sim_read32(timer_lo_addr);
  }
}

// Unpack transaction binary and write it to the aie array simulation
void test_utils::write_transaction_binary(unsigned char *data,
                                          uint64_t nbytes) {
  assert(nbytes >= 12);
  uint32_t major = data[0];
  uint32_t minor = data[1];
  uint32_t num_cols = data[4];

  uint32_t num_ops, txn_size;
  std::memcpy(&num_ops, &data[8], 4);
  std::memcpy(&txn_size, &data[12], 4);

  uint32_t i = 16;

  uint64_t addr = 0;
  uint32_t value = 0;
  uint32_t size = 0;
  uint32_t mask = 0;
  const uint32_t *data_ptr = nullptr;

  if (major == 0 && minor == 1) {
    while (i < nbytes) {
      uint8_t opc = data[i];
      if (opc == 0x00) {
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&size, &data[i + 20], 4);
        addr = static_cast<uint64_t>(addr1) << 32 | addr0;
        i += size;
        mlir_aie_sim_write32(addr, value);
      } else if (opc == 0x01) {
        std::memcpy(&addr, &data[i + 8], 4);
        std::memcpy(&size, &data[i + 12], 4);
        data_ptr = reinterpret_cast<const uint32_t *>(data + i + 16);
        i += size;
        size = size - 16;
        for (int j = 0; j < size / 4; j++)
          mlir_aie_sim_write32(addr + (4 * j), data_ptr[j]);
      } else if (opc == 0x03) {
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&mask, &data[i + 20], 4);
        std::memcpy(&size, &data[i + 24], 4);
        addr = static_cast<uint64_t>(addr1) << 32 | addr0;
        uint32_t r = mlir_aie_sim_read32(addr);
        uint32_t w = (r & ~mask) | (value & mask);
        mlir_aie_sim_write32(addr, w);
        i += size;
      } else {
        assert(0 && "unhandled transaction binary opcode");
      }
    }
  } else if (major == 1 && minor == 0) {
    while (i < nbytes) {
      uint8_t opc = data[i];
      if (opc == 0x00) {
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        i += 12;
        mlir_aie_sim_write32(addr, value);
      } else if (opc == 0x01) {
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&size, &data[i + 8], 4);
        data_ptr = reinterpret_cast<const uint32_t *>(data + i + 12);
        i += size;
        size = size - 12;
        for (int j = 0; j < size / 4; j++)
          mlir_aie_sim_write32(addr + (4 * j), data_ptr[j]);
      } else if (opc == 0x03) {
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        std::memcpy(&mask, &data[i + 12], 4);
        uint32_t r = mlir_aie_sim_read32(addr);
        uint32_t w = (r & ~mask) | (value & mask);
        mlir_aie_sim_write32(addr, w);
        i += 16;
      } else {
        assert(0 && "unhandled transaction binary opcode");
      }
    }
  } else {
    assert(0 && "unhandled transaction binary version");
  }
}

#endif // TEST_UTILS_XRT_ONLY

/* Tensor and Matrix processing utilities
These include CPU reference implementations for tensor operators
and tensor equality tests
*/

void matrix_utils::pack_int8_to_uint32(uint32_t *sim_mem, const int8_t *input,
                                       size_t num_bytes) {

  size_t num_words = (num_bytes + 3) / 4; // Round up to nearest 4 bytes
  for (size_t i = 0; i < num_words; ++i) {
    uint32_t word = 0;
    for (int j = 0; j < 4; ++j) {
      size_t idx = i * 4 + j;
      if (idx < num_bytes) {
        // Cast to uint8_t to avoid sign extension
        word |= ((uint32_t)((uint8_t)input[idx])) << (8 * j);
      }
    }

    sim_mem[i] = word;
  }
}
