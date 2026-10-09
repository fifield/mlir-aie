// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Runs each explicit-mode SRS variant and compares its output with the golden
// written by srs_modes.py, byte for byte.
//
//   ./test.exe input.bin <elf> <golden> [<elf> <golden> ...]

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match srs_modes.py.
constexpr int SHIFTS[] = {0, 1, 12, 32, 59};
constexpr int N_SHIFTS = sizeof(SHIFTS) / sizeof(SHIFTS[0]);
constexpr int IN_WORDS = 256;    // i32
constexpr int SHIFT_WORDS = 512; // i16 words per shift
constexpr int TAIL = SHIFT_WORDS * N_SHIFTS;
constexpr int OUT_WORDS = TAIL + 32; // i16
constexpr size_t IN_BYTES = IN_WORDS * sizeof(int32_t);
constexpr size_t OUT_BYTES = OUT_WORDS * sizeof(int16_t);

static std::vector<char> readFile(const std::string &path, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  if (!f.good() && !f.eof()) {
    std::cerr << "cannot read " << path << "\n";
    std::exit(2);
  }
  if (data.size() != bytes) {
    std::cerr << path << ": " << data.size() << " bytes, expected " << bytes
              << "\n";
    std::exit(2);
  }
  return data;
}

// Describe output word `w` as (shift, kind, lane) and its input value.
static void describe(int w, const int32_t *in) {
  if (w >= TAIL) {
    std::cout << "tail word " << w - TAIL;
    return;
  }
  int k = w / SHIFT_WORDS, r = w % SHIFT_WORDS;
  const char *kind;
  int lane0, lanes;
  if (r < 128) {
    kind = "i8s";
    lane0 = 2 * r;
    lanes = 2;
  } else if (r < 256) {
    kind = "i8u";
    lane0 = 2 * (r - 128);
    lanes = 2;
  } else {
    kind = "i16s";
    lane0 = r - 256;
    lanes = 1;
  }
  std::cout << "shift " << SHIFTS[k] << " " << kind << " lane " << lane0;
  if (lanes == 2)
    std::cout << "/" << lane0 + 1;
  std::cout << " input " << in[lane0];
  if (lanes == 2)
    std::cout << "/" << in[lane0 + 1];
}

int main(int argc, char **argv) {
  if (argc < 4 || (argc - 2) % 2 != 0) {
    std::cerr << "usage: " << argv[0]
              << " input.bin <elf> <golden> [<elf> <golden> ...]\n";
    return 2;
  }
  std::vector<char> input = readFile(argv[1], IN_BYTES);
  const int32_t *in = reinterpret_cast<const int32_t *>(input.data());

  auto device = xrt::device(0);
  bool allPass = true;
  for (int i = 2; i < argc; i += 2) {
    std::string elfPath = argv[i];
    std::vector<char> gold = readFile(argv[i + 1], OUT_BYTES);
    std::string variant = elfPath.substr(0, elfPath.rfind(".elf"));

    xrt::elf ctxElf{elfPath};
    xrt::hw_context context = xrt::hw_context(device, ctxElf);
    auto kernel = xrt::ext::kernel(context, "main:sequence");

    xrt::bo boIn = xrt::ext::bo{device, IN_BYTES};
    xrt::bo boOut = xrt::ext::bo{device, OUT_BYTES};
    memcpy(boIn.map<char *>(), input.data(), IN_BYTES);
    // The core writes every word, including the zero-padded tail.
    memset(boOut.map<char *>(), 0, OUT_BYTES);
    boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    kernel(boIn, boOut).wait2();

    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const int16_t *out = boOut.map<const int16_t *>();
    const int16_t *want = reinterpret_cast<const int16_t *>(gold.data());

    int errors = 0;
    for (int w = 0; w < OUT_WORDS; w++) {
      if (out[w] == want[w])
        continue;
      if (errors < 16) {
        std::cout << "  MISMATCH word " << w << " (";
        describe(w, in);
        std::cout << "): got 0x" << std::hex << uint16_t(out[w]) << " want 0x"
                  << uint16_t(want[w]) << std::dec << "\n";
      }
      errors++;
    }
    if (errors == 0) {
      std::cout << variant << ": PASS\n";
    } else {
      std::cout << variant << ": FAIL (" << errors << " / " << OUT_WORDS
                << " words)\n";
      allPass = false;
    }
  }
  std::cout << (allPass ? "PASS" : "FAIL") << std::endl;
  return allPass ? 0 : 1;
}
