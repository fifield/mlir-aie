//===- cert_debug_buffer.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CERT_REG_TELEMETRY_DEBUG_BUFFER_H
#define CERT_REG_TELEMETRY_DEBUG_BUFFER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>

// XRT exports these entry points from libxrt_coreutil, but their declaration
// lives in the non-installed core/common/api/bo_int.h. Keep this list and both
// signatures synchronized with the XRT header; a mismatch fails at link time.
namespace xrt_core {
class hwctx_handle;
namespace bo_int {
enum class use_type {
  debug = 1,
  kmd = 2,
  dtrace = 3,
  log = 4,
  debug_queue = 5,
  uc_debug = 6,
  host_only = 7,
  instruction = 8,
  preemption = 9,
  scratch_pad = 10,
  ctrl_scratch_pad = 11,
  pdi = 12,
  ctrlpkt = 13
};
xrt::bo create_bo(const xrt::hw_context &hwctx, size_t sz, use_type type);
void config_bo(const xrt::bo &bo, const std::map<uint32_t, size_t> &buf_sizes,
               const xrt_core::hwctx_handle *ctx_handle);
} // namespace bo_int
} // namespace xrt_core

namespace cert_example {

class DebugBuffer {
public:
  DebugBuffer(const xrt::hw_context &context, size_t words, uint32_t poison)
      : wordCount(words), poisonValue(poison),
        bo(xrt_core::bo_int::create_bo(context, words * sizeof(uint32_t),
                                       xrt_core::bo_int::use_type::uc_debug)),
        data(bo.map<uint32_t *>()) {
    std::fill_n(data, wordCount, poisonValue);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void configure(uint32_t ucIndex) {
    const std::map<uint32_t, size_t> segments{
        {ucIndex, wordCount * sizeof(uint32_t)}};
    xrt_core::bo_int::config_bo(bo, segments, nullptr);
  }

  void refresh() { bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); }

  size_t size() const { return wordCount; }
  uint32_t poison() const { return poisonValue; }
  uint32_t operator[](size_t index) const { return data[index]; }

  size_t firstChanged(size_t first) const {
    const uint32_t *it =
        std::find_if(data + first, data + wordCount,
                     [this](uint32_t word) { return word != poisonValue; });
    return static_cast<size_t>(it - data);
  }

private:
  size_t wordCount;
  uint32_t poisonValue;
  xrt::bo bo;
  uint32_t *data;
};

} // namespace cert_example

#endif
