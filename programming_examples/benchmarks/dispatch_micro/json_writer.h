//===- json_writer.h --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// Tiny no-deps JSON writer for benchmark results. Emits a single JSON object
// per bench invocation. Not a full JSON library: only the shapes used by
// bench.cpp are supported.
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace dispatch_micro {

class JsonWriter {
public:
  explicit JsonWriter(std::ostream &os) : os_(os) { os_ << "{"; }
  ~JsonWriter() { os_ << "}\n"; }

  JsonWriter &str(const char *k, const std::string &v) {
    return str(k, v.c_str());
  }
  JsonWriter &str(const char *k, const char *v) {
    sep();
    os_ << "\"" << k << "\":\"";
    escape(v);
    os_ << "\"";
    return *this;
  }
  JsonWriter &num(const char *k, long long v) {
    sep();
    os_ << "\"" << k << "\":" << v;
    return *this;
  }
  JsonWriter &num(const char *k, int v) {
    return num(k, static_cast<long long>(v));
  }
  JsonWriter &num(const char *k, unsigned v) {
    return num(k, static_cast<long long>(v));
  }
  JsonWriter &num(const char *k, double v) {
    sep();
    os_ << "\"" << k << "\":" << v;
    return *this;
  }
  JsonWriter &boolean(const char *k, bool v) {
    sep();
    os_ << "\"" << k << "\":" << (v ? "true" : "false");
    return *this;
  }
  JsonWriter &array_ns(const char *k, const std::vector<long long> &v) {
    sep();
    os_ << "\"" << k << "\":[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) os_ << ",";
      os_ << v[i];
    }
    os_ << "]";
    return *this;
  }
  // Emit raw, pre-formatted JSON for a value (caller owns syntactic validity).
  JsonWriter &raw(const char *k, const std::string &json) {
    sep();
    os_ << "\"" << k << "\":" << json;
    return *this;
  }

private:
  void sep() {
    if (first_) first_ = false;
    else os_ << ",";
  }
  void escape(const char *s) {
    for (; *s; ++s) {
      char c = *s;
      switch (c) {
        case '"':  os_ << "\\\""; break;
        case '\\': os_ << "\\\\"; break;
        case '\n': os_ << "\\n"; break;
        case '\r': os_ << "\\r"; break;
        case '\t': os_ << "\\t"; break;
        default:
          if (static_cast<unsigned char>(c) < 0x20)
            os_ << "?";
          else
            os_ << c;
      }
    }
  }

  std::ostream &os_;
  bool first_ = true;
};

} // namespace dispatch_micro
