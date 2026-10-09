// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Override aiecc's default main() and poison crRnd/crSat before core entry,
// so tests verify the core's mode setup independently of loader defaults.
int get_rnd(void) { return __builtin_aie4_get_ctrl_reg(46); }
int get_sat(void) { return __builtin_aie4_get_ctrl_reg(45); }

void core_0_2(void);
int main(void) {
  __builtin_aie4_set_ctrl_reg(46, 13);
  __builtin_aie4_set_ctrl_reg(45, 3);
  core_0_2();
  return 0;
}
