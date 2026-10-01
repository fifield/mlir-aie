// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// main() overrides aiecc's linker-script PROVIDE(main = core_0_2): it sets
// crVecRnd (24) and crRnd (46) to floor after the loader has run, then enters
// the core, so the results depend only on the compiler's own mode setup.
int get_rnd(void) { return __builtin_aie4_get_ctrl_reg(46); }

void core_0_2(void);
int main(void) {
  __builtin_aie4_set_ctrl_reg(24, 0);
  __builtin_aie4_set_ctrl_reg(46, 0);
  core_0_2();
  return 0;
}
