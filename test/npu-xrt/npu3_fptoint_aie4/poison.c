// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// main() overrides aiecc's linker-script PROVIDE(main = core_0_2): it sets
// crRnd (46) to 12, round to nearest even, before entering the core, so the
// results depend only on the compiler's own mode setup.
int get_rnd(void) { return __builtin_aie4_get_ctrl_reg(46); }

void core_0_2(void);
int main(void) {
  __builtin_aie4_set_ctrl_reg(46, 12);
  core_0_2();
  return 0;
}
