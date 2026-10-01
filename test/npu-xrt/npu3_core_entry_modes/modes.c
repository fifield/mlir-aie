// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Probe helpers linked into the test cores: read and write the AIE4 rounding
// controls (ids from Peano aie4/aie4_defines.h: crVecRnd 24, crRnd 46), and a
// main() that poisons both before the core runs.
int get_vec_rnd(void) { return __builtin_aie4_get_ctrl_reg(24); }
int get_rnd(void) { return __builtin_aie4_get_ctrl_reg(46); }
void set_modes(int vec_rnd, int rnd) {
  __builtin_aie4_set_ctrl_reg(24, vec_rnd);
  __builtin_aie4_set_ctrl_reg(46, rnd);
}

// Overrides aiecc's linker-script PROVIDE(main = core_0_2): poison both
// rounding controls to floor after the loader has run, then enter the core.
// The core's own entry writes must restore round-to-nearest-even.
void core_0_2(void);
int main(void) {
  __builtin_aie4_set_ctrl_reg(24, 0);
  __builtin_aie4_set_ctrl_reg(46, 0);
  core_0_2();
  return 0;
}
