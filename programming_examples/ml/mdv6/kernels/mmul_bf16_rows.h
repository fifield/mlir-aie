// Keep MMUL result-row extraction indices constant on the installed Peano
// AIE2P compiler. The sparse probe demonstrates that extract<8>(runtime_row)
// can copy a negative fourth-row lane into the second row even when the full
// 32-lane MMUL result is correct. Literal-index extraction avoids that lowering.
// This changes no MAC order, bf16 partial rounding, BN or SiLU arithmetic.
#pragma once

namespace mdv6 {
template <typename Fn>
inline __attribute__((always_inline)) void for_each_mmul_row(Fn fn) {
  fn.template operator()<0>();
  fn.template operator()<1>();
  fn.template operator()<2>();
  fn.template operator()<3>();
}
} // namespace mdv6
