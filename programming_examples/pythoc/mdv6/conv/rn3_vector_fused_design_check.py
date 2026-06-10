#!/usr/bin/env python3
"""Executable sizing/target checks for the next rn3 vector-fused design.

This is not a hardware test. It captures the production constraints discovered
while investigating rn3 dispatch reduction:

- The fast existing mc_*_rn3 path uses conv3x3_fused_packed_bf16, an mmul<4,8,8>
  PythoC kernel from kernels/rep_elan_bf16_pythoc.py.
- A naive fused pair that materialises the full 10x10x48 intermediate and holds
  the full conv1 48->48 weight block in one worker exceeds AIE tile L1.
- A production fused-pair kernel therefore needs mid-channel blocking and a
  partial-accumulating conv2 stage, not just two calls to the existing conv3x3
  kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

L1_BYTES = 64 * 1024
# Keep a conservative reserve for stack, objectfifo bookkeeping, barriers, RTPs,
# and compiler-introduced spill/local state. Previous scalar rn3pair used 4096B
# worker stacks; leave more than that because vector helpers are larger.
RESERVE_BYTES = 8 * 1024
USABLE_BYTES = L1_BYTES - RESERVE_BYTES


@dataclass(frozen=True)
class Footprint:
    mid_block: int
    input_b: int
    scratch_b: int
    out_b: int
    w1_b: int
    w2_b: int

    @property
    def persistent_b(self) -> int:
        return self.input_b + self.scratch_b + self.out_b

    @property
    def phase1_peak_b(self) -> int:
        return self.persistent_b + self.w1_b

    @property
    def phase2_peak_b(self) -> int:
        return self.persistent_b + self.w2_b

    @property
    def both_weights_b(self) -> int:
        return self.persistent_b + self.w1_b + self.w2_b


def bf16_bytes(n: int) -> int:
    return 2 * n


def footprint(tile: int = 8, ic: int = 48, mid_block: int = 16, oc: int = 16) -> Footprint:
    input_e = (tile + 4) * (tile + 4) * ic
    scratch_e = (tile + 2) * (tile + 2) * mid_block
    out_e = tile * tile * oc
    w1_e = mid_block * ic * 9 + 2 * mid_block
    w2_e = oc * mid_block * 9 + 2 * oc
    return Footprint(
        mid_block=mid_block,
        input_b=bf16_bytes(input_e),
        scratch_b=bf16_bytes(scratch_e),
        out_b=bf16_bytes(out_e),
        w1_b=bf16_bytes(w1_e),
        w2_b=bf16_bytes(w2_e),
    )


def main() -> int:
    print(f"L1_BYTES={L1_BYTES}")
    print(f"RESERVE_BYTES={RESERVE_BYTES}")
    print(f"USABLE_BYTES={USABLE_BYTES}")
    print()
    print("mid_block  persistent  +w1 peak  +w2 peak  both-wts  phase-fit")
    print("---------  ----------  --------  --------  --------  ---------")
    feasible = []
    for mb in (48, 32, 24, 16, 8):
        fp = footprint(mid_block=mb)
        phase_fit = fp.phase1_peak_b < USABLE_BYTES and fp.phase2_peak_b < USABLE_BYTES
        if phase_fit:
            feasible.append(mb)
        print(
            f"{mb:9d}  {fp.persistent_b:10d}  {fp.phase1_peak_b:8d}  "
            f"{fp.phase2_peak_b:8d}  {fp.both_weights_b:8d}  {str(phase_fit):>9}"
        )
    print()
    print("Recommended first vector spike: mid_block=16")
    print("  - phase1 input+scratch+out+w1 fits below conservative usable L1")
    print("  - phase2 input+scratch+out+w2 fits below conservative usable L1")
    print("  - both weights together are not required and should not be live together")
    print("  - conv2 must accumulate across 3 mid blocks before BN+SiLU")
    print()
    print("Production baseline to beat/approach:")
    print("  current 24x16/6-tile two-conv mc_re6_rn3 benchmark: launches=2, npu_ms=7.52, wall_ms=152.57")
    print("  current make profile warm mc_re6_rn3 bucket: 90.64 ms / 36 calls = 2.52 ms/call")
    assert 16 in feasible, "mid_block=16 should fit the conservative phase-by-phase L1 estimate"
    assert 48 not in feasible, "full mid=48 should remain rejected by the conservative estimate"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
