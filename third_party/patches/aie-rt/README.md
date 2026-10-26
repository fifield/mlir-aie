<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Vendored aie-rt patches

`third_party/aie-rt` is pinned to the internal AIE4 fork
[jefff/aie-rt](https://gitenterprise.xilinx.com/jefff/aie-rt), based on
`aie4-dev` with compatible fixes merged from internal
[ai-engine/aie-rt](https://gitenterprise.xilinx.com/ai-engine/aie-rt)
`main_aig` at `73e6472b`. This preserves the fork's AIE4Z support and
AIE4 DMA bounds-check fixes. The patches in this directory carry additional
functionality mlir-aie depends on, applied automatically at CMake configure
time (see
`runtime_lib/xaiengine/aiert.cmake`). Each patch has a public upstream PR, and
[#3851](https://github.com/Xilinx/mlir-aie/issues/3851) tracks what is needed
to drop them.

- `0001-cdo-sim-defork-fixes.patch`: works around aie-rt's `main_rts.h`
  dependency on a Vitis-only header by replacing the include in `xaie_sim.c`
  with local forward declarations, and drops an `UCHAR_MAX` initializer on
  `XAIE_IO_CUSTOM_OP_MAX` in `xaie_txn.h` (minor build/warning fix). Our
  pinned commit already carries an equivalent defork fix for `xaie_cdo.c`'s
  `cdo_rts.h` dependency and no longer has the `RscArrPerTile` leak in
  `xaie_io_common.c` natively, so this patch no longer needs to touch either
  file — re-check both if this patch stops applying after a submodule bump.
  Upstream: [Xilinx/aie-rt#25](https://github.com/Xilinx/aie-rt/pull/25).
- `0002-elfloader-zero-bss-gap.patch`: `_XAie_LoadDataMemSection` wrote
  `p_memsz` bytes from a buffer (`ElfMem + p_offset`) holding only `p_filesz`
  valid ones, substituting a zeroed buffer only when `p_filesz == 0`. That
  covers a pure-data segment and a pure-bss segment, but not the ordinary mixed
  `.data`+`.bss` segment (`0 < p_filesz < p_memsz`) that any link with both
  initialised and zero-initialised globals produces -- for those it read past
  the segment's file contents and wrote the following ELF bytes (`.comment`,
  `.symtab`) into tile data memory, where zero-initialised statics live. The
  System V gABI defines those trailing bytes to hold zero; that is how `.bss` is
  represented in a loadable segment. The patch allocates a `p_memsz` zeroed
  buffer whenever `p_memsz > p_filesz` and copies the initialised prefix into
  it. Regression test: `test/aiecc/bss_zero_init.mlir`. Upstream:
  [Xilinx/aie-rt#27](https://github.com/Xilinx/aie-rt/pull/27).
- `0003-remove-dead-blockwrite32-append-fns.patch`: deletes
  `_XAie_AppendBlockWrite32`/`_opt` in `xaie_txn.c`, two `static inline`
  functions left over from an upstream TXN-serialization refactor that no
  caller references anymore (block-write commands now go through
  `_XAie_AppendBWToTxnBuff`/`_XAie_AppendBWToBlockwriteBuff` instead). GCC
  doesn't flag this, but clang's `-Werror=unused` (used in CI) does. Upstream:
  [Xilinx/aie-rt#31](https://github.com/Xilinx/aie-rt/pull/31).
- `0004-fix-aie1-tiledma-intrleavecount-underflow.patch`: fixes an unsigned
  underflow in the AIE1 tile-DMA BD writer. `_XAie_TileDmaWriteBd` guards each
  BD field with `_XAie_CheckPrecisionExceeds(Lsb, _XAie_MaxBitsNeeded(v), 32)`,
  and passes `IntrleaveCount - 1U` for the interleave-count field. Interleaving
  is off by default, so `IntrleaveCount` is 0 and the subtraction wraps to
  `0xFFFFFFFF`, making `_XAie_MaxBitsNeeded` return 32 and the guard trip for
  any nonzero `Lsb`. The BD write then bails out with `XAIE_ERR` before
  programming a single word, so every AIE1 tile DMA silently goes
  unconfigured. The neighbouring `XAie_SetField` that actually writes the
  field masks the wrapped value, which is why only the new check is affected.
  Clamp the checked value to 0 when interleaving is disabled. Upstream:
  [Xilinx/aie-rt#32](https://github.com/Xilinx/aie-rt/pull/32).
- `0005-fix-txn-misaligned-header-writes.patch`: the `_XAie_Append*`
  serializers in `xaie_txn.c` cast the transaction buffer cursor to a header
  struct pointer and assign through it. Earlier ops advance the cursor by
  4-byte multiples, so headers with a `u64` field (`Write32`, `MaskWrite32`,
  `MaskPoll32`, `MaskPollBusy32`, `LoadPdi`, `CreateScratchpad`) land
  misaligned, which is undefined behaviour (UBSan `alignment`). The patch
  fills a zeroed stack header and `memcpy`s it into place, the same fix
  upstream applied to `_XAie_AppendDDRPatch_opt` in aa07f57f. The buffer is
  already zero-initialised, so the serialized bytes are unchanged. Upstream:
  [Xilinx/aie-rt#33](https://github.com/Xilinx/aie-rt/pull/33).

Note: `0003` and `0005` are maintained against this branch's aie-rt pin, where
the `_XAie_Append*` serializers take `(Cmd, TxnPtr)` rather than upstream's
`(DevInst, Cmd, TxnPtr)` and write `OpHdr.Col`/`Row` as 0; re-check them if
the submodule is bumped. All five patches apply unchanged to the merged
`a696248f` pin, and a second configure skips them successfully. `0003` is
generated with two lines of context
(`git diff -U2`) so its leading context does not overlap the
`_XAie_AppendMaskPollBusy32` body that `0005` rewrites; otherwise the
reverse-apply idempotency check in `aiert.cmake` fails on re-configure.
