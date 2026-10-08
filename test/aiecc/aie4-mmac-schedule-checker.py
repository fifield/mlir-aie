#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check AIE4 MMAC staging, commit and MMAC order in pipeline-timing terms.

Instruction issue order differs from register access order. Model issue
cycles by bundle index and register accesses by Peano's scheduling stages:

  insert  vmov [f_|i4_]za[i], y   writes za at issue + 3
          vmov [f_]za<n>, y      immediate chunk index, same stage
  load    vldb [f_]za[i], [p...]  writes za at issue + 6
  commit  vmov [f_|i4_]zb, za     reads za and writes zb at issue + 7
  mmac    mmac.[fi] ..., zb       reads zb at issue + 2

A read concurrent with a write sees the old value: read-after-write needs
R > W, while write-after-read permits W == R. Thus every tile's inserts
must write before its commit reads, and its MMACs must read after that
commit writes. The next tile must not overwrite staging or active matrix
state before the previous tile finishes reading it. Pipeline stalls do
not change these relative stage timings.

Group inserts by write cycle and MMACs by read cycle using the expected
counts per tile. Chunk identities are not tracked, so compensating swaps
of inserts between tiles can escape detection. Hardware tests complement
these timing and count checks.

Supported control flow is straight-line code or one loop. A backward
jnz/jz/jnzd loop includes its five delay slots; a zero-overhead loop spans
[ls, le] inclusively. Execute the body twice to check back-edge hazards.
Zero-overhead loops require immediate, aligned bounds, a full bundle at
le, and ls/le/lc setup outside the body with at least four bundles before
le. Calls, forward or multiple jumps, and mixed loop forms are rejected.
Ignore code after ret and its five delay slots.

Also reject accumulator (cm*/dm*/bm*) spills or fills through the stack.

Stream pops (vmov[abx] x|y, ss[...]) are counted and their spacing along
the path is reported. --pops-per-tile gates the count per tile: tile g is
the set of staging writes consumed by commit g (prime = tile 0, prefetches
issued before commit g = tile g), and pops are split into consecutive
groups of the listed sizes in issue order. Each tile's pops must issue
before its commit reads za. --inserts-per-tile also takes a list, one
entry per tile under the same grouping. --forbid-lda-s8 rejects scalar
lda.s8 loads anywhere in the function (A rows must use vector loads).
"""

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Self-test of aie4-mmac-schedule-checker.py on synthetic schedules
# (one insert and one MMAC per tile, one bundle per line). good.dis sits on
# all four timing boundaries:
# reads one cycle after the write they depend on (R = W + 1), and writes in
# the same cycle as the read of the old value (W = R). Each other input
# moves one instruction one cycle past a boundary.
#
# RUN: %python %s --write-fixtures %t
# RUN: %python %s %t/good.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 | FileCheck %s --check-prefix=GOOD
# GOOD: ORDER OK: 2 tiles, 2 inserts, 2 MMACs, no accumulator spills
#
# RUN: not %python %s %t/raw_insert.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 2>&1 | FileCheck %s --check-prefix=RAW-INSERT
# RAW-INSERT: error: tile 1: an insert writes za at 9, not before the commit reads it at 9
#
# RUN: not %python %s %t/war_insert.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 2>&1 | FileCheck %s --check-prefix=WAR-INSERT
# WAR-INSERT: error: tile 1: an insert writes za at 7, before commit 0 reads it at 8
#
# RUN: not %python %s %t/raw_mmac.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 2>&1 | FileCheck %s --check-prefix=RAW-MMAC
# RAW-MMAC: error: tile 0: an MMAC reads zb at 8, not after the commit writes it at 8
#
# RUN: not %python %s %t/war_mmac.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 2>&1 | FileCheck %s --check-prefix=WAR-MMAC
# WAR-MMAC: error: tile 0: commit 1 writes zb at 9, before an MMAC reads it at 10
#
# RUN: not %python %s %t/spill.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 2>&1 | FileCheck %s --check-prefix=SPILL
# SPILL: error: accumulator spill/fill at cycle 9: vst{{.*}}cmal0, [sp, #-0x40]
#
# RUN: not %python %s %t/good.dis --inserts-per-tile 1 --mmacs-per-tile 1,1,1 2>&1 | FileCheck %s --check-prefix=COUNT
# COUNT: error: expected 3 commits, found 2
#
# Loops: the path runs through local .L labels, and a single backward
# jump is modelled as the body executed twice, delay slots included. Each
# input ends in another function whose insert must not be counted.
# RUN: %python %s %t/loop_good.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 | FileCheck %s --check-prefix=LOOP-GOOD
# LOOP-GOOD: ORDER OK: 2 tiles, 2 inserts, 0 MMACs, no accumulator spills (loop body checked twice)
#
# The back edge: iteration 2's insert writes before iteration 1's commit reads.
# RUN: not %python %s %t/loop_war.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 2>&1 | FileCheck %s --check-prefix=LOOP-WAR
# LOOP-WAR: error: tile 1: an insert writes za at 13, before commit 0 reads it at 14
#
# A commit in the last delay slot belongs to the body (two commits found).
# RUN: not %python %s %t/loop_delay_slot.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 2>&1 | FileCheck %s --check-prefix=LOOP-DELAY
# LOOP-DELAY: error: tile 1: an insert writes za at 13, before commit 0 reads it at 16
#
# Other control flow is rejected.
# RUN: not %python %s %t/forward_jump.dis --inserts-per-tile 1 --mmacs-per-tile 1 2>&1 | FileCheck %s --check-prefix=FORWARD
# FORWARD: error: unsupported control flow at bundle 1 (jnz{{.*}}#0xc): not a backward jump to a bundle of this function; the check is straight-line or single-loop only
# A zero-overhead loop needs one write each of ls, le and lc.
# RUN: not %python %s %t/hw_loop.dis --inserts-per-tile 1 --mmacs-per-tile 1 2>&1 | FileCheck %s --check-prefix=HWLOOP
# HWLOOP: error: zero-overhead loop: expected one write of le, found 0
#
# Immediate-index inserts (vmov za<n>, y and f_za<n>) are inserts.
# RUN: %python %s %t/imm_index.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 | FileCheck %s --check-prefix=IMM
# IMM: ORDER OK: 2 tiles, 2 inserts, 2 MMACs, no accumulator spills

#
# Zero-overhead loops: the body is [ls, le] with no delay slots,
# checked twice. zol_good sits on the WAR boundary across the back edge;
# zol_war is one bundle shorter, so iteration 2's insert writes too early.
# RUN: %python %s %t/zol_good.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 | FileCheck %s --check-prefix=ZOL-GOOD
# ZOL-GOOD: ORDER OK: 2 tiles, 2 inserts, 0 MMACs, no accumulator spills (loop body checked twice)
# RUN: not %python %s %t/zol_war.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 2>&1 | FileCheck %s --check-prefix=ZOL-WAR
# ZOL-WAR: error: tile 1: an insert writes za at 16, before commit 0 reads it at 17
#
# Validate loop bounds, alignment, setup order and minimum distance.
# RUN: not %python %s %t/zol_unaligned.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 2>&1 | FileCheck %s --check-prefix=ZOL-ALIGN
# ZOL-ALIGN: error: zero-overhead loop: ls = 0x104 is not 256-bit aligned
# RUN: not %python %s %t/zol_lc_first.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 2>&1 | FileCheck %s --check-prefix=ZOL-ORDER
# ZOL-ORDER: error: zero-overhead loop: ls and le must be written before lc
# RUN: not %python %s %t/zol_short.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 2>&1 | FileCheck %s --check-prefix=ZOL-SHORT
# ZOL-SHORT: error: zero-overhead loop: the loop ends 3 bundle(s) after the last ls/le/lc write; at least 4 are required
# RUN: not %python %s %t/zol_compressed_le.dis --inserts-per-tile 1 --mmacs-per-tile 0,0 2>&1 | FileCheck %s --check-prefix=ZOL-LE
# ZOL-LE: error: zero-overhead loop: the bundle at le = 0x1a0 is not a full 256-bit instruction
#
# A real zero-overhead K loop from Peano (see the header of kloop_zol.dis).
# RUN: %python %s %t/kloop_zol.dis --mmacs-per-tile 4,4 | FileCheck %s --check-prefix=ZOL-REAL
# ZOL-REAL: ORDER OK: 2 tiles, 64 inserts, 8 MMACs, no accumulator spills (loop body checked twice)

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Check per-tile counts, stream-pop spacing and the scalar-load ban.
# Prefetches belong to the tile consumed by the following commit.
# Timing hazards are covered in aie4-mmac-schedule-checker.py.
#
# RUN: %python %s %t/pops_good.dis --inserts-per-tile 1,1 --mmacs-per-tile 1,1 --pops-per-tile 1,1 | FileCheck %s --check-prefix=FLAT-POPS
# FLAT-POPS: pops: 2
# FLAT-POPS-NEXT: pop spacing: 3x1
# FLAT-POPS-NEXT: ORDER OK: 2 tiles, 2 inserts, 2 MMACs, 2 pops, no accumulator spills
#
# Without --pops-per-tile, pops are reported but the ORDER line is unchanged.
# RUN: %python %s %t/pops_good.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 | FileCheck %s --check-prefix=FLAT-NOGATE
# FLAT-NOGATE: pops: 2
# FLAT-NOGATE: ORDER OK: 2 tiles, 2 inserts, 2 MMACs, no accumulator spills
#
# Pop counts are gated per tile and in total.
# RUN: not %python %s %t/pops_good.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 --pops-per-tile 1,2 2>&1 | FileCheck %s --check-prefix=FLAT-COUNT
# FLAT-COUNT: error: expected 3 pops, found 2
# RUN: not %python %s %t/pops_good.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 --pops-per-tile 2 2>&1 | FileCheck %s --check-prefix=FLAT-ENTRIES
# FLAT-ENTRIES: error: --pops-per-tile has 1 entries for 2 tiles
#
# A pop after its tile's commit reads za cannot feed that tile.
# RUN: not %python %s %t/pop_late.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 --pops-per-tile 1,2 2>&1 | FileCheck %s --check-prefix=FLAT-LATE
# FLAT-LATE: error: tile 1: a pop issues at 14, not before the commit reads za at 14
#
# Pop spacing gate.
# RUN: not %python %s %t/pops_good.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 --min-pop-spacing 4 2>&1 | FileCheck %s --check-prefix=FLAT-SPACING
# FLAT-SPACING: error: pops 3 bundle(s) apart; at least 4 required
# RUN: %python %s %t/pops_good.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 --min-pop-spacing 3 | FileCheck %s --check-prefix=FLAT-NOGATE
#
# Insert lists group inserts per tile: 2,0 puts tile 1's insert in tile 0,
# after commit 0 reads.
# RUN: not %python %s %t/good.dis --inserts-per-tile 2,0 --mmacs-per-tile 1,1 2>&1 | FileCheck %s --check-prefix=FLAT-LIST
# FLAT-LIST: error: tile 0: an insert writes za at 8, not before the commit reads it at 8
# RUN: not %python %s %t/good.dis --inserts-per-tile 1,1,1 --mmacs-per-tile 1,1 2>&1 | FileCheck %s --check-prefix=FLAT-LIST-LEN
# FLAT-LIST-LEN: error: --inserts-per-tile has 3 entries for 2 tiles
#
# lda.s8 is rejected only on request, anywhere in the function.
# RUN: not %python %s %t/lda_s8.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 --forbid-lda-s8 2>&1 | FileCheck %s --check-prefix=FLAT-LDA
# FLAT-LDA: error: lda.s8 at 0x10: lda.s8{{[[:blank:]]+}}r0, [p0]
# RUN: %python %s %t/lda_s8.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 | FileCheck %s --check-prefix=FLAT-NOLDA
# FLAT-NOLDA: ORDER OK: 2 tiles, 2 inserts, 2 MMACs, no accumulator spills
#
# The ban also scans beyond ret and its five delay slots, where the schedule
# path ends. The load at 0x50 still belongs to main (no new function label).
# RUN: not %python %s %t/lda_s8_after_ret.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 --forbid-lda-s8 2>&1 | FileCheck %s --check-prefix=FLAT-LDA-AFTER-RET
# FLAT-LDA-AFTER-RET: error: lda.s8 at 0x50: lda.s8{{[[:blank:]]+}}r0, [p0]
# RUN: %python %s %t/lda_s8_after_ret.dis --inserts-per-tile 1 --mmacs-per-tile 1,1 | FileCheck %s --check-prefix=FLAT-NOLDA

import argparse
import re
import sys

INSERT_WRITE = 3
LOAD_STAGING_WRITE = 6
COMMIT_READ = 7
COMMIT_WRITE = 7
MMAC_READ = 2

INSERT_RE = re.compile(r"^vmov\s+(f_|i4_)?za(\[|\d+\s*,)")
LOAD_STAGING_RE = re.compile(r"^vld[ab]?\s+(f_|i4_)?za(\[|\d+\s*,)")
COMMIT_RE = re.compile(r"^vmov\s+(f_|i4_)?zb,\s*(f_|i4_)?za\b")
MMAC_RE = re.compile(r"^mmac\.[fi]\b")
POP_RE = re.compile(r"^vmov[abx]?\s+[xy]\d+,\s*ss\[")
LDA_S8_RE = re.compile(r"^lda\.s8\b")
ACC_STACK_RE = re.compile(
    r"^(vst|vld[ab]?|vlda|vldb)\S*\s+(cm|dm|bm)[a-z]*\d*\s*,\s*\[sp\b"
)
LINE_RE = re.compile(r"^\s*([0-9a-f]+):[^\t]*\t(.*)$")
FUNC_RE = re.compile(r"^[0-9a-f]+ <([^>]+)>:")
DELAY_SLOTS = 5
JUMP_RE = re.compile(r"^(j[a-z]*)\b(.*)$")
RET_RE = re.compile(r"^ret\b")
HW_LOOP_RE = re.compile(r"^\S+\s+(ls|le|lc)\s*,")
HW_LOOP_IMM_RE = re.compile(r"^\S+\s+(ls|le)\s*,\s*#(0x[0-9a-f]+|\d+)\s*$")
BUNDLE_BYTES = 32  # a full 256-bit instruction
ZOL_MIN_DISTANCE = 4  # bundles from the last ls/le/lc write to the loop end
COND_BACK_JUMPS = ("jnz", "jz", "jnzd")


def parse(lines, function):
    """Return the bundles of `function` as (address, [slot strings]).

    Local labels (.L...) inside the function do not end it."""
    bundles = []
    inside = False
    for line in lines:
        m = FUNC_RE.match(line)
        if m:
            name = m.group(1)
            if not name.startswith(".L"):
                inside = name == function
            continue
        if not inside:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        slots = [s.strip() for s in m.group(2).split(";")]
        bundles.append((int(m.group(1), 16), [s for s in slots if s]))
    return bundles


def executed_path(bundles):
    """Return the slot lists along the checked path, or exit on unsupported
    control flow (see the module docstring)."""
    addr_index = {addr: i for i, (addr, _) in enumerate(bundles)}
    end = len(bundles)
    jumps = []
    zol = {}  # register -> list of (bundle, slot text, immediate or None)
    for i, (_, slots) in enumerate(bundles):
        for s in slots:
            m = HW_LOOP_RE.match(s)
            if m:
                imm = HW_LOOP_IMM_RE.match(s)
                value = int(imm.group(2), 0) if imm else None
                zol.setdefault(m.group(1), []).append((i, s, value))
            if RET_RE.match(s):
                end = min(end, i + DELAY_SLOTS + 1)
            m = JUMP_RE.match(s)
            if m:
                jumps.append((i, m.group(1), m.group(2), s))
    jumps = [j for j in jumps if j[0] < end]
    if zol:
        if jumps:
            sys.exit(
                f"error: unsupported control flow: a zero-overhead loop and "
                f"{len(jumps)} jump(s); the check is straight-line or "
                "single-loop only"
            )
        return zero_overhead_loop_path(bundles, addr_index, end, zol), 1
    if not jumps:
        return [slots for _, slots in bundles[:end]], 0
    if len(jumps) > 1:
        sys.exit(
            f"error: unsupported control flow: {len(jumps)} jumps; the check "
            "is straight-line or single-loop only"
        )
    i, op, operands, text = jumps[0]
    target = re.search(r"#0x([0-9a-f]+)\s*$", operands)
    if op not in COND_BACK_JUMPS or not target:
        sys.exit(
            f"error: unsupported control flow at bundle {i} ({text}); the "
            "check is straight-line or single-loop only"
        )
    t = addr_index.get(int(target.group(1), 16))
    if t is None or t > i:
        sys.exit(
            f"error: unsupported control flow at bundle {i} ({text}): not a "
            "backward jump to a bundle of this function; the check is "
            "straight-line or single-loop only"
        )
    body_end = i + DELAY_SLOTS + 1
    body = [slots for _, slots in bundles[t:body_end]]
    path = (
        [slots for _, slots in bundles[:t]]
        + body
        + body
        + [slots for _, slots in bundles[body_end:end]]
    )
    return path, 1


def zero_overhead_loop_path(bundles, addr_index, end, zol):
    """Path for a function with one zero-overhead loop."""

    def fail(msg):
        sys.exit(f"error: zero-overhead loop: {msg}")

    for reg in ("ls", "le", "lc"):
        writes = zol.get(reg, [])
        if len(writes) != 1:
            fail(
                f"expected one write of {reg}, found {len(writes)}; the check "
                "is straight-line or single-loop only"
            )
    (ls_i, ls_s, ls_addr), (le_i, le_s, le_addr) = zol["ls"][0], zol["le"][0]
    lc_i = zol["lc"][0][0]
    for reg, text, addr in (("ls", ls_s, ls_addr), ("le", le_s, le_addr)):
        if addr is None:
            fail(f"{reg} is not set from an immediate ({text})")
        if addr % BUNDLE_BYTES:
            fail(f"{reg} = {addr:#x} is not 256-bit aligned")
        if addr not in addr_index:
            fail(f"{reg} = {addr:#x} is not a bundle of this function")
    start, last = addr_index[ls_addr], addr_index[le_addr]
    if last < start:
        fail(f"le = {le_addr:#x} is before ls = {ls_addr:#x}")
    if last + 1 < len(bundles) and bundles[last + 1][0] - le_addr != BUNDLE_BYTES:
        fail(f"the bundle at le = {le_addr:#x} is not a full 256-bit instruction")
    if max(ls_i, le_i) > lc_i:
        fail("ls and le must be written before lc")
    last_write = max(ls_i, le_i, lc_i)
    if last - last_write < ZOL_MIN_DISTANCE:
        fail(
            f"the loop ends {last - last_write} bundle(s) after the last "
            f"ls/le/lc write; at least {ZOL_MIN_DISTANCE} are required"
        )
    if last_write >= start:
        fail("ls/le/lc are written inside the loop body")
    body = [slots for _, slots in bundles[start : last + 1]]
    return (
        [slots for _, slots in bundles[:start]]
        + body
        + body
        + [slots for _, slots in bundles[last + 1 : end]]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("disassembly")
    p.add_argument("--function", default="main")
    p.add_argument(
        "--inserts-per-tile",
        default="32",
        help="staging writes per tile: one count for every tile, or a "
        "comma-separated count for each tile in order",
    )
    p.add_argument(
        "--pops-per-tile",
        help="comma-separated stream-pop count for each tile in order",
    )
    p.add_argument(
        "--min-pop-spacing",
        type=int,
        help="minimum distance in bundles between consecutive pops",
    )
    p.add_argument(
        "--forbid-lda-s8",
        action="store_true",
        help="reject scalar lda.s8 loads",
    )
    p.add_argument(
        "--mmacs-per-tile",
        required=True,
        help="comma-separated MMAC count for each staged tile, in order",
    )
    args = p.parse_args()

    with open(args.disassembly) as f:
        bundles = parse(f.readlines(), args.function)
    if not bundles:
        sys.exit(f"error: function <{args.function}> not found or empty")
    path, loops = executed_path(bundles)

    # lda.s8 anywhere in the function, not only on the checked path.
    lda_errors = []
    if args.forbid_lda_s8:
        for addr, slots in bundles:
            lda_errors += [
                f"lda.s8 at {addr:#x}: {s}" for s in slots if LDA_S8_RE.match(s)
            ]

    inserts, commits, mmacs, pops = [], [], [], []
    errors = list(lda_errors)
    for cycle, slots in enumerate(path):
        for s in slots:
            if POP_RE.match(s):
                pops.append(cycle)
            if INSERT_RE.match(s):
                inserts.append(cycle + INSERT_WRITE)
            elif LOAD_STAGING_RE.match(s):
                inserts.append(cycle + LOAD_STAGING_WRITE)
            elif COMMIT_RE.match(s):
                commits.append(cycle)
            elif MMAC_RE.match(s):
                mmacs.append(cycle + MMAC_READ)
            if ACC_STACK_RE.match(s):
                errors.append(f"accumulator spill/fill at cycle {cycle}: {s}")

    per_tile = [int(x) for x in args.mmacs_per_tile.split(",")]
    tiles = len(per_tile)
    ins_list = [int(x) for x in args.inserts_per_tile.split(",")]
    if len(ins_list) == 1:
        ins_list *= tiles
    if len(ins_list) != tiles:
        sys.exit(
            f"error: --inserts-per-tile has {len(ins_list)} entries for "
            f"{tiles} tiles"
        )
    pop_list = None
    if args.pops_per_tile is not None:
        pop_list = [int(x) for x in args.pops_per_tile.split(",")]
        if len(pop_list) != tiles:
            sys.exit(
                f"error: --pops-per-tile has {len(pop_list)} entries for "
                f"{tiles} tiles"
            )
    if len(commits) != tiles:
        errors.append(f"expected {tiles} commits, found {len(commits)}")
    if len(inserts) != sum(ins_list):
        errors.append(f"expected {sum(ins_list)} inserts, found {len(inserts)}")
    if pop_list is not None and len(pops) != sum(pop_list):
        errors.append(f"expected {sum(pop_list)} pops, found {len(pops)}")
    spacing = {}
    for a, b in zip(pops, pops[1:]):
        spacing[b - a] = spacing.get(b - a, 0) + 1
    if args.min_pop_spacing is not None and spacing:
        closest = min(spacing)
        if closest < args.min_pop_spacing:
            errors.append(
                f"pops {closest} bundle(s) apart; at least "
                f"{args.min_pop_spacing} required"
            )
    if len(mmacs) != sum(per_tile):
        errors.append(f"expected {sum(per_tile)} MMACs, found {len(mmacs)}")
    if errors:
        sys.exit("\n".join("error: " + e for e in errors))

    inserts.sort()
    mmacs.sort()
    commits.sort()
    first = 0
    ins_first = 0
    pop_first = 0
    for g in range(tiles):
        read = commits[g] + COMMIT_READ
        write = commits[g] + COMMIT_WRITE
        group = inserts[ins_first : ins_first + ins_list[g]]
        ins_first += ins_list[g]
        if group and group[-1] >= read:
            errors.append(
                f"tile {g}: an insert writes za at {group[-1]}, not before the "
                f"commit reads it at {read}"
            )
        if pop_list is not None and pop_list[g]:
            tile_pops = pops[pop_first : pop_first + pop_list[g]]
            pop_first += pop_list[g]
            if tile_pops[-1] >= read:
                errors.append(
                    f"tile {g}: a pop issues at {tile_pops[-1]}, not before "
                    f"the commit reads za at {read}"
                )
        if g + 1 < tiles and ins_list[g + 1]:
            nxt = inserts[ins_first]
            if nxt < read:
                errors.append(
                    f"tile {g + 1}: an insert writes za at {nxt}, before commit "
                    f"{g} reads it at {read}"
                )
        rows = mmacs[first : first + per_tile[g]]
        first += per_tile[g]
        if rows and rows[0] <= write:
            errors.append(
                f"tile {g}: an MMAC reads zb at {rows[0]}, not after the commit "
                f"writes it at {write}"
            )
        if rows and g + 1 < tiles:
            nxt_write = commits[g + 1] + COMMIT_WRITE
            if nxt_write < rows[-1]:
                errors.append(
                    f"tile {g}: commit {g + 1} writes zb at {nxt_write}, before "
                    f"an MMAC reads it at {rows[-1]}"
                )
    if errors:
        sys.exit("\n".join("error: " + e for e in errors))
    shape = " (loop body checked twice)" if loops else ""
    pop_count = f", {len(pops)} pops" if pop_list is not None else ""
    if pops:
        print(f"pops: {len(pops)}")
        print(
            "pop spacing: " + " ".join(f"{d}x{n}" for d, n in sorted(spacing.items()))
        )
    print(
        f"ORDER OK: {tiles} tiles, {len(inserts)} inserts, {len(mmacs)} MMACs"
        f"{pop_count}, no accumulator spills{shape}"
    )


FIXTURES = {
    "forward_jump.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za[i0], y0
       4: 00 00 00 00      	jnz	r0, #0xc
       8: 00 00 00 00      	nop
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	vmov	zb, za
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      24: 00 00 00 00      	ret	lr
      28: 00 00 00 00      	nop
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	nop
      38: 00 00 00 00      	nop

0000003c <other>:
      3c: 00 00 00 00      	vmov	za[i0], y0
""",
    "good.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za[i0], y0
       4: 00 00 00 00      	vmov	zb, za
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	vmov	za[i0], y1
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
""",
    "hw_loop.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	movxm	ls, #0x10
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      10: 00 00 00 00      	ret	lr
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	nop
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	nop

00000028 <other>:
      28: 00 00 00 00      	vmov	za[i0], y0
""",
    "imm_index.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za5, y0
       4: 00 00 00 00      	vmov	zb, za
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	vmov	f_za7, y1
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8

00000024 <other>:
      24: 00 00 00 00      	vmov	za[i0], y0
""",
    "kloop_zol.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Peano zero-overhead-loop schedule for aie4-mmac-schedule-checker.test.
# Four bf16 rows per K tile; the loop contains staging, commit and MMACs.
# The checker executes the body twice to check hazards across the back edge.

o/elfs_main_core_0_2/elfs_main_core_0_2.elf:	file format elf32-aie

Disassembly of section .text:

00000000 <__start>:
       0: f4 00 00 03 48 10 00 00      	jl	#0x20
       8: f4 00 00 00 e0 05 0e 00      	movxm	sp, #0xe0000
      10: 0c 00        	nop
      12: 0c 00        	nop
      14: 0c 00        	nop
      16: 0c 00        	nop
      18: 04 0a 01 00 00 7e 7a 03      	nopa	;		nopv

00000020 <_main_init>:
      20: 7f 00 00 0a 01 00 00 03 4c 40 00 00 eb d2 c1 97 01 3e 10 00 00 00 00 00 00 00 00 00 00 00 00 00      	nop	;		paddxm	 [sp], #0x80;		nopb	;		nops	;		jl	pc[#0x80];		nopv	;		nopv2
      40: 41 0c 79 78 ff f7 c6 97 01 7e 7a 03  	nopa	;		nopb	;		nopx	;		st	 p7, [sp, #-124]
      4c: bc 18 06 f0  	st	 lr, [sp, #-128]
      50: 7c ca 8f ec  	mov	p7, sp
      54: f8 d4 00 30 18 20 00 7e aa fb	padda	 [p7], #-0x80;		movx	r1, #0x0;		mov	p0, #0x0
      5e: 0c 00        	nop
      60: 7e 00 00 0a 01 00 58 97 f0 f9 bf 17 eb d2 c1 97 01 7e 7a 03  	nopa	;		nopb	;		nops	;		done	;		nopm	;		nopv	;		nopv2
      74: 0c 00        	nop
      76: 0c 00        	nop
      78: 0c 00        	nop
      7a: 0c 00        	nop
      7c: 0c 00        	nop
      7e: 0c 00        	nop
      80: 3c 0e 36 f0  	lda	 lr, [sp, #-128]
      84: 0c 00        	nop
      86: 0c 00        	nop
      88: 0c 00        	nop
      8a: 0c 00        	nop
      8c: d4 00 f8 bf 07 0e f2 f0      	lda	 p7, [sp, #-124];		ret	lr
      94: 0c 00        	nop
      96: 0c 00        	nop
      98: 0c 00        	nop
      9a: 0c 00        	nop
      9c: 3c 3e f0 ff  	paddxm	 [sp], #-0x80

000000a0 <main>:
      a0: 7f 00 00 0a 01 00 00 c0 9f 00 00 00 eb d2 c1 97 01 7e 7a 03 00 00 00 00 00 00 00 00 00 00 00 00      	nop	;		nopa	;		nopb	;		nops	;		movxm	r2, #0x7e0;		nopv	;		nopv2
      c0: f4 00 00 00 60 81 0e 00      	movxm	p1, #0xe8000
      c8: f4 00 00 00 68 02 0e 00      	movxm	p2, #0xe0400
      d0: 7f 00 00 0a 01 00 00 80 93 fe ff ff eb d2 c1 97 01 aa 1c 00 00 00 00 00 00 00 00 00 00 00 00 00      	nop	;		mova	r7, #0x2;		nopb	;		nops	;		movxm	r26, #-0x640;		nopv	;		nopv2
      f0: f8 00 00 8a ff ff 7f 6a 4c 00	mova	r19, #0x1;		movxm	r28, #-0x600
      fa: f8 00 40 ca ff ff 7f 2a 5c 10	mova	r23, #0x80;		movxm	r30, #-0x5c0
     104: f8 00 80 2a fc ff 7f 2a 6c 0c	mova	r27, #0x60;		movxm	r1, #-0x580
     10e: 55 03 60 ed c9 05 80 55 f9 ff ff 2a 7c e8    	mova	r31, #-0xc0;		movxm	r5, #-0x540;		movs	i5, #0x19
     11c: 55 03 60 ed d1 06 00 56 fc ff ff 2a 50 80    	mova	r20, #-0x400;		movxm	r17, #-0x500;		movs	i6, #0x1a
     12a: 55 03 60 ed d9 07 80 56 fd ff ff 2a 00 c0    	mova	r0, #-0x200;		movxm	r21, #-0x4c0;		movs	i7, #0x1b
     138: 55 03 60 ed e1 00 00 57 fe ff ff 2a 10 c8    	mova	r4, #-0x1c0;		movxm	r25, #-0x480;		movs	i0, #0x1c
     146: 55 03 60 ed e9 01 80 57 ff ff ff 2a 18 d0    	mova	r6, #-0x180;		movxm	r29, #-0x440;		movs	i1, #0x1d
     154: 55 03 60 ed f1 02 03 14 f0 00 00 3e 10 00    	paddxm	 [sp], #0x80;		movxm	ls, #0x1e0;		movs	i2, #0x1e
     162: 55 03 c0 08 81 07 00 60 00 0f 00 2a 40 d8    	mova	r16, #-0x140;		movxm	p0, #0xf0000;		st	 r8, [sp, #-128]
     170: 55 03 c0 68 8d 07 00 60 03 0f 00 2e 02 84    	vlda	 cmah0, [p0, #0x80];		movxm	p3, #0xf0000;		st	 r11, [sp, #-116]
     17e: 55 03 c0 28 85 07 03 10 40 03 00 2e 80 c9    	vlda	 cmal0, [p3], #0x100;		movxm	le, #0x680;		st	 r9, [sp, #-124]
     18c: 55 03 c0 48 89 af 81 d0 f8 c3 14 2e 86 85    	vlda	 cmah1, [p3, #0x80];		movx	crintmmode, #0x2;		mov	r3, #0x4;		st	 r10, [sp, #-120]
     19a: 55 03 c0 88 91 af 1d 90 74 e0 18 2e 84 c9    	vlda	 cmal1, [p3], #0x100;		add.nc	lc, r3, #0x0;		mov	r18, #-0x100;		st	 r12, [sp, #-112]
     1a8: 55 03 c0 c8 99 af 11 98 32 00 93 2e 8a 85    	vlda	 cmah2, [p3, #0x80];		movx	r12, #-0x380;		mov	r10, #-0x3c0;		st	 r14, [sp, #-104]
     1b6: 55 03 c0 a8 95 af 13 98 33 40 a2 2e 88 81    	vlda	 cmal2, [p3, #0x0];		movx	r9, #-0x300;		mov	r14, #-0x340;		st	 r13, [sp, #-108]
     1c4: 55 03 c0 e8 9d af 15 d8 32 40 b3 2e 8e 8d    	vlda	 cmah3, [p3, #0x180];		movx	r13, #-0x280;		mov	r11, #-0x2c0;		st	 r15, [sp, #-100]
     1d2: 55 03 60 ed f9 ab 17 d8 33 00 02 2e 8c 89    	vlda	 cmal3, [p3, #0x100];		movx	r8, #0x0;		mov	r15, #-0x240;		movs	i3, #0x1f

000001e0 <.LBB0_1>:
     1e0: bc a2 a7 15  	lshl	 r22, r2, r19
     1e4: 7f 00 00 0a 01 00 00 80 10 f6 ff ff eb d2 c1 97 01 7e 7a 03 00 00 00 00 00 00 00 00 00 00 00 00      	nop	;		nopa	;		nopb	;		nops	;		movxm	r24, #-0xfc0;		nopv	;		nopv2
     204: bc 02 30 b6  	add	 r24, r22, r24
     208: f8 01 80 08 fb ff 7f dc 60 04	movs	dj0, r24;		movxm	r24, #-0xf80
     212: bc 02 30 b6  	add	 r24, r22, r24
     216: a1 00 30 c2 fe ff df 09 82 dc 60 04  	movs	dj0, r24;		vldb	 x0, [p2, dj0];		movxm	r24, #-0xf40
     222: bc 02 30 b6  	add	 r24, r22, r24
     226: a1 00 40 c2 fe ff df 19 82 dc 60 04  	movs	dj0, r24;		vldb	 x1, [p2, dj0];		movxm	r24, #-0xf00
     232: bc 02 30 b6  	add	 r24, r22, r24
     236: 55 03 e0 06 23 00 80 12 f6 ff ff be 11 41    	vlda	 x2, [p2, dj0];		movxm	r24, #-0xec0;		movs	dj0, r24
     244: bc 02 30 b6  	add	 r24, r22, r24
     248: a1 00 60 c2 fe ff df 39 82 dc 60 04  	movs	dj0, r24;		vldb	 x3, [p2, dj0];		movxm	r24, #-0xe80
     254: bc 02 30 b6  	add	 r24, r22, r24
     258: 55 03 e0 06 23 00 80 13 f6 ff ff be 21 41    	vlda	 x4, [p2, dj0];		movxm	r24, #-0xe40;		movs	dj0, r24
     266: bc 02 30 b6  	add	 r24, r22, r24
     26a: a1 00 80 c2 fe ff df 59 82 dc 60 04  	movs	dj0, r24;		vldb	 x5, [p2, dj0];		movxm	r24, #-0xe00
     276: bc 02 30 b6  	add	 r24, r22, r24
     27a: 55 03 e0 06 23 00 80 14 f6 ff ff be 31 41    	vlda	 x6, [p2, dj0];		movxm	r24, #-0xdc0;		movs	dj0, r24
     288: bc 02 30 b6  	add	 r24, r22, r24
     28c: a1 00 a0 c2 fe ff df 79 82 dc 60 04  	movs	dj0, r24;		vldb	 x7, [p2, dj0];		movxm	r24, #-0xd80
     298: bc 02 30 b6  	add	 r24, r22, r24
     29c: 55 03 e0 06 23 00 80 15 f6 ff ff be 41 41    	vlda	 x8, [p2, dj0];		movxm	r24, #-0xd40;		movs	dj0, r24
     2aa: bc 02 30 b6  	add	 r24, r22, r24
     2ae: a1 00 c0 c2 fe ff df 99 82 dc 60 04  	movs	dj0, r24;		vldb	 x9, [p2, dj0];		movxm	r24, #-0xd00
     2ba: bc 02 30 b6  	add	 r24, r22, r24
     2be: 55 03 e0 06 23 00 80 16 f6 ff ff be 51 41    	vlda	 x10, [p2, dj0];		movxm	r24, #-0xcc0;		movs	dj0, r24
     2cc: bc 02 30 b6  	add	 r24, r22, r24
     2d0: a1 00 e0 c2 fe ff df b9 82 dc 60 04  	movs	dj0, r24;		vldb	 x11, [p2, dj0];		movxm	r24, #-0xc80
     2dc: bc 02 30 b6  	add	 r24, r22, r24
     2e0: 55 03 e0 06 23 00 80 17 f6 ff ff be 61 41    	vlda	 x12, [p2, dj0];		movxm	r24, #-0xc40;		movs	dj0, r24
     2ee: bc 02 30 b6  	add	 r24, r22, r24
     2f2: 55 03 e0 06 23 00 00 18 f6 ff ff be 69 41    	vlda	 x13, [p2, dj0];		movxm	r24, #-0xc00;		movs	dj0, r24
     300: f4 00 58 67 22 04 30 b6      	add	 r24, r22, r24;		mov	i4, r8
     308: a1 00 10 c3 fe ff df e9 82 dc 60 04  	movs	dj0, r24;		vldb	 x14, [p2, dj0];		movxm	r24, #-0xbc0
     314: f4 00 58 17 99 00 30 b6      	add	 r24, r22, r24;		vmov	za[i4], y0
     31c: 55 03 e0 06 23 00 00 19 f6 ff ff be 79 41    	vlda	 x15, [p2, dj0];		movxm	r24, #-0xb80;		movs	dj0, r24
     32a: f4 00 58 67 4e 04 30 b6      	add	 r24, r22, r24;		mov	i4, r19
     332: a1 00 30 c3 fe ff df 09 82 dc 60 04  	movs	dj0, r24;		vldb	 x0, [p2, dj0];		movxm	r24, #-0xb40
     33e: f4 00 58 17 99 01 30 b6      	add	 r24, r22, r24;		vmov	za[i4], y1
     346: 55 03 e0 06 23 00 00 1a f6 ff ff be 09 41    	vlda	 x1, [p2, dj0];		movxm	r24, #-0xb00;		movs	dj0, r24
     354: f4 00 58 67 1e 04 30 b6      	add	 r24, r22, r24;		mov	i4, r7
     35c: a1 00 50 c3 fe ff df 29 82 dc 60 04  	movs	dj0, r24;		vldb	 x2, [p2, dj0];		movxm	r24, #-0xac0
     368: f8 ad 8b 4c 01 18 5b ac 3d 83	movs	i4, #0x3;		add	 r24, r22, r24;		vmov	za[i4], y2
     372: 55 03 e0 06 23 00 00 1b f6 ff ff be 19 41    	vlda	 x3, [p2, dj0];		movxm	r24, #-0xa80;		movs	dj0, r24
     380: f4 00 58 17 99 03 30 b6      	add	 r24, r22, r24;		vmov	za[i4], y3
     388: a1 00 70 c3 fe ff df 49 82 dc 60 04  	movs	dj0, r24;		vldb	 x4, [p2, dj0];		movxm	r24, #-0xa40
     394: f4 00 58 67 0e 04 30 b6      	add	 r24, r22, r24;		mov	i4, r3
     39c: 55 03 e0 06 23 00 00 1c f6 ff ff be 29 41    	vlda	 x5, [p2, dj0];		movxm	r24, #-0xa00;		movs	dj0, r24
     3aa: f4 00 58 17 99 04 30 b6      	add	 r24, r22, r24;		vmov	za[i4], y4
     3b2: 55 03 e0 06 23 00 80 1c f6 ff ff be 31 41    	vlda	 x6, [p2, dj0];		movxm	r24, #-0x9c0;		movs	dj0, r24
     3c0: f4 00 58 e7 2b 04 30 b6      	add	 r24, r22, r24;		mov	i4, #0x5
     3c8: 55 03 e0 06 23 00 00 1d f6 ff ff be 39 41    	vlda	 x7, [p2, dj0];		movxm	r24, #-0x980;		movs	dj0, r24
     3d6: f4 00 58 17 99 05 30 b6      	add	 r24, r22, r24;		vmov	za[i4], y5
     3de: a1 00 b0 c3 fe ff df 89 82 dc 60 04  	movs	dj0, r24;		vldb	 x8, [p2, dj0];		movxm	r24, #-0x940
     3ea: f4 00 58 e7 33 04 30 b6      	add	 r24, r22, r24;		mov	i4, #0x6
     3f2: 55 03 e0 06 23 00 00 1e f6 ff ff be 49 41    	vlda	 x9, [p2, dj0];		movxm	r24, #-0x900;		movs	dj0, r24
     400: f8 ad 8b 4c 03 18 5b ac 3d 87	movs	i4, #0x7;		add	 r24, r22, r24;		vmov	za[i4], y6
     40a: a1 00 d0 c3 fe ff df a9 82 dc 60 04  	movs	dj0, r24;		vldb	 x10, [p2, dj0];		movxm	r24, #-0x8c0
     416: f8 ad 8b cc 03 18 5b ac 3d 88	movs	i4, #0x8;		add	 r24, r22, r24;		vmov	za[i4], y7
     420: a1 00 e0 c3 fe ff df b9 82 dc 60 04  	movs	dj0, r24;		vldb	 x11, [p2, dj0];		movxm	r24, #-0x880
     42c: f8 ad 8b 4c 00 18 5b ac 3d 89	movs	i4, #0x9;		add	 r24, r22, r24;		vmov	za[i4], y0
     436: 55 03 e0 06 23 00 80 1f f6 ff ff be 61 41    	vlda	 x12, [p2, dj0];		movxm	r24, #-0x840;		movs	dj0, r24
     444: f8 ad 8b cc 00 18 5b ac 3d 8a	movs	i4, #0xa;		add	 r24, r22, r24;		vmov	za[i4], y1
     44e: a1 00 00 c2 ff ff df d9 82 dc 60 04  	movs	dj0, r24;		vldb	 x13, [p2, dj0];		movxm	r24, #-0x800
     45a: f8 ad 8b 4c 01 18 5b ac 3d 8b	movs	i4, #0xb;		add	 r24, r22, r24;		vmov	za[i4], y2
     464: 55 03 e0 06 23 00 80 10 fe ff ff be 71 41    	vlda	 x14, [p2, dj0];		movxm	r24, #-0x7c0;		movs	dj0, r24
     472: f8 ad 8b cc 01 18 5b ac 3d 8c	movs	i4, #0xc;		add	 r24, r22, r24;		vmov	za[i4], y3
     47c: a1 00 20 c2 ff ff df f9 82 dc 60 04  	movs	dj0, r24;		vldb	 x15, [p2, dj0];		movxm	r24, #-0x780
     488: f8 ad 8b 4c 02 18 5b ac 3d 8d	movs	i4, #0xd;		add	 r24, r22, r24;		vmov	za[i4], y4
     492: 55 03 e0 06 23 00 80 11 fe ff ff be 01 41    	vlda	 x0, [p2, dj0];		movxm	r24, #-0x740;		movs	dj0, r24
     4a0: f8 ad 8b cc 02 18 5b ac 3d 8e	movs	i4, #0xe;		add	 r24, r22, r24;		vmov	za[i4], y5
     4aa: a1 00 40 c2 ff ff df 19 82 dc 60 04  	movs	dj0, r24;		vldb	 x1, [p2, dj0];		movxm	r24, #-0x700
     4b6: f8 ad 8b 4c 03 18 5b ac 3d 8f	movs	i4, #0xf;		add	 r24, r22, r24;		vmov	za[i4], y6
     4c0: a1 00 50 c2 ff ff df 29 82 dc 60 04  	movs	dj0, r24;		vldb	 x2, [p2, dj0];		movxm	r24, #-0x6c0
     4cc: f8 ad 8b cc 03 18 5b ac 3d 90	movs	i4, #0x10;		add	 r24, r22, r24;		vmov	za[i4], y7
     4d6: 55 03 e0 06 23 00 00 13 fe ff ff be 19 41    	vlda	 x3, [p2, dj0];		movxm	r24, #-0x680;		movs	dj0, r24
     4e4: 00 c0 d8 ae 3d 91    	movs	i4, #0x11;		add	 r24, r22, r24
     4ea: a1 eb 22 13 80 c6 d6 49 82 dc 60 04  	movs	dj0, r24;		vldb	 x4, [p2, dj0];		add	 r24, r22, r26;		vmov	za[i4], y0
     4f6: c4 00 dc 60 04 be 29 41      	vlda	 x5, [p2, dj0];		movs	dj0, r24
     4fe: 00 e0 d8 ae 3d 92    	movs	i4, #0x12;		add	 r24, r22, r28
     504: a1 eb 22 33 80 c7 d6 69 82 dc 60 04  	movs	dj0, r24;		vldb	 x6, [p2, dj0];		add	 r24, r22, r30;		vmov	za[i4], y1
     510: 78 02 20 60 cb 79 82 dc 60 04	movs	dj0, r24;		vldb	 x7, [p2, dj0];		add	 r24, r22, r1
     51a: 78 02 a0 60 cb 89 82 dc 60 04	movs	dj0, r24;		vldb	 x8, [p2, dj0];		add	 r24, r22, r5
     524: 34 00 dc 60 04 e0 4c 41      	vldb	 x9, [p2, dj0];		movs	dj0, r24
     52c: 00 88 d8 ae 3d 93    	movs	i4, #0x13;		add	 r24, r22, r17
     532: a1 eb 22 53 40 c5 d6 a9 82 dc 60 04  	movs	dj0, r24;		vldb	 x10, [p2, dj0];		add	 r24, r22, r21;		vmov	za[i4], y2
     53e: 34 00 dc 60 04 e0 5c 41      	vldb	 x11, [p2, dj0];		movs	dj0, r24
     546: 00 c8 d8 ae 3d 94    	movs	i4, #0x14;		add	 r24, r22, r25
     54c: a1 eb 22 73 40 c7 d6 c9 82 dc 60 04  	movs	dj0, r24;		vldb	 x12, [p2, dj0];		add	 r24, r22, r29;		vmov	za[i4], y3
     558: 34 00 dc 60 04 e0 6c 41      	vldb	 x13, [p2, dj0];		movs	dj0, r24
     560: 00 a0 d8 ae 3d 95    	movs	i4, #0x15;		add	 r24, r22, r20
     566: a1 eb 22 93 80 c2 d6 e9 82 dc 60 04  	movs	dj0, r24;		vldb	 x14, [p2, dj0];		add	 r24, r22, r10;		vmov	za[i4], y4
     572: 34 00 dc 60 04 e0 7c 41      	vldb	 x15, [p2, dj0];		movs	dj0, r24
     57a: 00 60 d8 ae 3d 96    	movs	i4, #0x16;		add	 r24, r22, r12
     580: 55 03 e0 06 23 58 17 99 05 1c b6 be 01 41    	vlda	 x0, [p2, dj0];		add	 r24, r22, r14;		vmov	za[i4], y5;		movs	dj0, r24
     58e: c4 00 dc 60 04 be 09 41      	vlda	 x1, [p2, dj0];		movs	dj0, r24
     596: 00 48 d8 ae 3d 97    	movs	i4, #0x17;		add	 r24, r22, r9
     59c: 55 03 e0 06 23 58 17 99 06 16 b6 be 11 41    	vlda	 x2, [p2, dj0];		add	 r24, r22, r11;		vmov	za[i4], y6;		movs	dj0, r24
     5aa: c4 00 dc 60 04 be 19 41      	vlda	 x3, [p2, dj0];		movs	dj0, r24
     5b2: 00 68 d8 ae 3d 98    	movs	i4, #0x18;		add	 r24, r22, r13
     5b8: 55 03 e0 06 23 58 17 99 07 1e b6 be 21 41    	vlda	 x4, [p2, dj0];		add	 r24, r22, r15;		vmov	za[i4], y7;		movs	dj0, r24
     5c6: 98 70 83 11 00 00 5b be 29 41	vlda	 x5, [p2, dj0];		add	 r24, r22, r0;		movs	dj0, r24
     5d0: 78 02 80 60 cb 69 82 dc 60 04	movs	dj0, r24;		vldb	 x6, [p2, dj0];		add	 r24, r22, r4
     5da: 98 70 83 11 00 06 5b be 39 41	vlda	 x7, [p2, dj0];		add	 r24, r22, r6;		movs	dj0, r24
     5e4: 78 02 00 62 cb 89 82 dc 60 04	movs	dj0, r24;		vldb	 x8, [p2, dj0];		add	 r24, r22, r16
     5ee: 78 02 40 62 cb 99 82 dc 60 04	movs	dj0, r24;		vldb	 x9, [p2, dj0];		add	 r24, r22, r18
     5f8: b8 a8 8b 4c dc 60 04 be 51 41	vlda	 x10, [p2, dj0];		movs	dj0, r24;		vmov	za[i4], y0
     602: 98 05 a0 45 00 1f 5b e0 5c 41	vldb	 x11, [p2, dj0];		add	 r24, r22, r31;		addm.nc	dj0, r22, #-0x80
     60c: 44 00 ea 22 37 dc 60 24      	movs	dj1, r24;		vmov	za[i5], y1
     614: 14 00 00 6c 11 e0 6c 41      	vldb	 x13, [p2, dj0];		addm.nc	dj0, r22, #-0x40
     61c: 55 03 e0 c6 22 58 17 d9 a2 a7 dd be 71 41    	vlda	 x14, [p2, dj0];		lshl	 r22, r27, r19;		vmov	za[i6], y2;		movs	dj0, r22
     62a: f8 ac 8b fc 01 1f 5b be 61 45	vlda	 x12, [p2, dj1];		add	 r24, r22, r31;		vmov	za[i7], y3
     634: b8 a8 8b 0c de 60 04 be 79 41	vlda	 x15, [p2, dj0];		movs	dj0, r24;		vmov	za[i0], y4
     63e: e4 00 00 68 11 be 81 40      	vlda	 x0, [p1, dj0];		addm.nc	dj0, r22, #-0x80
     646: 7c ea 22 a7  	vmov	za[i1], y5
     64a: 7c ea 12 9a  	vmov	zb, za
     64e: e4 00 00 6c 11 be 89 40      	vlda	 x1, [p1, dj0];		addm.nc	dj0, r22, #-0x40
     656: 7c ea 22 cb  	vmov	za[i2], y6
     65a: b8 a8 8b bc df 58 04 be 91 40	vlda	 x2, [p1, dj0];		movs	dj0, r22;		vmov	za[i3], y7
     664: 3c be 99 40  	vlda	 x3, [p1, dj0]
     668: 0c 00        	nop
     66a: 7c 0f 00 00  	mmac.f	dma0, dma0, x0, zb, r8
     66e: 84 70 21 08 00 00 ee de      	add	 r27, r27, r23;		mmac.f	dma1, dma1, x1, zb, r8
     676: 58 02 00 c8 06 00 00 4e 08 02	mmac.f	dma2, dma2, x2, zb, r8;		movxm	r22, #0x800

00000680 <.L_LEnd0>:
     680: df 18 06 0a 01 00 58 97 f0 01 ac 10 eb d2 c1 97 01 7e 7a 03 00 00 00 00 00 00 00 00 00 00 00 00      	nop	;		nopa	;		nopb	;		nops	;		add	 r2, r2, r22;		nopm	;		nopv	;		mmac.f	dma3, dma3, x3, zb, r8
     6a0: 3c 0e bd f3  	lda	 r15, [sp, #-100]
     6a4: 3c 0e 39 f3  	lda	 r14, [sp, #-104]
     6a8: c4 00 5c 02 84 0e b5 f2      	lda	 r13, [sp, #-108];		vst	 cmah0, [p0, #0x80]
     6b0: c4 00 5c 00 c8 0e 31 f2      	lda	 r12, [sp, #-112];		vst	 cmal0, [p0], #0x100
     6b8: c4 00 5c 06 84 0e ad f1      	lda	 r11, [sp, #-116];		vst	 cmah1, [p0, #0x80]
     6c0: c4 00 5c 04 c8 0e 29 f1      	lda	 r10, [sp, #-120];		vst	 cmal1, [p0], #0x100
     6c8: c4 00 5c 0a 84 0e a5 f0      	lda	 r9, [sp, #-124];		vst	 cmah2, [p0, #0x80]
     6d0: 7f 00 00 0a 01 00 58 97 f0 f9 bf 07 17 02 e0 97 01 0e 21 f0 00 00 00 00 00 00 00 00 00 00 00 00      	nop	;		lda	 r8, [sp, #-128];		nopb	;		vst	 cmal2, [p0, #0x0];		ret	lr;		nopm	;		nopv	;		nopv2
     6f0: bc 5c 0c 88  	vst	 cmal3, [p0, #0x100]
     6f4: bc 5c 0e 8c  	vst	 cmah3, [p0, #0x180]
     6f8: 0c 00        	nop
     6fa: 0c 00        	nop
     6fc: 3c 3e f0 ff  	paddxm	 [sp], #-0x80
""",
    "lda_s8.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmovb	y0, ss[p0]
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	vmovb	y1, ss[p0]
      10: 00 00 00 00      	lda.s8	r0, [p0]
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	vmov	za[i0], y1
      1c: 00 00 00 00      	vmov	zb, za
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	nop
      28: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
""",
    "lda_s8_after_ret.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmovb	y0, ss[p0]
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	vmovb	y1, ss[p0]
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	vmov	za[i0], y1
      1c: 00 00 00 00      	vmov	zb, za
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	nop
      28: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
      38: 00 00 00 00      	ret
      3c: 00 00 00 00      	nop
      40: 00 00 00 00      	nop
      44: 00 00 00 00      	nop
      48: 00 00 00 00      	nop
      4c: 00 00 00 00      	nop
      50: 00 00 00 00      	lda.s8	r0, [p0]
""",
    "loop_delay_slot.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	nop

00000004 <.LBB0_1>:
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	nop
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	jnz	r0, #0x4
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	nop
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	vmov	zb, za
      28: 00 00 00 00      	ret	lr
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	nop
      38: 00 00 00 00      	nop
      3c: 00 00 00 00      	nop

00000040 <other>:
      40: 00 00 00 00      	vmov	za[i0], y0
""",
    "loop_good.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	nop

00000004 <.LBB0_1>:
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	nop
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	jnz	r0, #0x4
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	vmov	zb, za
      1c: 00 00 00 00      	nop
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	nop
      28: 00 00 00 00      	ret	lr
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	nop
      38: 00 00 00 00      	nop
      3c: 00 00 00 00      	nop
      40: 00 00 00 00      	vmov	za[i0], y0

00000044 <other>:
      44: 00 00 00 00      	vmov	za[i0], y0
""",
    "loop_war.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	nop

00000004 <.LBB0_1>:
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	nop
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	jnz	r0, #0x4
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	vmov	zb, za
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	nop
      28: 00 00 00 00      	ret	lr
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	nop
      38: 00 00 00 00      	nop
      3c: 00 00 00 00      	nop

00000040 <other>:
      40: 00 00 00 00      	vmov	za[i0], y0
""",
    "pop_late.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmovb	y0, ss[p0]
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	vmovb	y1, ss[p0]
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	vmov	za[i0], y1
      1c: 00 00 00 00      	vmov	zb, za
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	nop
      28: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
      38: 00 00 00 00      	vmovx	x0, ss[p1]
""",
    "pops_good.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmovb	y0, ss[p0]
       4: 00 00 00 00      	vmov	za[i0], y0
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	vmovb	y1, ss[p0]
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	vmov	za[i0], y1
      1c: 00 00 00 00      	vmov	zb, za
      20: 00 00 00 00      	nop
      24: 00 00 00 00      	nop
      28: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
      2c: 00 00 00 00      	nop
      30: 00 00 00 00      	nop
      34: 00 00 00 00      	mmac.i	dma0, dma0, x0, zb, r8
""",
    "raw_insert.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za[i0], y0
       4: 00 00 00 00      	vmov	zb, za
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	vmov	za[i0], y1
      1c: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
""",
    "raw_mmac.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za[i0], y0
       4: 00 00 00 00      	vmov	zb, za
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	vmov	za[i0], y1
      18: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      1c: 00 00 00 00      	nop
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
""",
    "spill.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za[i0], y0
       4: 00 00 00 00      	vmov	zb, za
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	vmov	za[i0], y1
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      24: 00 00 00 00      	vst	cmal0, [sp, #-0x40]
""",
    "war_insert.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za[i0], y0
       4: 00 00 00 00      	vmov	zb, za
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	vmov	za[i0], y1
      14: 00 00 00 00      	nop
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
""",
    "war_mmac.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	vmov	za[i0], y0
       4: 00 00 00 00      	vmov	zb, za
       8: 00 00 00 00      	vmov	zb, za
       c: 00 00 00 00      	nop
      10: 00 00 00 00      	nop
      14: 00 00 00 00      	vmov	za[i0], y1
      18: 00 00 00 00      	nop
      1c: 00 00 00 00      	nop
      20: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
      24: 00 00 00 00      	mmac.f	dma0, dma0, x0, zb, r8
""",
    "zol_compressed_le.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	movxm	ls, #0x100
      20: 00 00 00 00      	movxm	le, #0x1a0
      40: 00 00 00 00      	add.nc	lc, r0, #-0x2
      60: 00 00 00 00      	nop
      80: 00 00 00 00      	nop
      a0: 00 00 00 00      	nop
      c0: 00 00 00 00      	nop
      e0: 00 00 00 00      	nop
     100: 00 00 00 00      	vmov	za[i0], y0
     120: 00 00 00 00      	nop
     140: 00 00 00 00      	vmov	zb, za
     160: 00 00 00 00      	nop
     180: 00 00 00 00      	nop
     1a0: 00 00 00 00      	nop
     1a4: 00 00 00 00      	ret	lr
     1c4: 00 00 00 00      	nop
     1e4: 00 00 00 00      	nop
     204: 00 00 00 00      	nop
     224: 00 00 00 00      	nop
     244: 00 00 00 00      	nop

00000264 <other>:
     264: 00 00 00 00      	vmov	za[i0], y0
""",
    "zol_good.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	movxm	ls, #0x100
      20: 00 00 00 00      	movxm	le, #0x1a0
      40: 00 00 00 00      	add.nc	lc, r0, #-0x2
      60: 00 00 00 00      	nop
      80: 00 00 00 00      	nop
      a0: 00 00 00 00      	nop
      c0: 00 00 00 00      	nop
      e0: 00 00 00 00      	nop

00000100 <.LBB0_1>:
     100: 00 00 00 00      	vmov	za[i0], y0
     120: 00 00 00 00      	nop
     140: 00 00 00 00      	vmov	zb, za
     160: 00 00 00 00      	nop
     180: 00 00 00 00      	nop

000001a0 <.L_LEnd0>:
     1a0: 00 00 00 00      	nop
     1c0: 00 00 00 00      	ret	lr
     1e0: 00 00 00 00      	nop
     200: 00 00 00 00      	nop
     220: 00 00 00 00      	nop
     240: 00 00 00 00      	nop
     260: 00 00 00 00      	nop

00000280 <other>:
     280: 00 00 00 00      	vmov	za[i0], y0
""",
    "zol_lc_first.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	movxm	ls, #0x100
      20: 00 00 00 00      	add.nc	lc, r0, #-0x2
      40: 00 00 00 00      	movxm	le, #0x1a0
      60: 00 00 00 00      	nop
      80: 00 00 00 00      	nop
      a0: 00 00 00 00      	nop
      c0: 00 00 00 00      	nop
      e0: 00 00 00 00      	nop
     100: 00 00 00 00      	vmov	za[i0], y0
     120: 00 00 00 00      	nop
     140: 00 00 00 00      	vmov	zb, za
     160: 00 00 00 00      	nop
     180: 00 00 00 00      	nop
     1a0: 00 00 00 00      	nop
     1c0: 00 00 00 00      	ret	lr
     1e0: 00 00 00 00      	nop
     200: 00 00 00 00      	nop
     220: 00 00 00 00      	nop
     240: 00 00 00 00      	nop
     260: 00 00 00 00      	nop

00000280 <other>:
     280: 00 00 00 00      	vmov	za[i0], y0
""",
    "zol_short.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	nop
      20: 00 00 00 00      	nop
      40: 00 00 00 00      	nop
      60: 00 00 00 00      	nop
      80: 00 00 00 00      	nop
      a0: 00 00 00 00      	movxm	ls, #0x100
      c0: 00 00 00 00      	movxm	le, #0x140
      e0: 00 00 00 00      	add.nc	lc, r0, #-0x2
     100: 00 00 00 00      	vmov	za[i0], y0
     120: 00 00 00 00      	nop
     140: 00 00 00 00      	vmov	zb, za
     160: 00 00 00 00      	ret	lr
     180: 00 00 00 00      	nop
     1a0: 00 00 00 00      	nop
     1c0: 00 00 00 00      	nop
     1e0: 00 00 00 00      	nop
     200: 00 00 00 00      	nop

00000220 <other>:
     220: 00 00 00 00      	vmov	za[i0], y0
""",
    "zol_unaligned.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	movxm	ls, #0x104
      20: 00 00 00 00      	movxm	le, #0x1a0
      40: 00 00 00 00      	add.nc	lc, r0, #-0x2
      60: 00 00 00 00      	nop
      80: 00 00 00 00      	nop
      a0: 00 00 00 00      	nop
      c0: 00 00 00 00      	nop
      e0: 00 00 00 00      	nop
     100: 00 00 00 00      	vmov	za[i0], y0
     120: 00 00 00 00      	nop
     140: 00 00 00 00      	vmov	zb, za
     160: 00 00 00 00      	nop
     180: 00 00 00 00      	nop
     1a0: 00 00 00 00      	nop
     1c0: 00 00 00 00      	ret	lr
     1e0: 00 00 00 00      	nop
     200: 00 00 00 00      	nop
     220: 00 00 00 00      	nop
     240: 00 00 00 00      	nop
     260: 00 00 00 00      	nop

00000280 <other>:
     280: 00 00 00 00      	vmov	za[i0], y0
""",
    "zol_war.dis": """# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

syn.o:	file format elf32-aie

Disassembly of section .text:

00000000 <main>:
       0: 00 00 00 00      	movxm	ls, #0x100
      20: 00 00 00 00      	movxm	le, #0x180
      40: 00 00 00 00      	add.nc	lc, r0, #-0x2
      60: 00 00 00 00      	nop
      80: 00 00 00 00      	nop
      a0: 00 00 00 00      	nop
      c0: 00 00 00 00      	nop
      e0: 00 00 00 00      	nop

00000100 <.LBB0_1>:
     100: 00 00 00 00      	vmov	za[i0], y0
     120: 00 00 00 00      	nop
     140: 00 00 00 00      	vmov	zb, za
     160: 00 00 00 00      	nop

00000180 <.L_LEnd0>:
     180: 00 00 00 00      	nop
     1a0: 00 00 00 00      	ret	lr
     1c0: 00 00 00 00      	nop
     1e0: 00 00 00 00      	nop
     200: 00 00 00 00      	nop
     220: 00 00 00 00      	nop
     240: 00 00 00 00      	nop

00000260 <other>:
     260: 00 00 00 00      	vmov	za[i0], y0
""",
}


def write_fixtures(directory):
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in FIXTURES.items():
        (directory / name).write_text(text)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-fixtures":
        write_fixtures(sys.argv[2])
    else:
        main()
