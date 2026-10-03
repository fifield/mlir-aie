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
"""

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
    p.add_argument("--inserts-per-tile", type=int, default=32)
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

    inserts, commits, mmacs, errors = [], [], [], []
    for cycle, slots in enumerate(path):
        for s in slots:
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
    if len(commits) != tiles:
        errors.append(f"expected {tiles} commits, found {len(commits)}")
    if len(inserts) != args.inserts_per_tile * tiles:
        errors.append(
            f"expected {args.inserts_per_tile * tiles} inserts, found {len(inserts)}"
        )
    if len(mmacs) != sum(per_tile):
        errors.append(f"expected {sum(per_tile)} MMACs, found {len(mmacs)}")
    if errors:
        sys.exit("\n".join("error: " + e for e in errors))

    inserts.sort()
    mmacs.sort()
    commits.sort()
    first = 0
    for g in range(tiles):
        read = commits[g] + COMMIT_READ
        write = commits[g] + COMMIT_WRITE
        group = inserts[g * args.inserts_per_tile : (g + 1) * args.inserts_per_tile]
        if group[-1] >= read:
            errors.append(
                f"tile {g}: an insert writes za at {group[-1]}, not before the "
                f"commit reads it at {read}"
            )
        if g + 1 < tiles:
            nxt = inserts[(g + 1) * args.inserts_per_tile]
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
    print(
        f"ORDER OK: {tiles} tiles, {len(inserts)} inserts, {len(mmacs)} MMACs, "
        f"no accumulator spills{shape}"
    )


if __name__ == "__main__":
    main()
