#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static gate for the KERNEL=aievec core ELFs.

For every project/elfs_main_core_<c>_<r>/*.elf: .text size, exact counts of
commits, MMACs, stream pops, staging inserts and (OUT=i8) fused SRS stores,
one crRnd / crSat write each, no lda.s8, no vector stack traffic, and no
post-incremented staging index (the signature of Peano's
combine_indexed_to_autoinc miscompile).

The counts are exact. A mismatch means the K-loop or a runtime loop was
duplicated or unrolled. A software-pipelined K-loop is one whose previous
K-tile MMACs issue after the loop's commit; it adds one K-tile copy (its
prologue and epilogue). --allow-pipelined accepts such loops and adds that
copy to the expected counts; without it they are errors. --expect-*
overrides the counts for other flag sets.

With --schedule, each K-loop of core (0,2) is checked by
test/aiecc/aie4-mmac-schedule-checker.py. A K-loop is a zero-overhead
loop holding one commit; with KT == 2 the K-loop is inlined and there is
none. The check covers the straight-line block around the loop, starting
after the previous jump, so the prime and the previous tile's prefetch lie
outside it. The block holds the loop entry or pipeline prologue, two loop
iterations, the epilogue or exit, and the peeled last K-tile, so the
back-edge and both loop transitions are checked. A synthetic closing commit
conservatively stands for the tile back edge or strip end. Inserts are
grouped per commit; pops are only counted.
"""

import argparse
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
SCHEDULE_CHECKER = os.path.abspath(
    os.path.join(
        HERE, "..", "..", "..", "test", "aiecc", "aie4-mmac-schedule-checker.py"
    )
)
if not os.path.exists(SCHEDULE_CHECKER):
    sys.exit(
        f"error: {SCHEDULE_CHECKER} not found; check_aievec_core.py must run "
        "from an mlir-aie source tree"
    )
# Share the timing checker's slot patterns (e.g. the immediate-index insert
# `vmov za3, y0`).
schedule_spec = importlib.util.spec_from_file_location(
    "aie4_mmac_schedule", SCHEDULE_CHECKER
)
schedule_module = importlib.util.module_from_spec(schedule_spec)
schedule_spec.loader.exec_module(schedule_module)
COMMIT_RE = schedule_module.COMMIT_RE
INSERT_RE = schedule_module.INSERT_RE
MMAC_RE = schedule_module.MMAC_RE
POP_RE = schedule_module.POP_RE

DUPLICATION = (
    "count mismatch: the K-loop or a runtime loop was duplicated, unrolled or "
    "software-pipelined (compare the commit -> first-MMAC structure of the "
    "loops)"
)

LINE_RE = re.compile(r"^\s*([0-9a-f]+):[^\t]*\t(.*)$")
# Slot patterns, matched at the start of a slot.
PATTERNS = {
    "commit": COMMIT_RE,
    "mmac": MMAC_RE,
    "pop": POP_RE,
    "insert": INSERT_RE,
    "lda.s8": re.compile(r"^lda\.s8\b"),
    "vector [sp": re.compile(r"^v\S*\s.*\[sp\b"),
    "vst.srs": re.compile(r"^vst\.srs\b"),
    "crrnd": re.compile(r"^movx\s+crrnd,"),
    "crsat": re.compile(r"^movx\s+crsat,"),
    "autoinc za": re.compile(r"^vmov\s+\S*za\[i\d+\+\+"),
}
ZOL_RE = re.compile(r"^movxm\s+(ls|le),\s*#(0x[0-9a-f]+|\d+)\s*$")


def tool(name):
    """A Peano binutil: PEANO_INSTALL_DIR/bin first, then PATH."""
    peano = os.environ.get("PEANO_INSTALL_DIR")
    if peano and os.path.exists(os.path.join(peano, "bin", name)):
        return os.path.join(peano, "bin", name)
    path = shutil.which(name)
    if not path:
        sys.exit(f"error: {name} not found (set PEANO_INSTALL_DIR)")
    return path


def bundles(dis):
    """[(address, [slots])] of every function in a disassembly."""
    out = []
    for line in dis.splitlines():
        m = LINE_RE.match(line)
        if m:
            slots = [s.strip() for s in m.group(2).split(";")]
            out.append((int(m.group(1), 16), [s for s in slots if s]))
    return out


def text_size(elf):
    out = subprocess.run(
        [tool("llvm-size"), "-A", elf], check=True, capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        f = line.split()
        if f and f[0] == ".text":
            return int(f[1])
    sys.exit(f"error: {elf}: no .text")


def zol_loops(code):
    """[(start index, last index)] of zero-overhead loops, from ls/le moves.
    The ls and le writes of one loop may come in either order."""
    index = {a: i for i, (a, _) in enumerate(code)}
    pending, loops = {}, []
    for _, slots in code:
        for s in slots:
            m = ZOL_RE.match(s)
            if not m:
                continue
            pending[m.group(1)] = int(m.group(2), 0)
            if len(pending) == 2:
                ls, le = pending["ls"], pending["le"]
                if ls not in index or le not in index:
                    sys.exit(f"error: loop [{ls:#x}, {le:#x}] is not on bundles")
                loops.append((index[ls], index[le]))
                pending = {}
    if pending:
        sys.exit(f"error: unpaired zero-overhead loop register write {pending}")
    return loops


def count(code, lo, hi, key):
    return sum(
        1 for i in range(lo, hi + 1) for s in code[i][1] if PATTERNS[key].match(s)
    )


def first(code, lo, hi, key):
    for i in range(lo, hi + 1):
        if any(PATTERNS[key].match(s) for s in code[i][1]):
            return i
    return None


JUMP_RE = re.compile(r"^(j[a-z]*|ret)\b")
DELAY_SLOTS = 5


def schedule_check(code, start, last, args):
    """Run the timing checker on the straight-line block around the K-loop
    [start, last]: from the bundle after the previous jump to the delay
    slots of the next one, with the loop body executed twice. The block
    holds the tile's K-loop entry (a software-pipelined loop's prologue
    commit), the loop, its epilogue and the peeled last K-tile, so the
    entry, back-edge and exit transitions are all checked. Every commit
    there owns --mmacs-per-ktile MMACs; the inserts before the first commit
    belong to it, and the next commits own --pops-per-ktile inserts each. A
    closing commit appended after the block takes the rest (the next tile's
    prefetch). Pops are not grouped: the scheduler may pop a K-tile's W
    before the previous commit, and only the inserts write staging."""

    def is_jump(i):
        return any(JUMP_RE.match(x) for x in code[i][1])

    lo = start
    while lo > 0 and not is_jump(lo - 1):
        lo -= 1
    hi = last + 1
    while hi < len(code) and not is_jump(hi):
        hi += 1
    hi = min(hi + DELAY_SLOTS, len(code) - 1)
    block = [
        (a, [x for x in sl if not JUMP_RE.match(x)]) for a, sl in code[lo : hi + 1]
    ]

    def n(key, rng):
        return sum(1 for _, sl in rng for x in sl if PATTERNS[key].match(x))

    body = code[start : last + 1]
    pre, post = block[: start - lo], block[last + 1 - lo :]
    commits = n("commit", pre) + 2 * n("commit", body) + n("commit", post)
    mmacs = n("mmac", pre) + 2 * n("mmac", body) + n("mmac", post)
    m, p = args.mmacs_per_ktile, args.pops_per_ktile
    if mmacs != m * commits:
        return 1, (
            f"error: {mmacs} MMACs for {commits} commits in the K-loop block; "
            f"expected {m} per commit"
        )
    path = pre + body + body + post
    first_commit = next(
        k for k, (_, sl) in enumerate(path) if any(COMMIT_RE.match(x) for x in sl)
    )
    head_ins = n("insert", path[:first_commit])
    ins_rest = n("insert", path) - head_ins - p * (commits - 1)
    lines = ["00000000 <kloop>:"]
    for a, sl in block:
        lines.append(f"{a:8x}:      \t" + ";\t\t".join(sl or ["nop"]))
    lines.append(f"{block[-1][0] + 0x20:8x}:      \tvmov\tzb, za")
    mid = [p] * (commits - 1)
    cmd = [
        sys.executable,
        SCHEDULE_CHECKER,
        "-",
        "--function",
        "kloop",
        "--mmacs-per-tile",
        ",".join(map(str, [m] * commits + [0])),
        "--inserts-per-tile",
        ",".join(map(str, [head_ins] + mid + [ins_rest])),
        "--forbid-lda-s8",
    ] + (["--allow-acc-spills"] if args.allow_spills else [])
    with tempfile.NamedTemporaryFile("w", suffix=".dis", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        cmd[2] = f.name
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    return r.returncode, (r.stdout + r.stderr).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("project", help="aiecc --tmpdir directory")
    p.add_argument("--out", choices=["i8", "u8", "i32", "f32"], required=True)
    p.add_argument(
        "--float",
        action="store_true",
        help="fp8/bf16 core: zero_if is integer-only, so K-tile 0 is peeled "
        "(three commits per tile body, one more K-tile of MMACs and pops "
        "unless K has only two K-tiles)",
    )
    p.add_argument("--max-text", type=int, default=16384)
    p.add_argument(
        "--tile-copies",
        type=int,
        default=2,
        help="tile bodies per core: 2 (output-tile loop + peeled), 1 if NG*CT == 1",
    )
    p.add_argument(
        "--kt",
        type=int,
        help="K-tiles per output tile; sets the float K-tile structure "
        "(default: three commits per tile body unless --expect-kloops 0)",
    )
    p.add_argument("--expect-commits", type=int)
    p.add_argument("--expect-mmacs", type=int)
    p.add_argument("--expect-pops", type=int)
    p.add_argument("--expect-inserts", type=int)
    p.add_argument("--expect-srs", type=int)
    p.add_argument("--schedule", action="store_true", help="check K-loop timing")
    p.add_argument(
        "--expect-kloops",
        type=int,
        help="K-loops per core (default: --tile-copies; 0 when K == 128, where "
        "the one-iteration K-loop is inlined)",
    )
    p.add_argument(
        "--allow-pipelined",
        action="store_true",
        help="accept software-pipelined K-loops (MMACs of the previous "
        "K-tile before the loop's commit); each adds one K-tile copy (a "
        "commit, --mmacs-per-ktile MMACs and --pops-per-ktile pops and "
        "inserts) to the expected counts",
    )
    p.add_argument(
        "--allow-spills",
        action="store_true",
        help="do not gate vector stack traffic (K == 128: Peano spills "
        "accumulators when both K-tiles are straight-line code; fp8/bf16: "
        "Peano spills about a quarter of the accumulators around the last "
        "K-tile)",
    )
    p.add_argument("--mmacs-per-ktile", type=int, default=64)
    p.add_argument("--pops-per-ktile", type=int, default=32)
    args = p.parse_args()

    copies = args.tile_copies
    # K-tiles issued ahead of the last one in each tile body: the K-loop, plus
    # the peeled first K-tile for float types. With two K-tiles (no K-loop)
    # both types have one lead K-tile. A float core at three K-tiles inlines
    # its one-iteration K-loop but still has three commits.
    if args.kt is not None:
        lead = 2 if args.float and args.kt > 2 else 1
    else:
        lead = 2 if args.float and args.expect_kloops != 0 else 1
    # Prime 32 pops; the prefetching copy pops one K-tile of W per lead tile
    # and its last K-tile, the peeled copy only per lead tile.
    pops = 32 + (32 * (lead + 1) if copies == 2 else 0) + 32 * lead
    expect = {
        "commit": args.expect_commits or (lead + 1) * copies,
        "mmac": args.expect_mmacs or 64 * (lead + 1) * copies,
        "pop": args.expect_pops or pops,
        "insert": args.expect_inserts or pops,
        "lda.s8": 0,
        "vector [sp": 0,
        "autoinc za": 0,
    }
    if args.allow_spills:
        del expect["vector [sp"]
    kloops_expected = copies if args.expect_kloops is None else args.expect_kloops
    if args.out in ("i8", "u8"):
        expect.update(
            {"vst.srs": args.expect_srs or 64 * copies, "crrnd": 1, "crsat": 1}
        )

    elfs = sorted(glob.glob(os.path.join(args.project, "elfs_main_core_*", "*.elf")))
    if not elfs:
        sys.exit(f"error: no core ELFs under {args.project}/elfs_main_core_*")
    errors = []
    for elf in elfs:
        core = os.path.basename(os.path.dirname(elf))[len("elfs_main_") :]
        dis = subprocess.run(
            [tool("llvm-objdump"), "-d", "--no-show-raw-insn", elf],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        code = bundles(dis)
        counts = {k: 0 for k in PATTERNS}
        for _, slots in code:
            for s in slots:
                for k, rx in PATTERNS.items():
                    if rx.match(s):
                        counts[k] += 1
        size = text_size(elf)
        loops = zol_loops(code)
        # A K-loop is a zero-overhead loop holding one K-tile: one commit.
        # With KT == 2 the K-loop has one iteration and is inlined; the
        # output-tile loop (two commits) is then the innermost loop.
        kloops = [(s, e) for s, e in loops if count(code, s, e, "commit") == 1]
        shown = " ".join(f"{k} {counts[k]}" for k in expect)
        if args.allow_spills:
            shown += f" (vector [sp {counts['vector [sp']}, allowed)"
        print(f"{core}: .text {size} B; {shown}; ZOLs {len(loops)}", end="")
        for s, e in kloops:
            c = first(code, s, e, "commit")
            mm = first(code, s, e, "mmac")
            dist = (
                f", commit->first MMAC {mm - c}"
                if c is not None and mm is not None
                else ""
            )
            print(f"; K-loop {e - s + 1} bundles{dist}", end="")
        if not kloops:
            print("; no K-loop", end="")
        # A software-pipelined K-loop issues the previous K-tile's last MMACs
        # no later than its commit; its prologue and epilogue add one K-tile
        # copy. An unpipelined loop's first MMAC always follows the commit
        # (it reads the committed matrix), so equal bundles also count.
        piped = sum(
            1
            for s, e in kloops
            if first(code, s, e, "mmac") <= first(code, s, e, "commit")
        )
        if piped:
            print(f"; {piped} pipelined", end="")
        print()
        core_expect = dict(expect)
        if piped and not args.allow_pipelined:
            errors.append(
                f"{core}: {piped} software-pipelined K-loop(s); pass "
                "--allow-pipelined to accept them"
            )
        elif piped:
            extra = {
                "commit": 1,
                "mmac": args.mmacs_per_ktile,
                "pop": args.pops_per_ktile,
                "insert": args.pops_per_ktile,
            }
            for k, v in extra.items():
                core_expect[k] += piped * v
        if len(kloops) != kloops_expected:
            errors.append(
                f"{core}: {len(kloops)} K-loop(s), expected {kloops_expected}: "
                f"{DUPLICATION}"
            )
        if size > args.max_text:
            errors.append(f"{core}: .text {size} B exceeds {args.max_text} B")
        for k, n in core_expect.items():
            if counts[k] != n:
                why = ""
                if k in ("commit", "mmac", "pop", "insert", "vst.srs"):
                    why = f": {DUPLICATION}"
                elif k == "lda.s8":
                    why = ": A rows need vector loads with alignment = 64"
                elif k == "vector [sp":
                    why = ": accumulator or vector spills"
                elif k in ("crrnd", "crsat"):
                    why = ": modes must be written once, at core entry"
                elif k == "autoinc za":
                    why = (
                        ": Peano combine_indexed_to_autoinc folded a staging "
                        "slot index into a post-increment (loop counter hazard)"
                    )
                errors.append(f"{core}: {k} {counts[k]}, expected {n}{why}")
        if args.schedule and core == "core_0_2":
            for s, e in kloops:
                rc, out = schedule_check(code, s, e, args)
                tag = f"core_0_2 K-loop [{code[s][0]:#x}, {code[e][0]:#x}]"
                print(f"{tag}: " + out.replace("\n", f"\n{tag}: "))
                if rc:
                    errors.append(f"{tag}: schedule check failed")
    if errors:
        sys.exit("\n".join("error: " + e for e in errors))
    print(f"CHECK_AIEVEC OK: {len(elfs)} cores")


if __name__ == "__main__":
    main()
