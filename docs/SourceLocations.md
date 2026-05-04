# Source Location Information in mlir-aie

Investigation note: how MLIR `Location` info flows (or doesn't) through the
mlir-aie compiler today, and a proposal — with an initial proof-of-concept
patch — for carrying user source positions from IRON Python down through
transformation passes for use by diagnostics, debugging, and profiling.

## Why locations matter

Every MLIR op carries a `Location` attribute. Upstream MLIR uses it for:

- **Diagnostics**: errors, warnings, and `op->emitError()` cite the source
  position the op came from.
- **Pass debugging**: `--mlir-print-ir-after-all --mlir-print-debuginfo`
  shows how IR transforms while preserving the user-visible source mapping.
- **Debug info translation**: when lowering to the LLVM dialect,
  `mlir/lib/Target/LLVMIR/DebugTranslation.cpp` maps MLIR `Location`s to
  `llvm::DILocation` so the resulting object code carries DWARF that
  debuggers (gdb, lldb) and profilers can consume.
- **Provenance tagging**: `NameLoc` / `FusedLoc` let passes record *which*
  pass or transform produced an op, while still pointing at the original
  source — useful for understanding what canonicalization, fusion, or
  lowering did to a user-written op.

mlir-aie currently makes essentially no use of any of this.

## Current state

A survey of the source tree (mlir-aie commit on branch `mdv6`):

- **C++ passes**: 136+ call sites of `builder.getUnknownLoc()` /
  `UnknownLoc::get(...)` across 20 files. The hotspots are
  `AIEObjectFifoStatefulTransform.cpp` (38), `AIECreatePathFindFlows.cpp`
  (19), `AIECoreToStandard.cpp` (13), `AIEVecToLLVM.cpp` (8),
  `AIEHerdRouting.cpp` (6), and `AIEToConfiguration.cpp` (5). Most
  transformation patterns synthesize new ops without forwarding
  `op->getLoc()` from the source op being lowered or replaced. There is no
  use of `FusedLoc` or `NameLoc` anywhere.
- **Python frontend (IRON)**: Every IRON `resolve()` constructs MLIR ops
  inside a `with mlir_mod_ctx():` block, which the wrapper enters with a
  default `with Location.unknown():`. None of the user-facing IRON classes
  (`ObjectFifo`, `Worker`, `Runtime.fill/drain`, `Kernel`, `Buffer`)
  capture the Python call frame, and the `loc=` parameter on `resolve()`
  is unused.
- **aiecc.py**: 16 `with Location.unknown():` wrappings around module
  parses and post-load fixups. These are fine *as defaults* but mean any
  op created in those scopes gets `UnknownLoc` unless explicitly
  overridden.
- **Tests / examples**: zero `loc(...)` annotations in `.mlir` test inputs;
  no lit tests use `--mlir-print-debuginfo`.
- **Targets**: `lib/Targets/AIERT.cpp` and the CDO/NPU emitters do not
  consume location info — when they emit transactions or runtime config,
  there is nothing to consume because everything upstream is `UnknownLoc`.

In short: locations are dropped at the front door (Python) and again at
every transformation pass. There is nothing technical preventing their use;
it is purely a matter of plumbing.

## Infrastructure already in place

The pieces needed for this to work are all available:

- `mlir.ir.Location` Python bindings expose `Location.file`, `Location.name`
  (NameLoc), `Location.callsite` (CallSiteLoc), and `Location.fused`. They
  act as context managers — `with Location.file(...):` makes the location
  the thread-local default, so ops created without an explicit `loc=`
  inherit it.
- `aie/extras/util.py::get_user_code_loc()` already walks the Python frame
  stack to capture the user's source position. It is used today in a few
  places (`scf.if_`, buffer indexing in `dialects/aie.py`).
- C++ `OpBuilder` and `RewriterBase` already accept `Location` everywhere;
  `rewriter.replaceOpWithNewOp<T>(op, ...)` automatically propagates
  `op->getLoc()` to the replacement. Patterns just need to use it.
- LLVM `DebugTranslation` already turns MLIR `FileLineColLoc` into
  `DILocation` when lowering to the LLVM dialect — so once locations exist
  in the IR they would land in DWARF on the AIE-core side without further
  effort.

## Proposed approach (three phases)

### Phase 1 — Capture in Python (this POC)

At the IRON-level "this op exists because the user wrote *that*" boundary,
walk the Python stack and stash an `ir.Location` on the IRON object. When
`resolve()` builds MLIR ops, enter a `with self._user_loc:` block so the
ops inherit a `FileLineColLoc` (wrapped in a `NameLoc` carrying the IRON
identifier — `of0`, the worker's `core_fn` name, etc.).

Caveats:

- Frame capture must skip frames inside the IRON / aie / venv site-packages
  trees so the recorded position is the user's program, not framework
  internals. The helper in `python/iron/_loc.py` does this.
- Locations are captured at *construction*, not at `resolve()` time, because
  by the time `Program.resolve_program` walks the object graph the user's
  call frame is long gone.
- `Location` requires an active MLIR `Context`. IRON construction often
  happens before `mlir_mod_ctx()` is entered, so `capture_user_loc()`
  returns `None` in that case and the existing `Location.unknown()` default
  is preserved. (A future refinement is to capture the `(filename, line,
  col)` triple eagerly into a plain dataclass and lazily materialize the
  `Location` inside `resolve()`.)

### Phase 2 — Preserve through passes

Audit each pass under `lib/Conversion/`, `lib/Dialect/AIE/Transforms/`, and
`lib/Dialect/AIEX/Transforms/` and replace `builder.getUnknownLoc()` /
`b.getUnknownLoc()` with `op->getLoc()` (or, where multiple input ops
contribute, `builder.getFusedLoc({locA, locB, ...})`). Idiomatic MLIR
patterns to encourage:

- Use `rewriter.replaceOpWithNewOp<NewOp>(oldOp, ...)` rather than
  constructing the new op with an unknown location. This carries the
  source op's location automatically.
- For passes that synthesize many ops from one logical source op (like
  `AIEObjectFifoStatefulTransform` expanding an `objectfifo.create` into
  a forest of `lock` / `buffer` / `dma_bd` / `use_lock` ops), capture the
  source loc once at the top of the loop and pass it to every
  `Op::create(...)`. Optionally wrap with a `NameLoc` indicating which
  pass produced the synthesized op (e.g. `loc("ObjFifoExpand"(...))`) so
  diagnostics make the provenance clear.

### Phase 3 — Backends and tooling

Once locations survive the pass pipeline:

- **`aie-translate` / LLVM IR emission**: with the LLVM-dialect
  `DebugTranslation` already in place, AIE-core LLVM IR gets `DILocation`
  attached automatically. Ensure `aie-translate` does not strip debuginfo
  on the way out (likely a one-line change, if any).
- **CDO / NPU runtime emitters** (`lib/Targets/AIERT.cpp`,
  `AIETargetCDODirect.cpp`): consume locations to (a) embed a small
  source-line ↔ instruction map next to the transaction binary, and (b)
  annotate trace records with the originating IRON construct so profiling
  tools can attribute hot transactions back to the user's `Runtime.fill`
  call.
- **aiecc.py**: expose `--mlir-print-debuginfo` and a `--keep-loc` flag
  that preserves rather than canonicalizes locations across pass runs.
- **Tests**: add lit tests that pipe IRON-built modules through
  `aie-opt --mlir-print-debuginfo` and `FileCheck` for `loc(#...)`
  annotations, locking in the contract that key passes do not drop
  locations.

## POC patch summary

This branch contains a small end-to-end demonstration of Phase 1 plus a
Phase 2 example:

- `python/iron/_loc.py` — new helper. `capture_user_loc(name=...)` walks
  the Python frame stack, skipping frames inside the IRON / aie / venv
  trees, and returns `NameLoc.get(name, FileLineColLoc(...))`. Returns
  `None` when no MLIR context is active so callers can fall back safely.
- `python/iron/dataflow/objectfifo.py`, `python/iron/worker.py`,
  `python/iron/kernel.py`, `python/iron/runtime/runtime.py`,
  `python/iron/runtime/dmatask.py` — capture `self._user_loc` at
  construction time (or at the `Runtime.fill` / `drain` call site) and
  apply it as a `with` block around MLIR op creation in `resolve()`.
- `lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` —
  replace `builder.getUnknownLoc()` with `op.getLoc()` in
  `createObjectFifo`, `createObjectFifoLocks`, and the buffer allocation
  in `createObjectFifoElements`. The synthesized `ObjectFifoCreateOp`,
  `LockOp`s (prod / cons), and `BufferOp`s now inherit the source
  ObjectFifo's location.

After build, an IRON program emitted with
`aie-opt --mlir-print-debuginfo` (or by setting
`enable_debug_info=True` when printing the module from Python) will show
ops tagged with `loc("of0"("user.py":42:4))` style annotations, and those
annotations survive the objectfifo stateful transform onto the locks and
buffers it materializes.

## Effort and risk estimate

- Phase 1 (Python capture): ~1 day. Low risk — additive only, falls back
  to `Location.unknown()` cleanly.
- Phase 2 (pass plumbing): a few days of mechanical work, file-by-file
  audit, plus a lit test or two per pass to lock the behavior in. Medium
  risk only because some passes do non-trivial op-fusion; deciding when
  to use `FusedLoc` vs. picking one source loc requires judgment.
- Phase 3 (backend / tooling): a few days, mostly straightforward once
  the IR carries locations end-to-end. The DWARF path on AIE-core code is
  already implemented upstream.
