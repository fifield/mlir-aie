//===- AIE4CheckStaging.cpp - AIE4 flat staging lint ------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Reject matmul/flat-op coexistence and check staging provenance per scope.
// Track 32 slots and the active matrix with Defined(key) < Conflict < Undef.
// Branches join states; loops reach a fixed point before diagnostics are
// emitted, covering both the first iteration and back edges.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/Transforms/AIE4Staging.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/XLLVM/AIE4MatrixState.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/MapVector.h"

#include <array>

namespace xilinx::aievec {
#define GEN_PASS_DEF_AIEVECAIE4CHECKSTAGING
#include "aie/Dialect/AIEVec/Transforms/Passes.h.inc"
} // namespace xilinx::aievec

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

bool aievec::isAIE4FlatMatrixOp(Operation *op) {
  return isa<AIE4StageOp, AIE4StageStreamOp, AIE4CommitOp, AIE4MmacActiveOp>(
      op);
}

Operation *aievec::getAIE4ExecutionScope(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<AIE::DeviceOp, ModuleOp>(parent))
      continue;
    if (isa<AIE::CoreOp, FunctionOpInterface>(parent) ||
        parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return parent;
  }
  return nullptr;
}

/// Execution scopes holding a flat op, each with its first flat op in walk
/// order.
static llvm::MapVector<Operation *, Operation *>
collectFlatScopesWithFirstOp(Operation *root) {
  llvm::MapVector<Operation *, Operation *> scopes;
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isAIE4FlatMatrixOp(op))
      return;
    if (Operation *scope = getAIE4ExecutionScope(op))
      scopes.insert({scope, op});
  });
  return scopes;
}

llvm::DenseSet<Operation *> aievec::collectAIE4FlatScopes(Operation *root) {
  llvm::DenseSet<Operation *> scopes;
  for (auto &entry : collectFlatScopesWithFirstOp(root))
    scopes.insert(entry.first);
  return scopes;
}

namespace {

//===----------------------------------------------------------------------===//
// Op classification
//===----------------------------------------------------------------------===//

bool isCallLike(Operation *op) {
  return isa<CallOpInterface>(op) ||
         op->getName().getStringRef() == "llvm.call_intrinsic";
}

/// May replace staging or the matrix with unknown contents. Same tests as
/// the matrix_region body verifier.
bool isClobber(Operation *op) {
  return !op->isRegistered() || isCallLike(op) ||
         op->getName().getStringRef() == "llvm.inline_asm";
}

bool isAIEVecOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<AIEVecDialect>(dialect);
}

bool isMatrixStateOp(Operation *op) {
  return isAIE4FlatMatrixOp(op) || isa<AIE4MatrixRegionOp, AIE4MatMulOp>(op) ||
         xllvm::isAIE4MatrixStateWriter(op) ||
         op->hasTrait<xllvm::AIE4MatrixReader>() || isClobber(op);
}

//===----------------------------------------------------------------------===//
// Lattice
//===----------------------------------------------------------------------===//

constexpr unsigned kNumSlots = 32;

/// Staging/matrix contents identity. int8 is sign-agnostic; int4 staging
/// converts by signedness, so its key includes it.
struct Key {
  AIE4MatrixFormat fmt = AIE4MatrixFormat::Bf16;
  int8_t sign = -1;
  bool operator==(const Key &o) const { return fmt == o.fmt && sign == o.sign; }
  bool operator!=(const Key &o) const { return !(*this == o); }
};

Key makeKey(AIE4MatrixFormat fmt, BoolAttr rhsSigned) {
  Key k;
  k.fmt = fmt;
  if (fmt == AIE4MatrixFormat::Int4)
    k.sign = rhsSigned && rhsSigned.getValue() ? 1 : 0;
  return k;
}

std::string keyString(const Key &k) {
  std::string s = stringifyAIE4MatrixFormat(k.fmt).str();
  if (k.fmt == AIE4MatrixFormat::Int4)
    s += k.sign ? " (rhs_signed = true)" : " (rhs_signed = false)";
  return s;
}

enum class Kind : uint8_t { Defined, Conflict, Undef };

struct Cell {
  Kind kind = Kind::Undef;
  Key key;
  // Defined: last writer or commit. Undef: clobber, or nullptr if never
  // written. Conflict: a writer on one path.
  Operation *why = nullptr;
  // Conflict only: a writer on another path, with a key (`key2`) that
  // differs from `key`, the key of `why`.
  Operation *why2 = nullptr;
  Key key2;
  // Undef produced by an already-reported error: suppress cascades.
  bool diagnosed = false;

  static Cell defined(Key k, Operation *op) {
    Cell c;
    c.kind = Kind::Defined;
    c.key = k;
    c.why = op;
    return c;
  }
  static Cell undef(Operation *op, bool diagnosed = false) {
    Cell c;
    c.why = op;
    c.diagnosed = diagnosed;
    return c;
  }
  bool isDefinedAs(const Key &k) const {
    return kind == Kind::Defined && key == k;
  }
  bool sameAs(const Cell &o) const {
    return kind == o.kind && (kind != Kind::Defined || key == o.key);
  }
};

/// Least upper bound. An Undef result takes `why` and `diagnosed` only from
/// Undef operands: a Defined operand's `why` is a writer, not a clobber.
Cell join(const Cell &a, const Cell &b) {
  if (a.kind == Kind::Defined && b.kind == Kind::Defined && a.key == b.key)
    return a;
  if (a.kind == Kind::Undef || b.kind == Kind::Undef) {
    Cell c;
    for (const Cell *u : {&a, &b}) {
      if (u->kind != Kind::Undef)
        continue;
      if (!c.why)
        c.why = u->why;
      c.diagnosed |= u->diagnosed;
    }
    return c;
  }
  // Keep two distinct keys so diagnostics always identify a wrong-format
  // writer.
  SmallVector<std::pair<Operation *, Key>, 4> writers;
  for (const Cell *x : {&a, &b}) {
    writers.push_back({x->why, x->key});
    if (x->kind == Kind::Conflict)
      writers.push_back({x->why2, x->key2});
  }
  Cell c;
  c.kind = Kind::Conflict;
  c.why = writers.front().first;
  c.key = writers.front().second;
  for (auto &[why, key] : writers)
    if (key != c.key) {
      c.why2 = why;
      c.key2 = key;
      break;
    }
  return c;
}

struct State {
  std::array<Cell, kNumSlots> slot;
  Cell matrix;

  void setAll(const Cell &c) {
    slot.fill(c);
    matrix = c;
  }
  bool sameAs(const State &o) const {
    for (unsigned i = 0; i < kNumSlots; ++i)
      if (!slot[i].sameAs(o.slot[i]))
        return false;
    return matrix.sameAs(o.matrix);
  }
};

State join(const State &a, const State &b) {
  State r;
  for (unsigned i = 0; i < kNumSlots; ++i)
    r.slot[i] = join(a.slot[i], b.slot[i]);
  r.matrix = join(a.matrix, b.matrix);
  return r;
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

class ProvenanceChecker {
public:
  explicit ProvenanceChecker(bool strict) : strict(strict) {}

  /// Analyse every region of `scope` from an all-Undef entry state.
  void checkScope(Operation *scope) {
    for (Region &region : scope->getRegions()) {
      if (region.empty())
        continue;
      if (!region.hasOneBlock()) {
        if (regionContainsMatrixState(region))
          reportMultiBlock(scope, /*report=*/true);
        continue;
      }
      State state;
      transfer(region.front(), state, /*report=*/true);
    }
  }

  bool hadError() const { return errors; }

private:
  bool strict;
  bool errors = false;
  // Report each wrong-format writer once, even if it reaches several commits.
  DenseSet<Operation *> reportedWriters;
  DenseMap<Operation *, bool> containsCache;

  InFlightDiagnostic diag(Operation *op, const Twine &msg) {
    if (strict) {
      errors = true;
      return op->emitError(msg);
    }
    return op->emitWarning(msg);
  }

  //--- Subtree classification ----------------------------------------------

  /// True if `op` is, or contains, a matrix-state op. Does not descend into
  /// matrix_region bodies (the region counts) or isolated-from-above ops.
  bool containsMatrixState(Operation *op) {
    auto it = containsCache.find(op);
    if (it != containsCache.end())
      return it->second;
    bool result = isMatrixStateOp(op);
    if (!result && !op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      for (Region &region : op->getRegions())
        if (regionContainsMatrixState(region)) {
          result = true;
          break;
        }
    containsCache[op] = result;
    return result;
  }

  bool regionContainsMatrixState(Region &region) {
    for (Block &block : region)
      for (Operation &op : block)
        if (containsMatrixState(&op))
          return true;
    return false;
  }

  /// The first matrix-state op under `op` in walk order (same descent rules
  /// as containsMatrixState).
  Operation *firstMatrixStateOp(Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &child : block) {
          if (isMatrixStateOp(&child))
            return &child;
          if (child.hasTrait<OpTrait::IsIsolatedFromAbove>() ||
              !containsMatrixState(&child))
            continue;
          if (Operation *found = firstMatrixStateOp(&child))
            return found;
        }
    return nullptr;
  }

  //--- Diagnostics ---------------------------------------------------------

  void reportUnsupported(Operation *op, bool report) {
    if (!report)
      return;
    InFlightDiagnostic d =
        diag(op, "'" + op->getName().getStringRef() +
                     "' contains matrix-state operations; "
                     "aievec-aie4-check-staging supports only scf.for, scf.if "
                     "and aievec.aie4.matrix_region around them");
    if (Operation *first = firstMatrixStateOp(op))
      d.attachNote(first->getLoc()) << "matrix-state operation here";
  }

  void reportMultiBlock(Operation *parent, bool report) {
    if (!report)
      return;
    diag(parent, "region with multiple blocks contains matrix-state "
                 "operations; unstructured control flow is not supported "
                 "here");
  }

  /// Diagnose an unwritten, clobbered or wrong-format staging slot.
  void reportSlot(AIE4CommitOp commit, unsigned i, const Cell &cell,
                  const Key &key) {
    std::string prefix = "'aievec.aie4.commit' ";
    if (cell.kind == Kind::Defined) {
      InFlightDiagnostic d = diag(commit, prefix + "expects " + keyString(key) +
                                              " but staging slot " + Twine(i) +
                                              " holds " + keyString(cell.key));
      d.attachNote(cell.why->getLoc()) << "written here";
      return;
    }
    if (cell.kind == Kind::Conflict) {
      InFlightDiagnostic d = diag(
          commit, prefix + "expects " + keyString(key) + " but staging slot " +
                      Twine(i) + " holds different formats on different paths");
      if (cell.why)
        d.attachNote(cell.why->getLoc()) << "written here";
      if (cell.why2)
        d.attachNote(cell.why2->getLoc()) << "written here";
      return;
    }
    if (cell.why && isCallLike(cell.why)) {
      InFlightDiagnostic d =
          diag(commit, prefix + "reads staging slot " + Twine(i) +
                           " after a call that may "
                           "clobber it");
      d.attachNote(cell.why->getLoc()) << "call here";
      return;
    }
    InFlightDiagnostic d =
        diag(commit, prefix + "reads staging slot " + Twine(i) +
                         ", which may be unwritten or clobbered on some path");
    if (!cell.why)
      return;
    if (isa<AIE4StageOp, AIE4StageStreamOp>(cell.why))
      d.attachNote(cell.why->getLoc()) << "written with a dynamic index here";
    else
      d.attachNote(cell.why->getLoc()) << "clobbered here";
  }

  void reportMatrix(AIE4MmacActiveOp mmac, const Cell &cell, const Key &key) {
    if (cell.kind == Kind::Defined) {
      InFlightDiagnostic d = diag(
          mmac, "'aievec.aie4.mmac_active' expects a committed " +
                    keyString(key) + " matrix but the committed matrix holds " +
                    keyString(cell.key));
      d.attachNote(cell.why->getLoc()) << "committed here";
      return;
    }
    InFlightDiagnostic d =
        diag(mmac, "'aievec.aie4.mmac_active' may read an undefined matrix: "
                   "no commit of a fully staged " +
                       keyString(key) + " tile reaches it on every path");
    if (cell.kind == Kind::Conflict) {
      if (cell.why)
        d.attachNote(cell.why->getLoc()) << "committed here";
      if (cell.why2)
        d.attachNote(cell.why2->getLoc()) << "committed here";
    } else if (cell.why) {
      d.attachNote(cell.why->getLoc()) << "clobbered here";
    }
  }

  //--- Transfer functions --------------------------------------------------

  void writeSlot(State &state, Value slot, const Key &key, Operation *op) {
    APInt c;
    if (matchPattern(slot, m_ConstantInt(&c))) {
      int64_t i = c.getSExtValue();
      if (i >= 0 && i < static_cast<int64_t>(kNumSlots))
        state.slot[i] = Cell::defined(key, op);
      return;
    }
    // A dynamic index may write any slot. A slot that becomes Undef with no
    // clobber records the dynamic writer for the diagnostic note.
    for (Cell &cell : state.slot) {
      cell = join(cell, Cell::defined(key, op));
      if (cell.kind == Kind::Undef && !cell.why)
        cell.why = op;
    }
  }

  void transferCommit(AIE4CommitOp commit, State &state, bool report) {
    Key key = makeKey(commit.getFormat(), commit.getRhsSignedAttr());
    // Report the first failing slot that is not already diagnosed, so a
    // different bad write behind a silenced slot is still reported.
    bool failed = false;
    for (unsigned i = 0; i < kNumSlots; ++i) {
      const Cell &cell = state.slot[i];
      if (cell.isDefinedAs(key))
        continue;
      failed = true;
      if (!report)
        break;
      // Identify the wrong-format writer to suppress duplicate diagnostics.
      Operation *offender = nullptr;
      if (cell.kind == Kind::Defined)
        offender = cell.why;
      else if (cell.kind == Kind::Conflict)
        offender = cell.key != key ? cell.why : cell.why2;
      if ((cell.kind == Kind::Undef && cell.diagnosed) ||
          (offender && reportedWriters.contains(offender)))
        continue;
      reportSlot(commit, i, cell, key);
      if (offender)
        reportedWriters.insert(offender);
      // Report one bad staging image once: later commits stay silent on
      // the slots that failed here.
      for (Cell &bad : state.slot)
        if (!bad.isDefinedAs(key))
          bad = Cell::undef(bad.why, /*diagnosed=*/true);
      break;
    }
    if (failed)
      state.matrix = Cell::undef(commit, /*diagnosed=*/report);
    else
      state.matrix = Cell::defined(key, commit);
  }

  void transferMmac(AIE4MmacActiveOp mmac, State &state, bool report) {
    Key key = makeKey(mmac.getFormat(), mmac.getRhsSignedAttr());
    if (state.matrix.isDefinedAs(key))
      return;
    if (!report)
      return;
    if (!state.matrix.diagnosed)
      reportMatrix(mmac, state.matrix, key);
    state.matrix.diagnosed = true;
  }

  /// A `scf.for` that writes slot i on iteration i, for i in [0, 32), and
  /// has no other matrix-state op. Returns its staging op.
  Operation *matchInductionLoop(scf::ForOp loop) {
    auto isConst = [](Value v, int64_t expected) {
      std::optional<int64_t> c = getConstantIntValue(v);
      return c && *c == expected;
    };
    if (!isConst(loop.getLowerBound(), 0) ||
        !isConst(loop.getUpperBound(), kNumSlots) ||
        !isConst(loop.getStep(), 1))
      return nullptr;
    Operation *stage = nullptr;
    for (Operation &op : *loop.getBody()) {
      if (isa<AIE4StageOp, AIE4StageStreamOp>(op)) {
        if (stage)
          return nullptr;
        stage = &op;
        continue;
      }
      if (containsMatrixState(&op))
        return nullptr;
    }
    if (!stage)
      return nullptr;
    Value slot = isa<AIE4StageOp>(stage)
                     ? cast<AIE4StageOp>(stage).getSlot()
                     : cast<AIE4StageStreamOp>(stage).getSlot();
    while (Operation *def = slot.getDefiningOp()) {
      if (!isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::TruncIOp,
               arith::ExtSIOp, arith::ExtUIOp>(def))
        break;
      // Every cast in the chain must preserve [0, 32): 6 bits covers the
      // signed and unsigned readings.
      auto narrow = [](Type t) {
        return !t.isIndex() && t.getIntOrFloatBitWidth() < 6;
      };
      if (narrow(def->getOperand(0).getType()) ||
          narrow(def->getResult(0).getType()))
        return nullptr;
      slot = def->getOperand(0);
    }
    return slot == loop.getInductionVar() ? stage : nullptr;
  }

  /// Transfer through `region` (a loop body or branch). Returns false if the
  /// region is unsupported and the state was set to diagnosed Undef.
  bool transferRegion(Operation *parent, Region &region, State &state,
                      bool report) {
    if (region.empty())
      return true;
    if (!region.hasOneBlock()) {
      if (!regionContainsMatrixState(region))
        return true;
      reportMultiBlock(parent, report);
      state.setAll(Cell::undef(nullptr, /*diagnosed=*/report));
      return false;
    }
    transfer(region.front(), state, report);
    return true;
  }

  void transferFor(scf::ForOp loop, State &state, bool report) {
    if (!containsMatrixState(loop))
      return;
    if (Operation *stage = matchInductionLoop(loop)) {
      Key key =
          isa<AIE4StageOp>(stage)
              ? makeKey(cast<AIE4StageOp>(stage).getFormat(),
                        cast<AIE4StageOp>(stage).getRhsSignedAttr())
              : makeKey(cast<AIE4StageStreamOp>(stage).getFormat(),
                        cast<AIE4StageStreamOp>(stage).getRhsSignedAttr());
      state.slot.fill(Cell::defined(key, stage));
      return;
    }
    Region &body = loop.getRegion();
    if (!body.hasOneBlock()) {
      transferRegion(loop, body, state, report);
      return;
    }
    // Fixed point at the loop entry: the join of the first-iteration state
    // and every back-edge state. Each cell rises at most twice.
    State entry = state;
    State head = entry;
    for (unsigned iter = 0; iter < 2 * (kNumSlots + 1) + 1; ++iter) {
      State next = head;
      transfer(body.front(), next, /*report=*/false);
      next = join(entry, next);
      // Keep the latest notes (`why`) even when the kinds are stable.
      bool stable = next.sameAs(head);
      head = next;
      if (stable)
        break;
    }
    // Diagnose once at the fixed point; nested fixed-point visits stay silent.
    if (report) {
      State checked = head;
      transfer(body.front(), checked, /*report=*/true);
    }
    // After the loop: the join of the entry state (zero trips) and every
    // back-edge exit, which is the fixed point.
    state = head;
  }

  void transferIf(scf::IfOp ifOp, State &state, bool report) {
    if (!containsMatrixState(ifOp))
      return;
    State thenState = state;
    transferRegion(ifOp, ifOp.getThenRegion(), thenState, report);
    State elseState = state;
    transferRegion(ifOp, ifOp.getElseRegion(), elseState, report);
    state = join(thenState, elseState);
  }

  void transferOp(Operation *op, State &state, bool report) {
    if (auto stage = dyn_cast<AIE4StageOp>(op)) {
      writeSlot(state, stage.getSlot(),
                makeKey(stage.getFormat(), stage.getRhsSignedAttr()), op);
      return;
    }
    if (auto stage = dyn_cast<AIE4StageStreamOp>(op)) {
      writeSlot(state, stage.getSlot(),
                makeKey(stage.getFormat(), stage.getRhsSignedAttr()), op);
      return;
    }
    if (auto commit = dyn_cast<AIE4CommitOp>(op))
      return transferCommit(commit, state, report);
    if (auto mmac = dyn_cast<AIE4MmacActiveOp>(op))
      return transferMmac(mmac, state, report);
    if (auto region = dyn_cast<AIE4MatrixRegionOp>(op)) {
      // The region stages and commits its own tile; its body only reads it.
      state.setAll(Cell::defined(
          makeKey(region.getFormat(), region.getRhsSignedAttr()), op));
      return;
    }
    if (auto matmul = dyn_cast<AIE4MatMulOp>(op)) {
      Type elementType =
          cast<VectorType>(matmul.getLhs().getType()).getElementType();
      AIE4MatrixFormat format =
          elementType.isBF16()               ? AIE4MatrixFormat::Bf16
          : elementType.isSignlessInteger(8) ? AIE4MatrixFormat::Int8
          : elementType.isSignlessInteger(4) ? AIE4MatrixFormat::Int4
                                             : AIE4MatrixFormat::Fp8;
      state.setAll(
          Cell::defined(makeKey(format, matmul.getRhsSignedAttr()), op));
      return;
    }
    if (!isAIEVecOp(op)) {
      // Raw XLLVM inserts write only staging; raw commits only the matrix.
      // Raw MMACs read without a check.
      if (op->hasTrait<xllvm::AIE4StagingWriter>()) {
        state.slot.fill(Cell::undef(op));
        return;
      }
      if (op->hasTrait<xllvm::AIE4MatrixCommit>()) {
        state.matrix = Cell::undef(op);
        return;
      }
      if (op->hasTrait<xllvm::AIE4MatrixReader>())
        return;
    }
    if (isClobber(op)) {
      state.setAll(Cell::undef(op));
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(op))
      return transferFor(loop, state, report);
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      return transferIf(ifOp, state, report);
    if (op->getNumRegions() == 0 ||
        op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return;
    if (!containsMatrixState(op))
      return;
    reportUnsupported(op, report);
    state.setAll(Cell::undef(nullptr, /*diagnosed=*/report));
  }

  void transfer(Block &block, State &state, bool report) {
    for (Operation &op : block)
      transferOp(&op, state, report);
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct AIEVecAIE4CheckStagingPass
    : xilinx::aievec::impl::AIEVecAIE4CheckStagingBase<
          AIEVecAIE4CheckStagingPass> {
  using AIEVecAIE4CheckStagingBase::AIEVecAIE4CheckStagingBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    if (phase != "all" && phase != "coexistence" && phase != "provenance")
      return emitError(UnknownLoc::get(ctx))
             << "aievec-aie4-check-staging: unknown phase '" << phase
             << "'; expected 'coexistence', 'provenance' or 'all'";
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::MapVector<Operation *, Operation *> scopes =
        collectFlatScopesWithFirstOp(module);
    if (scopes.empty())
      return;
    bool failed = false;

    if (phase != "provenance") {
      module.walk([&](AIE4MatMulOp matmul) {
        Operation *scope = getAIE4ExecutionScope(matmul);
        auto it = scopes.find(scope);
        if (!scope || it == scopes.end())
          return;
        InFlightDiagnostic d =
            strict ? matmul.emitError() : matmul.emitWarning();
        d << "'aievec.aie4.matmul' must not share an execution scope with "
             "flat staging operations; a moved matmul could replace a primed "
             "or committed tile. Use 'aievec.aie4.matrix_region' here";
        d.attachNote(it->second->getLoc()) << "flat staging operation here";
        failed |= strict;
      });
    }

    if (phase != "coexistence") {
      ProvenanceChecker checker(strict);
      for (auto &entry : scopes)
        checker.checkScope(entry.first);
      failed |= checker.hadError();
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace
