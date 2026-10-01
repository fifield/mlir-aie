//===- AIECoreToStandard.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIECoreSymbols.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECORETOSTANDARD
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;

static StringRef getArchTargetTriple(AIEArch arch) {
  switch (arch) {
  case AIEArch::AIE1:
    return "aie";
  case AIEArch::AIE2:
    return "aie2";
  case AIEArch::AIE2p:
    return "aie2p";
  case AIEArch::AIE2ps:
    return "aie2ps";
  case AIEArch::AIE4:
    return "aie4";
  }
  llvm::report_fatal_error("unsupported arch");
}

// AIE2PS has no core intrinsic set (see declareAIEIntrinsics), so an op that
// lowers to an intrinsic call is diagnosed rather than left behind.
static LogicalResult checkCoreIntrinsics(Operation *op) {
  if (!isa<AIE2PSTargetModel>(op->getParentOfType<DeviceOp>().getTargetModel()))
    return success();
  return op->emitOpError("has no core lowering for AIE2ps");
}

using IntrinsicDecl =
    std::tuple<const char *, std::vector<Type>, std::vector<Type>>;
using IntrinsicDecls = std::vector<IntrinsicDecl>;

static auto getAIE1Intrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type int128Type = IntegerType::get(builder.getContext(), 128);
  Type int384Type = IntegerType::get(builder.getContext(), 384);
  Type floatType = Float32Type::get(builder.getContext());

  // Note that not all of these are valid for a particular design, or needed.
  // For right now, we will just accept the noise.
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie.event0", {}, {}},
      {"llvm.aie.event1", {}, {}},
      {"llvm.aie.put.ms",
       {int32Type, int32Type},
       {}}, //(%channel, %value) -> ()
      {"llvm.aie.put.wms",
       {int32Type, int128Type},
       {}}, //(%channel, %value) -> ()
      {"llvm.aie.put.fms",
       {int32Type, floatType},
       {}},                                          //(%channel, %value) -> ()
      {"llvm.aie.get.ss", {int32Type}, {int32Type}}, //(%channel, %value) -> ()
      {"llvm.aie.get.wss",
       {int32Type},
       {int128Type}},                                 //(%channel, %value) -> ()
      {"llvm.aie.get.fss", {int32Type}, {floatType}}, //(%channel, %value) -> ()
      {"llvm.aie.put.mcd", {int384Type}, {}},
      {"llvm.aie.get.scd", {}, {int384Type}},
      {"llvm.aie.lock.acquire.reg",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie.lock.release.reg",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
  };
  return functions;
}

static auto getAIE2Intrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type accType = VectorType::get({16}, int32Type);
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie2.event", {int32Type}, {}},
      {"llvm.aie2.put.ms", {int32Type, int32Type}, {}}, //(%value, %tlast) -> ()
      {"llvm.aie2.get.ss", {}, {int32Type, int32Type}}, //() -> (%value, %tlast)
      {"llvm.aie2.mcd.write.vec",
       {accType, int32Type},
       {}}, // (%value, %enable) -> ()
      {"llvm.aie2.scd.read.vec",
       {int32Type},
       {accType}}, // (%enable) -> (%value)
      {"llvm.aie2.acquire",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie2.release",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie2.set.ctrl.reg",
       {int32Type, int32Type},
       {}}, //(%reg_id, %value) -> ()
  };
  return functions;
}

static auto getAIE2pIntrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type accType = VectorType::get({16}, int32Type);
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie2p.event", {int32Type}, {}},
      {"llvm.aie2p.put.ms",
       {int32Type, int32Type},
       {}}, //(%value, %tlast) -> ()
      {"llvm.aie2p.get.ss",
       {},
       {int32Type, int32Type}}, //() -> (%value, %tlast)
      {"llvm.aie2p.mcd.write.vec",
       {accType, int32Type},
       {}}, // (%value, %enable) -> ()
      {"llvm.aie2p.scd.read.vec",
       {int32Type},
       {accType}}, // (%enable) -> (%value)
      {"llvm.aie2p.acquire",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie2p.release",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie2p.set.ctrl.reg",
       {int32Type, int32Type},
       {}}, //(%reg_id, %value) -> ()
  };
  return functions;
}

static auto getAIE4Intrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type accType = VectorType::get({16}, int32Type);
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie4.event", {int32Type}, {}}, //(%event_id) -> ()
      {"llvm.aie4.acquire",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie4.release",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie4.put.ms", {int32Type, int32Type}, {}}, //(%value, %tlast) -> ()
      {"llvm.aie4.put.ms.v8",
       {VectorType::get({8}, int32Type), int32Type},
       {}}, //(%value, %tlast) -> ()
      {"llvm.aie4.put.ms.v16",
       {VectorType::get({16}, int32Type), int32Type},
       {}}, //(%value, %tlast) -> ()
      // The operand selects input stream 0 or 1; TLAST is read separately.
      {"llvm.aie4.get.ss",
       {int32Type},
       {VectorType::get({32}, int32Type)}}, //(%stream) -> (%value)
      {"llvm.aie4.getl.wss",
       {int32Type},
       {VectorType::get({16}, int32Type)}}, //(%stream) -> (%value)
      {"llvm.aie4.mcd.write.vec",
       {accType, int32Type},
       {}}, // (%value, %enable) -> ()
      {"llvm.aie4.scd.read.vec",
       {int32Type},
       {accType}}, // (%enable) -> (%value)
      {"llvm.aie4.set.ctrl.reg",
       {int32Type, int32Type},
       {}}, //(%reg_id, %value) -> ()
  };
  return functions;
}

static void declareAIEIntrinsics(AIEArch arch, OpBuilder &builder) {
  auto registerIntrinsics = [&builder](const IntrinsicDecls &functions) {
    for (auto &i : functions) {
      auto [name, argTypes, retTypes] = i;
      func::FuncOp::create(
          builder, builder.getUnknownLoc(), name,
          FunctionType::get(builder.getContext(), argTypes, retTypes))
          .setPrivate();
    }
  };
  switch (arch) {
  case AIEArch::AIE1:
    registerIntrinsics(getAIE1Intrinsics(builder));
    return;
  case AIEArch::AIE2:
    registerIntrinsics(getAIE2Intrinsics(builder));
    return;
  case AIEArch::AIE2p:
    registerIntrinsics(getAIE2pIntrinsics(builder));
    return;
  case AIEArch::AIE4:
    registerIntrinsics(getAIE4Intrinsics(builder));
    return;
  case AIEArch::AIE2ps:
    // No intrinsic set: checkCoreIntrinsics rejects ops that need one.
    return;
  }
  llvm::report_fatal_error("unsupported arch");
}

// Move all the ops with OpTy inside device, to just before the device.
template <typename OpTy>
static void outlineOps(DeviceOp device) {
  SmallVector<OpTy, 16> ops;
  for (const auto &op : device.getOps<OpTy>())
    ops.push_back(op);

  for (const auto &op : ops)
    op->moveBefore(device);
}

namespace {
template <typename MyAIEOp>
struct AIEOpRemoval : OpConversionPattern<MyAIEOp> {
  using OpConversionPattern<MyAIEOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEOp::Adaptor;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEOp>(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(MyAIEOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEDebugOpToStdLowering : OpConversionPattern<DebugOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEDebugOpToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(DebugOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkCoreIntrinsics(op)))
      return failure();
    std::string funcName = "debug_i32";
    auto func = module.lookupSymbol<func::FuncOp>(funcName);
    if (!func)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 1> args;
    args.push_back(op.getArg());
    func::CallOp::create(rewriter, op.getLoc(), func, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEPutStreamToStdLowering : OpConversionPattern<PutStreamOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEPutStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(PutStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkCoreIntrinsics(op)))
      return failure();
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.put.";
    else if (targetModel.getTargetArch() == AIEArch::AIE2)
      funcName = "llvm.aie2.put.";
    else if (targetModel.getTargetArch() == AIEArch::AIE4)
      funcName = "llvm.aie4.put.";
    else
      funcName = "llvm.aie2p.put.";

    if (auto vectorType = dyn_cast<VectorType>(op.getStreamValue().getType()))
      funcName += vectorType.getNumElements() == 8 ? "ms.v8" : "ms.v16";
    else if (op.isWideStream())
      funcName += "wms";
    else if (op.isFloatStream())
      funcName += "fms";
    else
      funcName += "ms";

    auto putMSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!putMSFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE1) {
      args.push_back(op.getChannel());
      args.push_back(op.getStreamValue());
    } else {
      args.push_back(op.getStreamValue());
      args.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(0))); // tlast
    }
    func::CallOp::create(rewriter, op.getLoc(), putMSFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEGetStreamToStdLowering : OpConversionPattern<GetStreamOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEGetStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(GetStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkCoreIntrinsics(op)))
      return failure();
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    if (targetModel.getTargetArch() == AIEArch::AIE4)
      return lowerAIE4(op, rewriter);
    // The verifier restricts non-AIE4 results to scalar types.
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.get.";
    else if (targetModel.getTargetArch() == AIEArch::AIE2)
      funcName = "llvm.aie2.get.";
    else
      funcName = "llvm.aie2p.get.";

    if (op.isWideStream())
      funcName += "wss";
    else if (op.isFloatStream())
      funcName += "fss";
    else
      funcName += "ss";

    auto getSSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!getSSFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      args.push_back(op.getChannel());
    auto getSSCall =
        func::CallOp::create(rewriter, op.getLoc(), getSSFunc, args);
    rewriter.replaceOp(op, getSSCall.getResult(0));
    // Capture TLAST in AIEv2?
    return success();
  }

  // The verifier guarantees fixed rank-1 vectors of 16 or 32 i32 elements
  // and rejects constant stream indices outside 0 and 1.
  LogicalResult lowerAIE4(GetStreamOp op,
                          ConversionPatternRewriter &rewriter) const {
    auto vecTy = cast<VectorType>(op.getStreamValue().getType());

    StringRef funcName = vecTy.getNumElements() == 32 ? "llvm.aie4.get.ss"
                                                      : "llvm.aie4.getl.wss";
    auto getSSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!getSSFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;

    Location loc = op.getLoc();
    Type i32Ty = rewriter.getI32Type();
    Value stream = op.getChannel();
    if (!stream.getType().isSignlessInteger())
      return op.emitOpError("AIE4 stream index must be a signless integer");
    unsigned width = stream.getType().getIntOrFloatBitWidth();
    if (width < 32)
      stream = arith::ExtUIOp::create(rewriter, loc, i32Ty, stream);
    else if (width > 32)
      stream = arith::TruncIOp::create(rewriter, loc, i32Ty, stream);

    auto call =
        func::CallOp::create(rewriter, loc, getSSFunc, ValueRange{stream});
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct AIEPutCascadeToStdLowering : OpConversionPattern<PutCascadeOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEPutCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(PutCascadeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkCoreIntrinsics(op)))
      return failure();
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.put.mcd";
    else if (targetModel.getTargetArch() == AIEArch::AIE2)
      funcName = "llvm.aie2.mcd.write.vec";
    else if (targetModel.getTargetArch() == AIEArch::AIE4)
      funcName = "llvm.aie4.mcd.write.vec";
    else
      funcName = "llvm.aie2p.mcd.write.vec";
    auto putMCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!putMCDFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    Value cascadeValue = op.getCascadeValue();

    // Check if we need a bitcast for the input value
    Type expectedInputType = putMCDFunc.getFunctionType().getInput(0);
    Type actualInputType = cascadeValue.getType();

    if (expectedInputType != actualInputType) {
      // Create a bitcast operation to convert from actual input type to
      // expected type
      cascadeValue = vector::BitCastOp::create(rewriter, op.getLoc(),
                                               expectedInputType, cascadeValue);
    }

    args.push_back(cascadeValue);
    // Every generation after AIE1 takes an explicit enable operand. Keyed on
    // NOT being AIE1, rather than on isa<AIE2TargetModel>, which AIE4's target
    // model is not derived from -- that left the AIE4 call one operand short
    // and aborted with "incorrect number of operands for callee".
    if (!isa<AIE1TargetModel>(targetModel))
      args.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(1))); // enable

    func::CallOp::create(rewriter, op.getLoc(), putMCDFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEGetCascadeToStdLowering : OpConversionPattern<GetCascadeOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEGetCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(GetCascadeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkCoreIntrinsics(op)))
      return failure();
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.get.scd";
    else if (targetModel.getTargetArch() == AIEArch::AIE2)
      funcName = "llvm.aie2.scd.read.vec";
    else if (targetModel.getTargetArch() == AIEArch::AIE4)
      funcName = "llvm.aie4.scd.read.vec";
    else
      funcName = "llvm.aie2p.scd.read.vec";
    auto getSCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!getSCDFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    // Every generation after AIE1 takes an explicit enable operand. Keyed on
    // NOT being AIE1, rather than on isa<AIE2TargetModel>, which AIE4's target
    // model is not derived from -- that left the AIE4 call one operand short
    // and aborted with "incorrect number of operands for callee".
    if (!isa<AIE1TargetModel>(targetModel))
      args.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(1))); // enable

    auto getSCDCall =
        func::CallOp::create(rewriter, op.getLoc(), getSCDFunc, args);
    Value result = getSCDCall.getResult(0);

    // Check if we need a bitcast
    Type expectedType = op.getResult().getType();
    Type intrinsicReturnType = result.getType();

    if (expectedType != intrinsicReturnType) {
      // Create a bitcast operation to convert from intrinsic return type to
      // expected type
      result = vector::BitCastOp::create(rewriter, op.getLoc(), expectedType,
                                         result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct AIEUseLockToStdLowering : OpConversionPattern<UseLockOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEUseLockToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}
  LogicalResult
  matchAndRewrite(UseLockOp useLock, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<DeviceOp>(useLock->getParentOp())) {
      auto device = useLock->getParentOfType<DeviceOp>();
      if (!device) {
        return module.emitOpError("Device Not found!");
      }
      const auto &targetModel = device.getTargetModel();
      // A lock in a core or in any function needs an intrinsic, whether or
      // not an exported core calls the function. Any other lock (DMA
      // regions, runtime sequences) is erased with its parent.
      if (isa<AIE2PSTargetModel>(targetModel)) {
        if (useLock->getParentOfType<CoreOp>() ||
            useLock->getParentOfType<func::FuncOp>())
          return checkCoreIntrinsics(useLock);
        rewriter.eraseOp(useLock);
        return success();
      }

      // Generate the intrinsic name
      std::string funcName;
      if (targetModel.getTargetArch() == AIEArch::AIE1)
        funcName = "llvm.aie.lock.";
      else if (targetModel.getTargetArch() == AIEArch::AIE2)
        funcName = "llvm.aie2.";
      else if (targetModel.getTargetArch() == AIEArch::AIE2p)
        funcName = "llvm.aie2p.";
      else
        funcName = "llvm.aie4.";

      if (useLock.acquire() || useLock.acquireGE())
        funcName += "acquire";
      else if (useLock.release())
        funcName += "release";
      if (targetModel.getTargetArch() == AIEArch::AIE1)
        funcName += ".reg";

      auto useLockFunc = module.lookupSymbol<func::FuncOp>(funcName);
      if (!useLockFunc)
        return useLock.emitOpError("Could not find the intrinsic function!");

      SmallVector<Value, 2> args;
      auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
      args.push_back(arith::IndexCastOp::create(rewriter, useLock.getLoc(),
                                                i32Ty, useLock.getLock()));

      Value value = adaptor.getValue();
      // AIE2 acquire-greater-equal is encoded as a negative value, so negate
      // it at runtime.
      if (useLock.acquireGE()) {
        Value zero = arith::ConstantOp::create(
            rewriter, useLock.getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
        value = arith::SubIOp::create(rewriter, useLock.getLoc(), zero, value);
      }
      args.push_back(value);

      func::CallOp::create(rewriter, useLock.getLoc(), useLockFunc, args);
    }
    rewriter.eraseOp(useLock);
    return success();
  }
};

struct AIEBufferToStandard : OpConversionPattern<BufferOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  AIEBufferToStandard(MLIRContext *context, ModuleOp &m,
                      PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}
  LogicalResult
  matchAndRewrite(BufferOp buffer, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The core's own sections fill this extent; nothing refers to it by name.
    if (buffer.getCoreData()) {
      rewriter.eraseOp(buffer);
      return success();
    }
    rewriter.setInsertionPointToStart(module.getBody());
    auto t = llvm::cast<MemRefType>(buffer.getType());
    auto symName = buffer.name().getValue();
    // A declaration, never a definition: a buffer lives at the address the
    // allocator chose, and the linker supplies that address from the generated
    // script or BCF. `initial_value` is device state, written by whoever
    // configures the device -- the CDO writer, or mlir_aie_initialize_buffers
    // on the simulator path. A definition here would place the bytes wherever
    // `*(.data*)` lands, which costs the core's data region a copy it never
    // reads.
    memref::GlobalOp::create(rewriter, buffer.getLoc(), symName,
                             rewriter.getStringAttr("public"), buffer.getType(),
                             /*initial_value=*/nullptr, /*constant*/ false,
                             /*alignment*/ nullptr);

    for (auto &use : make_early_inc_range(buffer.getResult().getUses())) {
      Operation *user = use.getOwner();
      rewriter.setInsertionPoint(user);
      auto allocated =
          memref::GetGlobalOp::create(rewriter, buffer.getLoc(), t, symName);
      // Assume that buffers are aligned so they can be vectorized.
      memref::AssumeAlignmentOp::create(rewriter, buffer.getLoc(), allocated,
                                        32);

      use.set(allocated.getResult());
    }

    rewriter.eraseOp(buffer);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4 core-entry control writes (decision D19)
//===----------------------------------------------------------------------===//
//
// Control-register ids for llvm.aie4.set.ctrl.reg, from Peano's
// aie4/aie4_defines.h; reset values and encodings from ISA 2.10 (pp. 39-44).
constexpr int32_t kAIE4CrVecRnd = 24;     // VMAC-unit rounding, reset 0 (floor)
constexpr int32_t kAIE4CrRnd = 46;        // SRS/conversion rounding, reset 0
constexpr int32_t kAIE4CrF2BConvSat = 53; // MX conversion saturation, reset 3
constexpr int32_t kAIE4RndConvEven = 12;  // convergent rounding to even (RNE)
constexpr int32_t kAIE4RndSymFloor = 2;   // symmetric floor: toward zero
constexpr int32_t kAIE4F2BConvSatMax = 3; // Inf and Huge -> max representable

struct AIE4CoreEntryModes {
  // Write crRnd = rndMode, unless the core reads the ambient crRnd through an
  // explicit op (see scanAIE4CoreEntryModes).
  bool rnd = true;
  int32_t rndMode = kAIE4RndConvEven;
  // crF2BConvSat = 3: the core has an MX conversion. No op in the tree
  // reads it yet; aie4-r2-mx-backend adds the readers.
  bool f2bConvSat = false;
};

// Does `op` round a result to bf16 or f16 through a crRnd reader? Peano
// computes scalar and vector bf16/f16 arithmetic, int -> bf16/f16
// conversions and f32 -> bf16/f16 truncations in f32 or integer, then
// narrows with VCONV, which rounds with crRnd.
static bool isGenericAIE4RndReader(Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  Type elemTy = getElementTypeOrSelf(op->getResult(0).getType());
  if (!elemTy.isBF16() && !elemTy.isF16())
    return false;
  if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::RemFOp, arith::SIToFPOp, arith::UIToFPOp, arith::TruncFOp>(op))
    return true;
  return isa<math::MathDialect>(op->getDialect());
}

// Walk `root` and every function body it calls, transitively. aiecc inlines
// device-level helpers into the core, so their readers count as the core's.
static void walkAIE4CoreAndCallees(Operation *root,
                                   function_ref<WalkResult(Operation *)> fn) {
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 8> visited{root};
  while (!worklist.empty()) {
    Operation *region = worklist.pop_back_val();
    WalkResult r = region->walk([&](Operation *op) -> WalkResult {
      if (auto call = dyn_cast<func::CallOp>(op)) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (callee && !callee.isExternal() && visited.insert(callee).second)
          worklist.push_back(callee);
      }
      return fn(op);
    });
    if (r.wasInterrupted())
      return;
  }
}

// D19's per-core modes for an AIE4 core (crVecRnd = 12 always).
//
// crRnd = 12 by default: every generic bf16/f16 rounding op (arith
// arithmetic, int -> float, f32 truncation, math) assumes round to nearest
// even, and the loader leaves crRnd at its reset value, 0 (floor). The
// explicit aievec.aie4.srs_i32_i16 and aievec.aie4.conv_f32_bf16 ops are
// defined to read the ambient crRnd (release-1 references assume floor), so a
// core containing one keeps the ambient mode.
//
// aievec.aie4.conv_f32_i32 implements arith.fptosi/fptoui, which truncate
// toward zero, so a core containing it gets crRnd = 2. The mode cannot be
// switched around each conversion instead: when the result feeds a store,
// Peano fuses the two into vst.conv and drops the adjacent set.ctrl.reg
// calls (FaaSApps/llvm-aie#3175).
//
// A core with two of these requirements is rejected: no single crRnd serves
// both.
static FailureOr<AIE4CoreEntryModes> scanAIE4CoreEntryModes(CoreOp core) {
  AIE4CoreEntryModes modes;
  Operation *genericRnd = nullptr;
  Operation *ambientRnd = nullptr;
  Operation *truncRnd = nullptr;
  bool failedScan = false;
  walkAIE4CoreAndCallees(core, [&](Operation *op) -> WalkResult {
    if (auto truncOp = dyn_cast<arith::TruncFOp>(op)) {
      if (auto mode = truncOp.getRoundingmode();
          mode && *mode != arith::RoundingMode::to_nearest_even &&
          isGenericAIE4RndReader(op)) {
        truncOp.emitOpError(
            "requests a rounding mode other than to_nearest_even; AIE4 "
            "cores fix conversion rounding (crRnd) to round-to-nearest-even "
            "at core entry");
        failedScan = true;
        return WalkResult::interrupt();
      }
    }
    if (!genericRnd && isGenericAIE4RndReader(op))
      genericRnd = op;
    if (!ambientRnd &&
        isa<aievec::AIE4SRSI32I16Op, aievec::AIE4ConvF32BF16Op>(op))
      ambientRnd = op;
    if (!truncRnd && isa<aievec::AIE4ConvF32I32Op>(op))
      truncRnd = op;
    return WalkResult::advance();
  });
  if (failedScan)
    return failure();
  if (genericRnd && ambientRnd) {
    auto diag = core.emitOpError(
        "mixes ops that need round-to-nearest-even (crRnd = 12) with an "
        "explicit op that reads the ambient crRnd; setting crRnd for one "
        "would change the other's numerics");
    diag.attachNote(genericRnd->getLoc()) << "needs crRnd = 12";
    diag.attachNote(ambientRnd->getLoc()) << "reads the ambient crRnd";
    diag.attachNote()
        << "split them into separate cores, or use "
           "aievec.aie4.conv_f32_bf16 if floor rounding is acceptable; "
           "explicit ops will get a rounding attribute (bead "
           "aie4-r2-explicit-rnd)";
    return failure();
  }
  if (truncRnd && (genericRnd || ambientRnd)) {
    Operation *other = genericRnd ? genericRnd : ambientRnd;
    auto diag =
        core.emitOpError(
            "mixes a float -> int conversion, which needs crRnd = 2 (toward "
            "zero), with an op that ")
        << (genericRnd ? "needs round-to-nearest-even (crRnd = 12)"
                       : "reads the ambient crRnd")
        << "; setting crRnd for one would change the other's numerics";
    diag.attachNote(truncRnd->getLoc()) << "needs crRnd = 2";
    diag.attachNote(other->getLoc())
        << (genericRnd ? "needs crRnd = 12" : "reads the ambient crRnd");
    diag.attachNote() << "split them into separate cores";
    return failure();
  }
  modes.rnd = ambientRnd == nullptr;
  if (truncRnd)
    modes.rndMode = kAIE4RndSymFloor;
  return modes;
}

// Write the modes at the start of the core function, as the AIE2/AIE2P setup
// does. Readers of these registers carry implicit uses of them in Peano's
// MIR, so the writes stay ahead of every reader that needs them.
static void emitAIE4CoreEntryModes(func::FuncOp coreFunc,
                                   const AIE4CoreEntryModes &modes,
                                   ModuleOp module,
                                   ConversionPatternRewriter &rewriter) {
  auto setCtrlReg = module.lookupSymbol<func::FuncOp>("llvm.aie4.set.ctrl.reg");
  assert(setCtrlReg && "AIE4 intrinsics are declared before outlining");
  Location loc = coreFunc.getLoc();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&coreFunc.getBody().front());
  auto write = [&](int32_t reg, int32_t value) {
    Value r = arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getI32IntegerAttr(reg));
    Value v = arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getI32IntegerAttr(value));
    func::CallOp::create(rewriter, loc, setCtrlReg, ValueRange{r, v});
  };
  write(kAIE4CrVecRnd, kAIE4RndConvEven);
  if (modes.rnd)
    write(kAIE4CrRnd, modes.rndMode);
  if (modes.f2bConvSat)
    write(kAIE4CrF2BConvSat, kAIE4F2BConvSatMax);
}

// With crRnd = 12 guaranteed, an explicit to_nearest_even on arith.truncf is
// what the hardware does anyway. Drop it in the core and its callees: the
// attributed form lowers to llvm.experimental.constrained.fptrunc, which
// Peano cannot select.
static void dropAIE4RNETruncAttrs(CoreOp core,
                                  ConversionPatternRewriter &rewriter) {
  walkAIE4CoreAndCallees(core, [&](Operation *op) -> WalkResult {
    if (auto truncOp = dyn_cast<arith::TruncFOp>(op))
      if (truncOp.getRoundingmode() == arith::RoundingMode::to_nearest_even)
        rewriter.modifyOpInPlace(truncOp, [&] {
          truncOp->removeAttr(truncOp.getRoundingmodeAttrName());
        });
    return WalkResult::advance();
  });
}

struct AIECoreToStandardFunc : OpConversionPattern<CoreOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  IRMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers;
  int tileCol = 0;
  int tileRow = 0;

  AIECoreToStandardFunc(
      MLIRContext *context, ModuleOp &m, IRMapping &mapper,
      DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers,
      PatternBenefit benefit = 1, int tileCol = 1, int tileRow = 1)
      : OpConversionPattern(context, benefit), module(m), mapper(mapper),
        tileToBuffers(tileToBuffers), tileCol(tileCol), tileRow(tileRow) {}

  LogicalResult
  matchAndRewrite(CoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    int col = op.colIndex();
    int row = op.rowIndex();

    // Only pull code for the indicated function
    if ((tileRow != row && tileRow != -1) ||
        (tileCol != col && tileCol != -1)) {
      rewriter.eraseOp(op);
      return success();
    }

    // AIE4 cores get D19's control writes at entry; check first that the
    // core's crRnd requirements are consistent.
    auto coreDevice = op->getParentOfType<DeviceOp>();
    bool isAIE4 = coreDevice &&
                  coreDevice.getTargetModel().getTargetArch() == AIEArch::AIE4;
    AIE4CoreEntryModes aie4Modes;
    if (isAIE4) {
      FailureOr<AIE4CoreEntryModes> scanned = scanAIE4CoreEntryModes(op);
      if (failed(scanned))
        return failure();
      aie4Modes = *scanned;
      if (aie4Modes.rnd && aie4Modes.rndMode == kAIE4RndConvEven)
        dropAIE4RNETruncAttrs(op, rewriter);
    }

    // The parent should be an AIE.device op.
    rewriter.setInsertionPointAfter(op->getParentOp());

    std::string coreName = coreFrameSymbolName(col, row);
    auto coreFunc =
        func::FuncOp::create(rewriter, op.getLoc(), coreName,
                             FunctionType::get(rewriter.getContext(), {}, {}));

    rewriter.cloneRegionBefore(op.getBody(), coreFunc.getBody(),
                               coreFunc.getBody().begin(), mapper);

    // Set saturation and rounding modes at core entry for AIE2/AIE2p, but
    // only if the core body contains aievec.srs or bf16 aievec.matmul ops.
    // Skip for cores with only lock/stream ops to avoid breaking existing
    // test SSA naming.
    bool hasSRS = false;
    bool hasIntegerSRS = false;
    bool hasBF16Matmul = false;
    auto scan = [&](Operation *childOp) {
      StringRef opName = childOp->getName().getStringRef();
      if (opName == "aievec.srs") {
        hasSRS = true;
        // Check if this is an integer SRS (e.g., i32→i8) vs float SRS
        // (e.g., f32→bf16). Integer SRS needs positive_inf rounding to
        // match C++ kernel behavior; float SRS works better with floor.
        if (childOp->getNumResults() > 0) {
          auto resultType = childOp->getResult(0).getType();
          if (auto vecType = dyn_cast<VectorType>(resultType)) {
            if (vecType.getElementType().isInteger())
              hasIntegerSRS = true;
          }
        }
      }
      // Detect bf16 matmul ops — these need conv_even rounding to avoid
      // systematic negative bias from floor rounding in BFP16 arithmetic.
      // On AIE2P a bf16 matmul is a bfp16 one: aievec.to_bfp16 converts its
      // bf16 operands, and aievec.matmul_aie2p_bfp16 may take them loaded.
      if (opName == "aievec.to_bfp16" || opName == "aievec.matmul_aie2p_bfp16")
        hasBF16Matmul = true;
      if (opName == "aievec.matmul" || opName == "aievec.matmul_aie2p") {
        if (childOp->getNumOperands() > 0) {
          auto lhsType = childOp->getOperand(0).getType();
          if (auto vecType = dyn_cast<VectorType>(lhsType))
            if (vecType.getElementType().isBF16())
              hasBF16Matmul = true;
        }
      }
    };
    coreFunc.walk(scan);
    auto device = op->getParentOfType<DeviceOp>();
    // AIE2ps cannot set the modes anywhere, so the functions the core calls
    // must be free of these ops too.
    if (device && isa<AIE2PSTargetModel>(device.getTargetModel())) {
      SmallVector<func::FuncOp> worklist;
      DenseSet<Operation *> visited;
      auto addCallees = [&](Operation *root) {
        root->walk([&](func::CallOp call) {
          auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              device, call.getCalleeAttr());
          if (callee && visited.insert(callee).second)
            worklist.push_back(callee);
        });
      };
      addCallees(coreFunc);
      while (!worklist.empty()) {
        func::FuncOp callee = worklist.pop_back_val();
        callee.walk(scan);
        addCallees(callee);
      }
    }
    if (hasSRS || hasBF16Matmul) {
      if (device) {
        if (isa<AIE2PSTargetModel>(device.getTargetModel()))
          return op.emitOpError("needs the rounding and saturation modes set "
                                "for aievec.srs or bf16 matmul, which has no "
                                "lowering for AIE2ps");
        AIEArch arch = device.getTargetModel().getTargetArch();
        if (arch == AIEArch::AIE2 || arch == AIEArch::AIE2p) {
          std::string ctrlRegFuncName = (arch == AIEArch::AIE2p)
                                            ? "llvm.aie2p.set.ctrl.reg"
                                            : "llvm.aie2.set.ctrl.reg";
          auto ctrlRegFunc = module.lookupSymbol<func::FuncOp>(ctrlRegFuncName);
          if (ctrlRegFunc) {
            Block &entryBlock = coreFunc.getBody().front();
            rewriter.setInsertionPointToStart(&entryBlock);
            Location loc = op.getLoc();
            // Rounding register index differs between AIE2 and AIE2P:
            //   AIE2:  crRnd=6
            //   AIE2P: crRnd=1
            // Saturation register uses AIE2 index (9) for both architectures.
            // On AIE2P, index 9 maps to crPackSize (no-op for saturation),
            // preserving the pre-existing behavior. The AIE2P crSat fix
            // (index 0) requires updating downstream tests and is tracked
            // separately.
            int satRegIdx = 9;
            int rndRegIdx = (arch == AIEArch::AIE2p) ? 1 : 6;
            // saturation_mode::saturate = 1
            auto cSatIdx = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(satRegIdx));
            auto c1 = arith::ConstantOp::create(rewriter, loc,
                                                rewriter.getI32IntegerAttr(1));
            func::CallOp::create(rewriter, loc, ctrlRegFunc,
                                 ValueRange{cSatIdx, c1});
            // Rounding mode:
            //  - conv_even (12) for bf16 matmul: eliminates systematic
            //    negative bias from floor rounding in BFP16 arithmetic,
            //    matching ::aie::set_rounding(aie::rounding_mode::conv_even)
            //    used in external C++ matmul kernels.
            //  - positive_inf (9) for integer SRS (shift-round-saturate on
            //    integer data, e.g., i32→i8), unless the core has a bf16
            //    matmul.
            //  - conv_even (12) for float-only SRS (f32→bf16), matching the
            //    round-to-nearest-even semantics of arith.truncf.
            int roundingMode = hasIntegerSRS && !hasBF16Matmul ? 9 : 12;
            auto cRndIdx = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(rndRegIdx));
            auto cRoundingMode = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(roundingMode));
            func::CallOp::create(rewriter, loc, ctrlRegFunc,
                                 ValueRange{cRndIdx, cRoundingMode});
          }
        }
      }
    }

    if (isAIE4)
      emitAIE4CoreEntryModes(coreFunc, aie4Modes, module, rewriter);

    // Rewrite the AIE.end() op
    coreFunc.getBody().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (isa<EndOp>(childOp)) {
        func::ReturnOp::create(rewriter, childOp->getLoc(), ValueRange({}));
        rewriter.eraseOp(childOp);
      }
    });

    rewriter.eraseOp(op);
    return success();
  }
};

// Lower AIE.event to llvm.aie.event intrinsic
struct AIEEventOpToStdLowering : OpConversionPattern<EventOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEEventOpToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(EventOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    std::string funcName;
    SmallVector<Value, 1> args;
    switch (device.getTargetModel().getTargetArch()) {
    case AIEArch::AIE1:
      funcName = "llvm.aie.event" + std::to_string(op.getVal());
      break;
    case AIEArch::AIE2:
      funcName = "llvm.aie2.event";
      args.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(op.getVal())));
      break;
    case AIEArch::AIE2p:
      funcName = "llvm.aie2p.event";
      args.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(op.getVal())));
      break;
    case AIEArch::AIE2ps:
      return op.emitOpError("has no core lowering for AIE2ps");
    case AIEArch::AIE4:
      funcName = "llvm.aie4.event";
      args.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(op.getVal())));
      break;
    }
    auto eventFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!eventFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    func::CallOp::create(rewriter, op.getLoc(), eventFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIECoreToStandardPass
    : xilinx::AIE::impl::AIECoreToStandardBase<AIECoreToStandardPass> {
  AIECoreToStandardPass() = default;
  AIECoreToStandardPass(const AIECoreToStandardOptions &options) {
    deviceName = options.deviceName;
    tileCol = options.tileCol;
    tileRow = options.tileRow;
  }

  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    DeviceOp deviceOp = DeviceOp::getForSymbolInModuleOrError(m, deviceName);
    if (!deviceOp) {
      return signalPassFailure();
    }
    const auto &targetModel = deviceOp.getTargetModel();

    // Each core body is later outlined into a module-level function, so it must
    // be self-contained. CSE/canonicalization can hoist constant operands --
    // e.g. the lock values that are now SSA operands -- out of the core into
    // the enclosing device region, where they are shared with the mem/shim DMA
    // blocks. Such an external reference would dangle once the device (and the
    // hoisted constant) is erased. Sink a private copy of every
    // externally-defined constant used inside a core back into that core.
    deviceOp.walk([&](CoreOp core) {
      Block &entry = core.getBody().front();
      DenseMap<Value, Value> localCopies;
      SmallVector<std::pair<OpOperand *, Value>> toRewrite;
      core.getBody().walk([&](Operation *op) {
        for (OpOperand &operand : op->getOpOperands()) {
          Value v = operand.get();
          Operation *def = v.getDefiningOp();
          if (!def || core->isProperAncestor(def))
            continue; // block argument or already defined inside the core
          if (!def->hasTrait<mlir::OpTrait::ConstantLike>())
            continue;
          Value &copy = localCopies[v];
          if (!copy) {
            OpBuilder b(&entry, entry.begin());
            copy = b.clone(*def)->getResult(0);
          }
          toRewrite.emplace_back(&operand, copy);
        }
      });
      for (auto &[operand, copy] : toRewrite)
        operand->set(copy);
    });

    // Copy data layout attribute from DeviceOp to ModuleOp if present
    if (auto dlAttr = deviceOp->getAttr(DLTIDialect::kDataLayoutAttrName)) {
      m->setAttr(DLTIDialect::kDataLayoutAttrName, dlAttr);
    }

    // Ensure that we don't have an incorrect target triple.  This may override
    // some bogus target triple in the original mlir.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               builder.getStringAttr(
                   getArchTargetTriple(targetModel.getTargetArch())));

    DenseMap<Operation *, SmallVector<BufferOp, 4>> tileToBuffers;

    // Populate intrinsic functions
    // Intrinsic information:
    // peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td Also take a look
    // at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());
    declareAIEIntrinsics(targetModel.getTargetArch(), builder);

    IRMapping mapper;
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<VectorDialect>();
    target.addLegalDialect<aievec::AIEVecDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<ub::UBDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<index::IndexDialect>();
    target.addLegalDialect<ptr::PtrDialect>();
    // A core body may already hold LLVM ops: aie.dma_push_queue lowers to the
    // stores that push a task before the body is outlined.
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<func::FuncOp, ModuleOp, UnrealizedConversionCastOp>();
    // Ops this pass lowers must be illegal: a failed rewrite of an op with
    // unknown legality is silently left in place.
    // A core that is not exported is erased, so its ops are left alone.
    target.addIllegalOp<BufferOp>();
    target.addDynamicallyLegalOp<PutStreamOp, GetStreamOp, PutCascadeOp,
                                 GetCascadeOp, DebugOp, UseLockOp, EventOp>(
        [col = static_cast<int>(tileCol),
         row = static_cast<int>(tileRow)](Operation *op) {
          auto core = op->getParentOfType<CoreOp>();
          return core && ((row != core.rowIndex() && row != -1) ||
                          (col != core.colIndex() && col != -1));
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<AIEPutStreamToStdLowering, AIEGetStreamToStdLowering,
                 AIEPutCascadeToStdLowering, AIEGetCascadeToStdLowering,
                 AIEDebugOpToStdLowering, AIEUseLockToStdLowering,
                 AIEEventOpToStdLowering>(m.getContext(), m);

    patterns.add<AIEBufferToStandard>(m.getContext(), m, /*benefit*/ 1);
    if (failed(applyPartialConversion(deviceOp, target, std::move(patterns))))
      return signalPassFailure();

    // Core-entry validation can reject a core during outlining. Make that
    // failure fail the pass instead of leaving the core for device removal.
    ConversionTarget outlineTarget(target);
    outlineTarget.addIllegalOp<CoreOp>();

    RewritePatternSet outlinePatterns(&getContext());
    outlinePatterns.add<AIECoreToStandardFunc>(m.getContext(), m, mapper,
                                               tileToBuffers, /*benefit*/ 1,
                                               tileCol, tileRow);
    if (failed(applyPartialConversion(deviceOp, outlineTarget,
                                      std::move(outlinePatterns))))
      return signalPassFailure();

    // Move all the func.func ops and memref.globals from the device to the
    // module
    outlineOps<memref::GlobalOp>(deviceOp);
    outlineOps<func::FuncOp>(deviceOp);

    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<
        AIEOpRemoval<DeviceOp>, AIEOpRemoval<TileOp>, AIEOpRemoval<FlowOp>,
        AIEOpRemoval<MemOp>, AIEOpRemoval<ShimDMAOp>, AIEOpRemoval<ShimMuxOp>,
        AIEOpRemoval<SwitchboxOp>, AIEOpRemoval<LockOp>, AIEOpRemoval<BufferOp>,
        AIEOpRemoval<ExternalBufferOp>, AIEOpRemoval<ShimDMAAllocationOp>,
        AIEOpRemoval<CascadeFlowOp>, AIEOpRemoval<ConfigureCascadeOp>>(
        m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> AIE::createAIECoreToStandardPass() {
  return std::make_unique<AIECoreToStandardPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
AIE::createAIECoreToStandardPass(const AIECoreToStandardOptions &options) {
  return std::make_unique<AIECoreToStandardPass>(options);
}
