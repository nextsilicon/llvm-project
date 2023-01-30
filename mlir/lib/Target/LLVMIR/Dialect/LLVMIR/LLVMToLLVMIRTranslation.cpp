//===- LLVMToLLVMIRTranslation.cpp - Translate LLVM dialect to LLVM IR ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Operator.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::getLLVMConstant;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc"

/// Convert MLIR integer comparison predicate to LLVM IR comparison predicate.
static llvm::CmpInst::Predicate getLLVMCmpPredicate(ICmpPredicate p) {
  switch (p) {
  case LLVM::ICmpPredicate::eq:
    return llvm::CmpInst::Predicate::ICMP_EQ;
  case LLVM::ICmpPredicate::ne:
    return llvm::CmpInst::Predicate::ICMP_NE;
  case LLVM::ICmpPredicate::slt:
    return llvm::CmpInst::Predicate::ICMP_SLT;
  case LLVM::ICmpPredicate::sle:
    return llvm::CmpInst::Predicate::ICMP_SLE;
  case LLVM::ICmpPredicate::sgt:
    return llvm::CmpInst::Predicate::ICMP_SGT;
  case LLVM::ICmpPredicate::sge:
    return llvm::CmpInst::Predicate::ICMP_SGE;
  case LLVM::ICmpPredicate::ult:
    return llvm::CmpInst::Predicate::ICMP_ULT;
  case LLVM::ICmpPredicate::ule:
    return llvm::CmpInst::Predicate::ICMP_ULE;
  case LLVM::ICmpPredicate::ugt:
    return llvm::CmpInst::Predicate::ICMP_UGT;
  case LLVM::ICmpPredicate::uge:
    return llvm::CmpInst::Predicate::ICMP_UGE;
  }
  llvm_unreachable("incorrect comparison predicate");
}

static llvm::CmpInst::Predicate getLLVMCmpPredicate(FCmpPredicate p) {
  switch (p) {
  case LLVM::FCmpPredicate::_false:
    return llvm::CmpInst::Predicate::FCMP_FALSE;
  case LLVM::FCmpPredicate::oeq:
    return llvm::CmpInst::Predicate::FCMP_OEQ;
  case LLVM::FCmpPredicate::ogt:
    return llvm::CmpInst::Predicate::FCMP_OGT;
  case LLVM::FCmpPredicate::oge:
    return llvm::CmpInst::Predicate::FCMP_OGE;
  case LLVM::FCmpPredicate::olt:
    return llvm::CmpInst::Predicate::FCMP_OLT;
  case LLVM::FCmpPredicate::ole:
    return llvm::CmpInst::Predicate::FCMP_OLE;
  case LLVM::FCmpPredicate::one:
    return llvm::CmpInst::Predicate::FCMP_ONE;
  case LLVM::FCmpPredicate::ord:
    return llvm::CmpInst::Predicate::FCMP_ORD;
  case LLVM::FCmpPredicate::ueq:
    return llvm::CmpInst::Predicate::FCMP_UEQ;
  case LLVM::FCmpPredicate::ugt:
    return llvm::CmpInst::Predicate::FCMP_UGT;
  case LLVM::FCmpPredicate::uge:
    return llvm::CmpInst::Predicate::FCMP_UGE;
  case LLVM::FCmpPredicate::ult:
    return llvm::CmpInst::Predicate::FCMP_ULT;
  case LLVM::FCmpPredicate::ule:
    return llvm::CmpInst::Predicate::FCMP_ULE;
  case LLVM::FCmpPredicate::une:
    return llvm::CmpInst::Predicate::FCMP_UNE;
  case LLVM::FCmpPredicate::uno:
    return llvm::CmpInst::Predicate::FCMP_UNO;
  case LLVM::FCmpPredicate::_true:
    return llvm::CmpInst::Predicate::FCMP_TRUE;
  }
  llvm_unreachable("incorrect comparison predicate");
}

static llvm::AtomicRMWInst::BinOp getLLVMAtomicBinOp(AtomicBinOp op) {
  switch (op) {
  case LLVM::AtomicBinOp::xchg:
    return llvm::AtomicRMWInst::BinOp::Xchg;
  case LLVM::AtomicBinOp::add:
    return llvm::AtomicRMWInst::BinOp::Add;
  case LLVM::AtomicBinOp::sub:
    return llvm::AtomicRMWInst::BinOp::Sub;
  case LLVM::AtomicBinOp::_and:
    return llvm::AtomicRMWInst::BinOp::And;
  case LLVM::AtomicBinOp::nand:
    return llvm::AtomicRMWInst::BinOp::Nand;
  case LLVM::AtomicBinOp::_or:
    return llvm::AtomicRMWInst::BinOp::Or;
  case LLVM::AtomicBinOp::_xor:
    return llvm::AtomicRMWInst::BinOp::Xor;
  case LLVM::AtomicBinOp::max:
    return llvm::AtomicRMWInst::BinOp::Max;
  case LLVM::AtomicBinOp::min:
    return llvm::AtomicRMWInst::BinOp::Min;
  case LLVM::AtomicBinOp::umax:
    return llvm::AtomicRMWInst::BinOp::UMax;
  case LLVM::AtomicBinOp::umin:
    return llvm::AtomicRMWInst::BinOp::UMin;
  case LLVM::AtomicBinOp::fadd:
    return llvm::AtomicRMWInst::BinOp::FAdd;
  case LLVM::AtomicBinOp::fsub:
    return llvm::AtomicRMWInst::BinOp::FSub;
  }
  llvm_unreachable("incorrect atomic binary operator");
}

static llvm::AtomicOrdering getLLVMAtomicOrdering(AtomicOrdering ordering) {
  switch (ordering) {
  case LLVM::AtomicOrdering::not_atomic:
    return llvm::AtomicOrdering::NotAtomic;
  case LLVM::AtomicOrdering::unordered:
    return llvm::AtomicOrdering::Unordered;
  case LLVM::AtomicOrdering::monotonic:
    return llvm::AtomicOrdering::Monotonic;
  case LLVM::AtomicOrdering::acquire:
    return llvm::AtomicOrdering::Acquire;
  case LLVM::AtomicOrdering::release:
    return llvm::AtomicOrdering::Release;
  case LLVM::AtomicOrdering::acq_rel:
    return llvm::AtomicOrdering::AcquireRelease;
  case LLVM::AtomicOrdering::seq_cst:
    return llvm::AtomicOrdering::SequentiallyConsistent;
  }
  llvm_unreachable("incorrect atomic ordering");
}

static llvm::FastMathFlags getFastmathFlags(FastmathFlagsInterface &op) {
  using llvmFMF = llvm::FastMathFlags;
  using FuncT = void (llvmFMF::*)(bool);
  const std::pair<FastmathFlags, FuncT> handlers[] = {
      // clang-format off
      {FastmathFlags::nnan,     &llvmFMF::setNoNaNs},
      {FastmathFlags::ninf,     &llvmFMF::setNoInfs},
      {FastmathFlags::nsz,      &llvmFMF::setNoSignedZeros},
      {FastmathFlags::arcp,     &llvmFMF::setAllowReciprocal},
      {FastmathFlags::contract, &llvmFMF::setAllowContract},
      {FastmathFlags::afn,      &llvmFMF::setApproxFunc},
      {FastmathFlags::reassoc,  &llvmFMF::setAllowReassoc},
      // clang-format on
  };
  llvm::FastMathFlags ret;
  ::mlir::LLVM::FastmathFlags fmfMlir = op.getFastmathAttr().getValue();
  for (auto it : handlers)
    if (bitEnumContainsAll(fmfMlir, it.first))
      (ret.*(it.second))(true);
  return ret;
}

namespace {
/// A helper class that converts a LoopAnnotationAttr into a corresponding
/// llvm::MDNode.
struct LoopAnnotationConverter {
  LoopAnnotationConverter(LoopAnnotationAttr attr, llvm::LLVMContext &ctx,
                          LLVM::ModuleTranslation &moduleTranslation,
                          Operation &opInst)
      : attr(attr), ctx(ctx), moduleTranslation(moduleTranslation),
        opInst(opInst) {}
  llvm::MDNode *convert();

private:
  void createAndAddUnitNode(StringRef name);
  void createAndAddBoolNode(StringRef name, bool val);
  void createAndAddI32Node(StringRef name, uint32_t val);
  void createAndAddNestedNode(StringRef name, llvm::MDNode *node);

  llvm::MDNode *convertFollowup(LoopAnnotationAttr loopMD);
  void convertLoopOptions(LoopVectorizeAttr options);
  void convertLoopOptions(LoopInterleaveAttr options);
  void convertLoopOptions(LoopUnrollAttr options);
  void convertLoopOptions(LoopUnrollAndJamAttr options);
  void convertLoopOptions(LoopLICMAttr options);
  void convertLoopOptions(LoopDistributeAttr options);
  void convertLoopOptions(LoopPipelineAttr options);
  void convertPassthrough(DictionaryAttr passthrough);

  LoopAnnotationAttr attr;
  llvm::LLVMContext &ctx;
  LLVM::ModuleTranslation &moduleTranslation;
  Operation &opInst;
  llvm::SmallVector<llvm::Metadata *> mdNodes;
};
} // namespace

void LoopAnnotationConverter::createAndAddUnitNode(StringRef name) {
  mdNodes.push_back(llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name)}));
}

void LoopAnnotationConverter::createAndAddBoolNode(StringRef name, bool val) {
  llvm::Constant *cstValue = llvm::ConstantInt::getBool(ctx, val);
  mdNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name),
                              llvm::ConstantAsMetadata::get(cstValue)}));
}

void LoopAnnotationConverter::createAndAddI32Node(StringRef name,
                                                  uint32_t val) {
  llvm::Constant *cstValue = llvm::ConstantInt::get(
      llvm::IntegerType::get(ctx, /*NumBits=*/32), val, /*isSigned=*/false);
  mdNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name),
                              llvm::ConstantAsMetadata::get(cstValue)}));
}
void LoopAnnotationConverter::createAndAddNestedNode(StringRef name,
                                                     llvm::MDNode *node) {
  mdNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name), node}));
}

llvm::MDNode *
LoopAnnotationConverter::convertFollowup(LoopAnnotationAttr loopMD) {
  return LoopAnnotationConverter(loopMD, ctx, moduleTranslation, opInst)
      .convert();
}

void LoopAnnotationConverter::convertLoopOptions(LoopVectorizeAttr options) {
  if (auto disable = options.getDisable())
    createAndAddBoolNode("llvm.loop.vectorize.enable", !disable.getValue());
  if (auto predicateEnable = options.getPredicateEnable())
    createAndAddBoolNode("llvm.loop.vectorize.predicate.enable",
                         predicateEnable.getValue());
  if (auto scalableEnable = options.getScalableEnable())
    createAndAddBoolNode("llvm.loop.vectorize.scalable.enable",
                         scalableEnable.getValue());
  if (auto width = options.getWidth())
    createAndAddI32Node("llvm.loop.vectorize.width", width.getInt());
  if (auto followupVec = options.getFollowupVectorized())
    createAndAddNestedNode("llvm.loop.vectorize.followup_vectorized",
                           convertFollowup(followupVec));
  if (auto followupEpi = options.getFollowupEpilogue())
    createAndAddNestedNode("llvm.loop.vectorize.followup_epilogue",
                           convertFollowup(followupEpi));
  if (auto followupAll = options.getFollowupAll())
    createAndAddNestedNode("llvm.loop.vectorize.followup_all",
                           convertFollowup(followupAll));
}

void LoopAnnotationConverter::convertLoopOptions(LoopInterleaveAttr options) {
  createAndAddI32Node("llvm.loop.interleave.count",
                      options.getCount().getInt());
}

void LoopAnnotationConverter::convertLoopOptions(LoopUnrollAttr options) {
  if (auto disable = options.getDisable()) {
    StringRef name;
    if (disable.getValue())
      name = "llvm.loop.unroll.disable";
    else
      name = "llvm.loop.unroll.enable";
    createAndAddUnitNode(name);
  }
  if (auto count = options.getCount())
    createAndAddI32Node("llvm.loop.unroll.count", count.getInt());
  if (auto runtimeDisable = options.getRuntimeDisable())
    createAndAddBoolNode("llvm.loop.unroll.runtime.disable",
                         runtimeDisable.getValue());
  if (auto full = options.getFull())
    if (full.getValue())
      createAndAddUnitNode("llvm.loop.unroll.full");
  if (auto followup = options.getFollowup())
    createAndAddNestedNode("llvm.loop.unroll.followup",
                           convertFollowup(followup));
  if (auto followupRem = options.getFollowupRemainder())
    createAndAddNestedNode("llvm.loop.unroll.followup_remainder",
                           convertFollowup(followupRem));
}

void LoopAnnotationConverter::convertLoopOptions(LoopUnrollAndJamAttr options) {
  if (auto disable = options.getDisable()) {
    StringRef name;
    if (disable.getValue())
      name = "llvm.loop.unroll_and_jam.disable";
    else
      name = "llvm.loop.unroll_and_jam.enable";
    createAndAddUnitNode(name);
  }
  if (auto count = options.getCount())
    createAndAddI32Node("llvm.loop.unroll_and_jam.count", count.getInt());
  if (auto followupOut = options.getFollowupOuter())
    createAndAddNestedNode("llvm.loop.unroll_and_jam.followup_outer",
                           convertFollowup(followupOut));
  if (auto followupIn = options.getFollowupInner())
    createAndAddNestedNode("llvm.loop.unroll_and_jam.followup_inner",
                           convertFollowup(followupIn));
  if (auto followupRemOut = options.getFollowupRemainderOuter())
    createAndAddNestedNode("llvm.loop.unroll_and_jam.followup_remainder_outer",
                           convertFollowup(followupRemOut));
  if (auto followupRemIn = options.getFollowupRemainderInner())
    createAndAddNestedNode("llvm.loop.unroll_and_jam.followup_remainder_inner",
                           convertFollowup(followupRemIn));
  if (auto followupAll = options.getFollowupAll())
    createAndAddNestedNode("llvm.loop.unroll_and_jam.followup_all",
                           convertFollowup(followupAll));
}

void LoopAnnotationConverter::convertLoopOptions(LoopLICMAttr options) {
  if (auto disable = options.getDisable())
    if (disable.getValue())
      createAndAddUnitNode("llvm.licm.disable");

  if (auto versioningDisable = options.getVersioningDisable())
    if (versioningDisable.getValue())
      createAndAddUnitNode("llvm.loop.licm_versioning.disable");
}

void LoopAnnotationConverter::convertLoopOptions(LoopDistributeAttr options) {
  if (auto disable = options.getDisable())
    createAndAddBoolNode("llvm.loop.distribute.enable", !disable.getValue());
  if (auto followupCoi = options.getFollowupCoincident())
    createAndAddNestedNode("llvm.loop.distribute.followup_coincident",
                           convertFollowup(followupCoi));
  if (auto followupSeq = options.getFollowupSequential())
    createAndAddNestedNode("llvm.loop.distribute.followup_sequential",
                           convertFollowup(followupSeq));
  if (auto followupFb = options.getFollowupFallback())
    createAndAddNestedNode("llvm.loop.distribute.followup_fallback",
                           convertFollowup(followupFb));
  if (auto followupAll = options.getFollowupAll())
    createAndAddNestedNode("llvm.loop.distribute.followup_all",
                           convertFollowup(followupAll));
}

void LoopAnnotationConverter::convertLoopOptions(LoopPipelineAttr options) {
  if (auto disable = options.getDisable())
    createAndAddBoolNode("llvm.loop.pipeline.disable", disable.getValue());
  if (auto initiationinterval = options.getInitiationinterval())
    createAndAddI32Node("llvm.loop.pipeline.initiationinterval",
                        initiationinterval.getInt());
}

llvm::MDNode *LoopAnnotationConverter::convert() {
  llvm::MDNode *loopMD = moduleTranslation.lookupLoopMetadata(attr);
  if (loopMD)
    return loopMD;

  // Reserve operand 0 for loop id self reference.
  auto dummy = llvm::MDNode::getTemporary(ctx, std::nullopt);
  mdNodes.push_back(dummy.get());

  if (auto disableNonforced = attr.getDisableNonforced())
    if (disableNonforced.getValue())
      createAndAddUnitNode("llvm.loop.disable_nonforced");
  if (auto mustProgress = attr.getMustProgress())
    if (mustProgress.getValue())
      createAndAddUnitNode("llvm.loop.mustprogress");

  if (auto options = attr.getVectorize())
    convertLoopOptions(options);
  if (auto options = attr.getInterleave())
    convertLoopOptions(options);
  if (auto options = attr.getUnroll())
    convertLoopOptions(options);
  if (auto options = attr.getUnrollAndJam())
    convertLoopOptions(options);
  if (auto options = attr.getLicm())
    convertLoopOptions(options);
  if (auto options = attr.getDistribute())
    convertLoopOptions(options);
  if (auto options = attr.getPipeline())
    convertLoopOptions(options);

  ArrayRef<SymbolRefAttr> parallelAccessGroups = attr.getParallelAccesses();
  if (!parallelAccessGroups.empty()) {
    SmallVector<llvm::Metadata *> parallelAccess;
    parallelAccess.push_back(
        llvm::MDString::get(ctx, "llvm.loop.parallel_accesses"));
    for (SymbolRefAttr accessGroupRef : parallelAccessGroups)
      parallelAccess.push_back(
          moduleTranslation.getAccessGroup(opInst, accessGroupRef));
    mdNodes.push_back(llvm::MDNode::get(ctx, parallelAccess));
  }

  // Create loop options and set the first operand to itself.
  loopMD = llvm::MDNode::get(ctx, mdNodes);
  loopMD->replaceOperandWith(0, loopMD);

  // Store a map from this Attribute to the LLVM metadata in case we
  // encounter it again.
  moduleTranslation.mapLoopMetadata(attr, loopMD);
  return loopMD;
}

static void setLoopMetadata(Operation &opInst, llvm::Instruction &llvmInst,
                            llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::LLVMContext &ctx = module->getContext();
  auto attr =
      opInst.getAttrOfType<LoopAnnotationAttr>(LLVMDialect::getLoopAttrName());
  if (!attr)
    return;
  llvm::MDNode *loopMD =
      LoopAnnotationConverter(attr, ctx, moduleTranslation, opInst).convert();
  llvmInst.setMetadata(llvm::LLVMContext::MD_loop, loopMD);
}

/// Convert the value of a DenseI64ArrayAttr to a vector of unsigned indices.
static SmallVector<unsigned> extractPosition(ArrayRef<int64_t> indices) {
  SmallVector<unsigned> position;
  llvm::append_range(position, indices);
  return position;
}

/// Get the declaration of an overloaded llvm intrinsic. First we get the
/// overloaded argument types and/or result type from the CallIntrinsicOp, and
/// then use those to get the correct declaration of the overloaded intrinsic.
static FailureOr<llvm::Function *>
getOverloadedDeclaration(CallIntrinsicOp &op, llvm::Intrinsic::ID id,
                         llvm::Module *module,
                         LLVM::ModuleTranslation &moduleTranslation) {
  SmallVector<llvm::Type *, 8> allArgTys;
  for (Type type : op->getOperandTypes())
    allArgTys.push_back(moduleTranslation.convertType(type));

  llvm::Type *resTy;
  if (op.getNumResults() == 0)
    resTy = llvm::Type::getVoidTy(module->getContext());
  else
    resTy = moduleTranslation.convertType(op.getResult(0).getType());

  // ATM we do not support variadic intrinsics.
  llvm::FunctionType *ft = llvm::FunctionType::get(resTy, allArgTys, false);

  SmallVector<llvm::Intrinsic::IITDescriptor, 8> table;
  getIntrinsicInfoTableEntries(id, table);
  ArrayRef<llvm::Intrinsic::IITDescriptor> tableRef = table;

  SmallVector<llvm::Type *, 8> overloadedArgTys;
  if (llvm::Intrinsic::matchIntrinsicSignature(ft, tableRef,
                                               overloadedArgTys) !=
      llvm::Intrinsic::MatchIntrinsicTypesResult::MatchIntrinsicTypes_Match) {
    return op.emitOpError("intrinsic type is not a match");
  }

  ArrayRef<llvm::Type *> overloadedArgTysRef = overloadedArgTys;
  return llvm::Intrinsic::getDeclaration(module, id, overloadedArgTysRef);
}

/// Builder for LLVM_CallIntrinsicOp
static LogicalResult
convertCallLLVMIntrinsicOp(CallIntrinsicOp &op, llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Intrinsic::ID id =
      llvm::Function::lookupIntrinsicID(op.getIntrinAttr());
  if (!id)
    return op.emitOpError()
           << "couldn't find intrinsic: " << op.getIntrinAttr();

  llvm::Function *fn = nullptr;
  if (llvm::Intrinsic::isOverloaded(id)) {
    auto fnOrFailure =
        getOverloadedDeclaration(op, id, module, moduleTranslation);
    if (failed(fnOrFailure))
      return failure();
    fn = *fnOrFailure;
  } else {
    fn = llvm::Intrinsic::getDeclaration(module, id, {});
  }

  auto *inst =
      builder.CreateCall(fn, moduleTranslation.lookupValues(op.getOperands()));
  if (op.getNumResults() == 1)
    moduleTranslation.mapValue(op->getResults().front()) = inst;
  return success();
}

/// Constructs branch weights metadata if the provided `weights` hold a value,
/// otherwise returns nullptr.
static llvm::MDNode *
convertBranchWeights(std::optional<ElementsAttr> weights,
                     LLVM::ModuleTranslation &moduleTranslation) {
  if (!weights)
    return nullptr;
  SmallVector<uint32_t> weightValues;
  weightValues.reserve(weights->size());
  for (APInt weight : weights->cast<DenseIntElementsAttr>())
    weightValues.push_back(weight.getLimitedValue());
  return llvm::MDBuilder(moduleTranslation.getLLVMContext())
      .createBranchWeights(weightValues);
}

static LogicalResult
convertOperationImpl(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {

  llvm::IRBuilder<>::FastMathFlagGuard fmfGuard(builder);
  if (auto fmf = dyn_cast<FastmathFlagsInterface>(opInst))
    builder.setFastMathFlags(getFastmathFlags(fmf));

#include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicConversions.inc"

  // Helper function to reconstruct the function type for an indirect call given
  // the result and argument types. The function cannot reconstruct the type of
  // variadic functions since the call operation does not carry enough
  // information to distinguish normal and variadic arguments. Supporting
  // indirect variadic calls requires an additional type attribute on the call
  // operation that stores the LLVM function type of the callee.
  // TODO: Support indirect calls to variadic function pointers.
  auto getCalleeFunctionType = [&](TypeRange resultTypes, ValueRange args) {
    Type resultType = resultTypes.empty()
                          ? LLVMVoidType::get(opInst.getContext())
                          : resultTypes.front();
    return llvm::cast<llvm::FunctionType>(moduleTranslation.convertType(
        LLVMFunctionType::get(opInst.getContext(), resultType,
                              llvm::to_vector(args.getTypes()), false)));
  };

  // Emit function calls.  If the "callee" attribute is present, this is a
  // direct function call and we also need to look up the remapped function
  // itself.  Otherwise, this is an indirect call and the callee is the first
  // operand, look it up as a normal value.
  if (auto callOp = dyn_cast<LLVM::CallOp>(opInst)) {
    auto operands = moduleTranslation.lookupValues(callOp.getOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    llvm::CallInst *call;
    if (auto attr = callOp.getCalleeAttr()) {
      call = builder.CreateCall(
          moduleTranslation.lookupFunction(attr.getValue()), operandsRef);
    } else {
      call = builder.CreateCall(getCalleeFunctionType(callOp.getResultTypes(),
                                                      callOp.getArgOperands()),
                                operandsRef.front(), operandsRef.drop_front());
    }
    llvm::MDNode *branchWeights =
        convertBranchWeights(callOp.getBranchWeights(), moduleTranslation);
    if (branchWeights)
      call->setMetadata(llvm::LLVMContext::MD_prof, branchWeights);
    // If the called function has a result, remap the corresponding value.  Note
    // that LLVM IR dialect CallOp has either 0 or 1 result.
    if (opInst.getNumResults() != 0) {
      moduleTranslation.mapValue(opInst.getResult(0), call);
      return success();
    }
    // Check that LLVM call returns void for 0-result functions.
    return success(call->getType()->isVoidTy());
  }

  if (auto inlineAsmOp = dyn_cast<LLVM::InlineAsmOp>(opInst)) {
    // TODO: refactor function type creation which usually occurs in std-LLVM
    // conversion.
    SmallVector<Type, 8> operandTypes;
    llvm::append_range(operandTypes, inlineAsmOp.getOperands().getTypes());

    Type resultType;
    if (inlineAsmOp.getNumResults() == 0) {
      resultType = LLVM::LLVMVoidType::get(&moduleTranslation.getContext());
    } else {
      assert(inlineAsmOp.getNumResults() == 1);
      resultType = inlineAsmOp.getResultTypes()[0];
    }
    auto ft = LLVM::LLVMFunctionType::get(resultType, operandTypes);
    llvm::InlineAsm *inlineAsmInst =
        inlineAsmOp.getAsmDialect()
            ? llvm::InlineAsm::get(
                  static_cast<llvm::FunctionType *>(
                      moduleTranslation.convertType(ft)),
                  inlineAsmOp.getAsmString(), inlineAsmOp.getConstraints(),
                  inlineAsmOp.getHasSideEffects(),
                  inlineAsmOp.getIsAlignStack(),
                  convertAsmDialectToLLVM(*inlineAsmOp.getAsmDialect()))
            : llvm::InlineAsm::get(static_cast<llvm::FunctionType *>(
                                       moduleTranslation.convertType(ft)),
                                   inlineAsmOp.getAsmString(),
                                   inlineAsmOp.getConstraints(),
                                   inlineAsmOp.getHasSideEffects(),
                                   inlineAsmOp.getIsAlignStack());
    llvm::CallInst *inst = builder.CreateCall(
        inlineAsmInst,
        moduleTranslation.lookupValues(inlineAsmOp.getOperands()));
    if (auto maybeOperandAttrs = inlineAsmOp.getOperandAttrs()) {
      llvm::AttributeList attrList;
      for (const auto &it : llvm::enumerate(*maybeOperandAttrs)) {
        Attribute attr = it.value();
        if (!attr)
          continue;
        DictionaryAttr dAttr = attr.cast<DictionaryAttr>();
        TypeAttr tAttr =
            dAttr.get(InlineAsmOp::getElementTypeAttrName()).cast<TypeAttr>();
        llvm::AttrBuilder b(moduleTranslation.getLLVMContext());
        llvm::Type *ty = moduleTranslation.convertType(tAttr.getValue());
        b.addTypeAttr(llvm::Attribute::ElementType, ty);
        // shift to account for the returned value (this is always 1 aggregate
        // value in LLVM).
        int shift = (opInst.getNumResults() > 0) ? 1 : 0;
        attrList = attrList.addAttributesAtIndex(
            moduleTranslation.getLLVMContext(), it.index() + shift, b);
      }
      inst->setAttributes(attrList);
    }

    if (opInst.getNumResults() != 0)
      moduleTranslation.mapValue(opInst.getResult(0), inst);
    return success();
  }

  if (auto invOp = dyn_cast<LLVM::InvokeOp>(opInst)) {
    auto operands = moduleTranslation.lookupValues(invOp.getCalleeOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    llvm::Instruction *result;
    if (auto attr = opInst.getAttrOfType<FlatSymbolRefAttr>("callee")) {
      result = builder.CreateInvoke(
          moduleTranslation.lookupFunction(attr.getValue()),
          moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
          moduleTranslation.lookupBlock(invOp.getSuccessor(1)), operandsRef);
    } else {
      result = builder.CreateInvoke(
          getCalleeFunctionType(invOp.getResultTypes(), invOp.getArgOperands()),
          operandsRef.front(),
          moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
          moduleTranslation.lookupBlock(invOp.getSuccessor(1)),
          operandsRef.drop_front());
    }
    llvm::MDNode *branchWeights =
        convertBranchWeights(invOp.getBranchWeights(), moduleTranslation);
    if (branchWeights)
      result->setMetadata(llvm::LLVMContext::MD_prof, branchWeights);
    moduleTranslation.mapBranch(invOp, result);
    // InvokeOp can only have 0 or 1 result
    if (invOp->getNumResults() != 0) {
      moduleTranslation.mapValue(opInst.getResult(0), result);
      return success();
    }
    return success(result->getType()->isVoidTy());
  }

  if (auto lpOp = dyn_cast<LLVM::LandingpadOp>(opInst)) {
    llvm::Type *ty = moduleTranslation.convertType(lpOp.getType());
    llvm::LandingPadInst *lpi =
        builder.CreateLandingPad(ty, lpOp.getNumOperands());
    lpi->setCleanup(lpOp.getCleanup());

    // Add clauses
    for (llvm::Value *operand :
         moduleTranslation.lookupValues(lpOp.getOperands())) {
      // All operands should be constant - checked by verifier
      if (auto *constOperand = dyn_cast<llvm::Constant>(operand))
        lpi->addClause(constOperand);
    }
    moduleTranslation.mapValue(lpOp.getResult(), lpi);
    return success();
  }

  // Emit branches.  We need to look up the remapped blocks and ignore the
  // block arguments that were transformed into PHI nodes.
  if (auto brOp = dyn_cast<LLVM::BrOp>(opInst)) {
    llvm::BranchInst *branch =
        builder.CreateBr(moduleTranslation.lookupBlock(brOp.getSuccessor()));
    moduleTranslation.mapBranch(&opInst, branch);
    setLoopMetadata(opInst, *branch, builder, moduleTranslation);
    return success();
  }
  if (auto condbrOp = dyn_cast<LLVM::CondBrOp>(opInst)) {
    llvm::MDNode *branchWeights =
        convertBranchWeights(condbrOp.getBranchWeights(), moduleTranslation);
    llvm::BranchInst *branch = builder.CreateCondBr(
        moduleTranslation.lookupValue(condbrOp.getOperand(0)),
        moduleTranslation.lookupBlock(condbrOp.getSuccessor(0)),
        moduleTranslation.lookupBlock(condbrOp.getSuccessor(1)), branchWeights);
    moduleTranslation.mapBranch(&opInst, branch);
    setLoopMetadata(opInst, *branch, builder, moduleTranslation);
    return success();
  }
  if (auto switchOp = dyn_cast<LLVM::SwitchOp>(opInst)) {
    llvm::MDNode *branchWeights =
        convertBranchWeights(switchOp.getBranchWeights(), moduleTranslation);
    llvm::SwitchInst *switchInst = builder.CreateSwitch(
        moduleTranslation.lookupValue(switchOp.getValue()),
        moduleTranslation.lookupBlock(switchOp.getDefaultDestination()),
        switchOp.getCaseDestinations().size(), branchWeights);

    auto *ty = llvm::cast<llvm::IntegerType>(
        moduleTranslation.convertType(switchOp.getValue().getType()));
    for (auto i :
         llvm::zip(switchOp.getCaseValues()->cast<DenseIntElementsAttr>(),
                   switchOp.getCaseDestinations()))
      switchInst->addCase(
          llvm::ConstantInt::get(ty, std::get<0>(i).getLimitedValue()),
          moduleTranslation.lookupBlock(std::get<1>(i)));

    moduleTranslation.mapBranch(&opInst, switchInst);
    return success();
  }

  // Emit addressof.  We need to look up the global value referenced by the
  // operation and store it in the MLIR-to-LLVM value mapping.  This does not
  // emit any LLVM instruction.
  if (auto addressOfOp = dyn_cast<LLVM::AddressOfOp>(opInst)) {
    LLVM::GlobalOp global =
        addressOfOp.getGlobal(moduleTranslation.symbolTable());
    LLVM::LLVMFuncOp function =
        addressOfOp.getFunction(moduleTranslation.symbolTable());

    // The verifier should not have allowed this.
    assert((global || function) &&
           "referencing an undefined global or function");

    moduleTranslation.mapValue(
        addressOfOp.getResult(),
        global ? moduleTranslation.lookupGlobal(global)
               : moduleTranslation.lookupFunction(function.getName()));
    return success();
  }

  return failure();
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the LLVM dialect to LLVM IR.
class LLVMDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    return convertOperationImpl(*op, builder, moduleTranslation);
  }
};
} // namespace

void mlir::registerLLVMDialectTranslation(DialectRegistry &registry) {
  registry.insert<LLVM::LLVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerLLVMDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
