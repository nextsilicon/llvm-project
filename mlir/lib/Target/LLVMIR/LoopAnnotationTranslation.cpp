#include "LoopAnnotationTranslation.h"

using namespace mlir;

void LLVM::detail::LoopAnnotationConverter::addUnitNode(StringRef name) {
  mdNodes.push_back(llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name)}));
}

void LLVM::detail::LoopAnnotationConverter::addUnitNode(StringRef name,
                                                        BoolAttr attr) {
  if (attr && attr.getValue())
    addUnitNode(name);
}

void LLVM::detail::LoopAnnotationConverter::convertBoolNode(StringRef name,
                                                            BoolAttr attr,
                                                            bool negated) {
  if (!attr)
    return;
  bool val = negated ^ attr.getValue();
  llvm::Constant *cstValue = llvm::ConstantInt::getBool(ctx, val);
  mdNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name),
                              llvm::ConstantAsMetadata::get(cstValue)}));
}

void LLVM::detail::LoopAnnotationConverter::convertI32Node(StringRef name,
                                                           IntegerAttr attr) {
  if (!attr)
    return;
  uint32_t val = attr.getInt();
  llvm::Constant *cstValue = llvm::ConstantInt::get(
      llvm::IntegerType::get(ctx, /*NumBits=*/32), val, /*isSigned=*/false);
  mdNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name),
                              llvm::ConstantAsMetadata::get(cstValue)}));
}

void LLVM::detail::LoopAnnotationConverter::convertFollowupNode(
    StringRef name, LLVM::LoopAnnotationAttr attr) {
  if (!attr)
    return;

  llvm::MDNode *node =
      LoopAnnotationConverter(attr, ctx, moduleTranslation, opInst).convert();
  mdNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name), node}));
}

void LLVM::detail::LoopAnnotationConverter::convertLoopOptions(
    LoopVectorizeAttr options) {
  convertBoolNode("llvm.loop.vectorize.enable", options.getDisable(), true);
  convertBoolNode("llvm.loop.vectorize.predicate.enable",
                  options.getPredicateEnable());
  convertBoolNode("llvm.loop.vectorize.scalable.enable",
                  options.getScalableEnable());
  convertI32Node("llvm.loop.vectorize.width", options.getWidth());
  convertFollowupNode("llvm.loop.vectorize.followup_vectorized",
                      options.getFollowupVectorized());
  convertFollowupNode("llvm.loop.vectorize.followup_epilogue",
                      options.getFollowupEpilogue());
  convertFollowupNode("llvm.loop.vectorize.followup_all",
                      options.getFollowupAll());
}

void LLVM::detail::LoopAnnotationConverter::convertLoopOptions(
    LoopInterleaveAttr options) {
  convertI32Node("llvm.loop.interleave.count", options.getCount());
}

void LLVM::detail::LoopAnnotationConverter::convertLoopOptions(
    LoopUnrollAttr options) {
  if (auto disable = options.getDisable())
    addUnitNode(disable.getValue() ? "llvm.loop.unroll.disable"
                                   : "llvm.loop.unroll.enable");
  convertI32Node("llvm.loop.unroll.count", options.getCount());
  convertBoolNode("llvm.loop.unroll.runtime.disable",
                  options.getRuntimeDisable());
  addUnitNode("llvm.loop.unroll.full", options.getFull());
  convertFollowupNode("llvm.loop.unroll.followup", options.getFollowup());
  convertFollowupNode("llvm.loop.unroll.followup_remainder",
                      options.getFollowupRemainder());
}

void LLVM::detail::LoopAnnotationConverter::convertLoopOptions(
    LoopUnrollAndJamAttr options) {
  if (auto disable = options.getDisable())
    addUnitNode(disable.getValue() ? "llvm.loop.unroll_and_jam.disable"
                                   : "llvm.loop.unroll_and_jam.enable");
  convertI32Node("llvm.loop.unroll_and_jam.count", options.getCount());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_outer",
                      options.getFollowupOuter());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_inner",
                      options.getFollowupInner());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_remainder_outer",
                      options.getFollowupRemainderOuter());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_remainder_inner",
                      options.getFollowupRemainderInner());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_all",
                      options.getFollowupAll());
}

void LLVM::detail::LoopAnnotationConverter::convertLoopOptions(
    LoopLICMAttr options) {
  addUnitNode("llvm.licm.disable", options.getDisable());
  addUnitNode("llvm.loop.licm_versioning.disable",
              options.getVersioningDisable());
}

void LLVM::detail::LoopAnnotationConverter::convertLoopOptions(
    LoopDistributeAttr options) {
  convertBoolNode("llvm.loop.distribute.enable", options.getDisable(), true);
  convertFollowupNode("llvm.loop.distribute.followup_coincident",
                      options.getFollowupCoincident());
  convertFollowupNode("llvm.loop.distribute.followup_sequential",
                      options.getFollowupSequential());
  convertFollowupNode("llvm.loop.distribute.followup_fallback",
                      options.getFollowupFallback());
  convertFollowupNode("llvm.loop.distribute.followup_all",
                      options.getFollowupAll());
}

void LLVM::detail::LoopAnnotationConverter::convertLoopOptions(
    LoopPipelineAttr options) {
  convertBoolNode("llvm.loop.pipeline.disable", options.getDisable());
  convertI32Node("llvm.loop.pipeline.initiationinterval",
                 options.getInitiationinterval());
}

llvm::MDNode *LLVM::detail::LoopAnnotationConverter::convert() {
  llvm::MDNode *loopMD = moduleTranslation.lookupLoopMetadata(attr);
  if (loopMD)
    return loopMD;

  // Reserve operand 0 for loop id self reference.
  auto dummy = llvm::MDNode::getTemporary(ctx, std::nullopt);
  mdNodes.push_back(dummy.get());

  addUnitNode("llvm.loop.disable_nonforced", attr.getDisableNonforced());
  addUnitNode("llvm.loop.mustprogress", attr.getMustProgress());

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
          moduleTranslation.getAccessGroup(*opInst, accessGroupRef));
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
