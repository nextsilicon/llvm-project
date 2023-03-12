//===- InliningUtils.cpp ---- Misc utilities for inlining -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous inlining utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/InliningUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "inlining"

using namespace mlir;

/// Remap locations from the inlined blocks with CallSiteLoc locations with the
/// provided caller location.
static void
remapInlinedLocations(iterator_range<Region::iterator> inlinedBlocks,
                      Location callerLoc) {
  DenseMap<Location, Location> mappedLocations;
  auto remapOpLoc = [&](Operation *op) {
    auto it = mappedLocations.find(op->getLoc());
    if (it == mappedLocations.end()) {
      auto newLoc = CallSiteLoc::get(op->getLoc(), callerLoc);
      it = mappedLocations.try_emplace(op->getLoc(), newLoc).first;
    }
    op->setLoc(it->second);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOpLoc);
}

static void remapInlinedOperands(iterator_range<Region::iterator> inlinedBlocks,
                                 IRMapping &mapper) {
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOperands);
}

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

bool InlinerInterface::isLegalToInline(Operation *call, Operation *callable,
                                       bool wouldBeCloned) const {
  if (auto *handler = getInterfaceFor(call))
    return handler->isLegalToInline(call, callable, wouldBeCloned);
  return false;
}

bool InlinerInterface::isLegalToInline(Region *dest, Region *src,
                                       bool wouldBeCloned,
                                       IRMapping &valueMapping) const {
  if (auto *handler = getInterfaceFor(dest->getParentOp()))
    return handler->isLegalToInline(dest, src, wouldBeCloned, valueMapping);
  return false;
}

bool InlinerInterface::isLegalToInline(Operation *op, Region *dest,
                                       bool wouldBeCloned,
                                       IRMapping &valueMapping) const {
  if (auto *handler = getInterfaceFor(op))
    return handler->isLegalToInline(op, dest, wouldBeCloned, valueMapping);
  return false;
}

bool InlinerInterface::isTypeConvertible(Operation *call, Operation *callable,
                                         Type sourceType, Type targetType,
                                         DictionaryAttr argOrResAttrs,
                                         bool isResult) const {
  if (auto *handler = getInterfaceFor(call))
    return handler->isTypeConvertible(call, callable, sourceType, targetType,
                                      argOrResAttrs, isResult);
  return false;
}

bool InlinerInterface::shouldAnalyzeRecursively(Operation *op) const {
  auto *handler = getInterfaceFor(op);
  return handler ? handler->shouldAnalyzeRecursively(op) : true;
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op, Block *newDest) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, newDest);
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op,
                                        ArrayRef<Value> valuesToRepl) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, valuesToRepl);
}

Value InlinerInterface::handleArgument(OpBuilder &builder, Operation *call,
                                       Operation *callable, Value argument,
                                       Type targetType,
                                       DictionaryAttr argumentAttrs) const {
  auto *handler = getInterfaceFor(call);
  assert(handler && "expected valid dialect handler");
  return handler->handleArgument(builder, call, callable, argument, targetType,
                                 argumentAttrs);
}

Value InlinerInterface::handleResult(OpBuilder &builder, Operation *call,
                                     Operation *callable, Value result,
                                     Type targetType,
                                     DictionaryAttr resultAttrs) const {
  auto *handler = getInterfaceFor(call);
  assert(handler && "expected valid dialect handler");
  return handler->handleResult(builder, call, callable, result, targetType,
                               resultAttrs);
}

void InlinerInterface::processInlinedCallBlocks(
    Operation *call, iterator_range<Region::iterator> inlinedBlocks) const {
  auto *handler = getInterfaceFor(call);
  assert(handler && "expected valid dialect handler");
  handler->processInlinedCallBlocks(call, inlinedBlocks);
}

/// Utility to check that all of the operations within 'src' can be inlined.
static bool isLegalToInline(InlinerInterface &interface, Region *src,
                            Region *insertRegion, bool shouldCloneInlinedRegion,
                            IRMapping &valueMapping) {
  for (auto &block : *src) {
    for (auto &op : block) {
      // Check this operation.
      if (!interface.isLegalToInline(&op, insertRegion,
                                     shouldCloneInlinedRegion, valueMapping)) {
        LLVM_DEBUG({
          llvm::dbgs() << "* Illegal to inline because of op: ";
          op.dump();
        });
        return false;
      }
      // Check any nested regions.
      if (interface.shouldAnalyzeRecursively(&op) &&
          llvm::any_of(op.getRegions(), [&](Region &region) {
            return !isLegalToInline(interface, &region, insertRegion,
                                    shouldCloneInlinedRegion, valueMapping);
          }))
        return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Inline Methods
//===----------------------------------------------------------------------===//

/// Returns the array of argument attribute dictionaries. The attribute
/// dictionaries are non-null even if no attributes are present.
static SmallVector<DictionaryAttr>
getArgumentAttributes(CallableOpInterface callable) {
  SmallVector<DictionaryAttr> argAttrs(
      callable.getCallableRegion()->getNumArguments(),
      DictionaryAttr::get(callable->getContext(), {}));
  if (ArrayAttr arrayAttr = callable.getCallableArgAttrs()) {
    assert(arrayAttr.size() == argAttrs.size());
    for (auto [idx, attr] : llvm::enumerate(arrayAttr))
      argAttrs[idx] = cast<DictionaryAttr>(attr);
  }
  return argAttrs;
}

/// Returns the array of result attribute dictionaries. The attribute
/// dictionaries are non-null even if no attributes are present.
static SmallVector<DictionaryAttr>
getResultAttributes(CallableOpInterface callable) {
  SmallVector<DictionaryAttr> resAttrs(
      callable.getCallableResults().size(),
      DictionaryAttr::get(callable->getContext(), {}));
  if (ArrayAttr arrayAttr = callable.getCallableResAttrs()) {
    assert(arrayAttr.size() == resAttrs.size());
    for (auto [idx, attr] : llvm::enumerate(arrayAttr))
      resAttrs[idx] = cast<DictionaryAttr>(attr);
  }
  return resAttrs;
}

static void handleArgumentImpl(InlinerInterface &interface, OpBuilder &builder,
                               CallOpInterface call,
                               CallableOpInterface callable,
                               IRMapping &mapper) {
  if (!call || !callable)
    return;

  // Unpack the argument attributes.
  SmallVector<DictionaryAttr> argAttrs = getArgumentAttributes(callable);

  // Run the argument attribute handler for the given argument and attribute.
  for (auto [blockArg, argAttr] : llvm::zip_equal(
           callable.getCallableRegion()->getArguments(), argAttrs)) {
    Value newArgument = interface.handleArgument(builder, call, callable,
                                                 mapper.lookup(blockArg),
                                                 blockArg.getType(), argAttr);
    assert(newArgument.getType() == blockArg.getType() &&
           "expected the handled argument type to match the target type");

    // Update the mapping to point the new argument returned by the handler.
    mapper.map(blockArg, newArgument);
  }
}

static void handleResultImpl(InlinerInterface &interface, OpBuilder &builder,
                             CallOpInterface call, CallableOpInterface callable,
                             ValueRange results) {
  if (!call || !callable)
    return;

  // Unpack the result attributes.
  SmallVector<DictionaryAttr> resAttrs = getResultAttributes(callable);

  // Run the result attribute handler for the given result and attribute.
  SmallVector<DictionaryAttr> resultAttributes;
  for (auto [result, resAttr, callResult] :
       llvm::zip_equal(results, resAttrs, call->getResults())) {
    // Store the original result users before running the handler.
    DenseSet<Operation *> resultUsers;
    for (Operation *user : result.getUsers())
      resultUsers.insert(user);

    Value newResult = interface.handleResult(builder, call, callable, result,
                                             callResult.getType(), resAttr);
    assert(newResult.getType() == callResult.getType() &&
           "expected the handled result type to match the target type");

    // Replace the result uses except for the ones introduce by the handler.
    result.replaceUsesWithIf(newResult, [&](OpOperand &operand) {
      return resultUsers.count(operand.getOwner());
    });
  }
}

static LogicalResult
inlineRegionImpl(InlinerInterface &interface, Region *src, Block *inlineBlock,
                 Block::iterator inlinePoint, IRMapping &mapper,
                 ValueRange resultsToReplace, TypeRange regionResultTypes,
                 std::optional<Location> inlineLoc,
                 bool shouldCloneInlinedRegion, Operation *call = nullptr) {
  assert(resultsToReplace.size() == regionResultTypes.size());
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  // Check that all of the region arguments have been mapped.
  auto *srcEntryBlock = &src->front();
  if (llvm::any_of(srcEntryBlock->getArguments(),
                   [&](BlockArgument arg) { return !mapper.contains(arg); }))
    return failure();

  // Check that the operations within the source region are valid to inline.
  Region *insertRegion = inlineBlock->getParent();
  if (!interface.isLegalToInline(insertRegion, src, shouldCloneInlinedRegion,
                                 mapper) ||
      !isLegalToInline(interface, src, insertRegion, shouldCloneInlinedRegion,
                       mapper))
    return failure();

  // Run the argument attribute handler before inlining the callable region.
  OpBuilder builder(inlineBlock, inlinePoint);
  auto callable = dyn_cast<CallableOpInterface>(src->getParentOp());
  handleArgumentImpl(interface, builder, call, callable, mapper);

  // Check to see if the region is being cloned, or moved inline. In either
  // case, move the new blocks after the 'insertBlock' to improve IR
  // readability.
  Block *postInsertBlock = inlineBlock->splitBlock(inlinePoint);
  if (shouldCloneInlinedRegion)
    src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapper);
  else
    insertRegion->getBlocks().splice(postInsertBlock->getIterator(),
                                     src->getBlocks(), src->begin(),
                                     src->end());

  // Get the range of newly inserted blocks.
  auto newBlocks = llvm::make_range(std::next(inlineBlock->getIterator()),
                                    postInsertBlock->getIterator());
  Block *firstNewBlock = &*newBlocks.begin();

  // Remap the locations of the inlined operations if a valid source location
  // was provided.
  if (inlineLoc && !inlineLoc->isa<UnknownLoc>())
    remapInlinedLocations(newBlocks, *inlineLoc);

  // If the blocks were moved in-place, make sure to remap any necessary
  // operands.
  if (!shouldCloneInlinedRegion)
    remapInlinedOperands(newBlocks, mapper);

  // Process the newly inlined blocks.
  if (call)
    interface.processInlinedCallBlocks(call, newBlocks);
  interface.processInlinedBlocks(newBlocks);

  // Handle the case where only a single block was inlined.
  if (std::next(newBlocks.begin()) == newBlocks.end()) {
    // Run the result attribute handler on the terminator operands.
    Operation *firstBlockTerminator = firstNewBlock->getTerminator();
    builder.setInsertionPoint(firstBlockTerminator);
    handleResultImpl(interface, builder, call, callable,
                     firstBlockTerminator->getOperands());

    // Have the interface handle the terminator of this block.
    interface.handleTerminator(firstBlockTerminator,
                               llvm::to_vector<6>(resultsToReplace));
    firstBlockTerminator->erase();

    // Merge the post insert block into the cloned entry block.
    firstNewBlock->getOperations().splice(firstNewBlock->end(),
                                          postInsertBlock->getOperations());
    postInsertBlock->erase();
  } else {
    // Otherwise, there were multiple blocks inlined. Add arguments to the post
    // insertion block to represent the results to replace.
    for (const auto &resultToRepl : llvm::enumerate(resultsToReplace)) {
      resultToRepl.value().replaceAllUsesWith(
          postInsertBlock->addArgument(regionResultTypes[resultToRepl.index()],
                                       resultToRepl.value().getLoc()));
    }

    // Run the result attribute handler on the post insertion block arguments.
    builder.setInsertionPointToStart(postInsertBlock);
    handleResultImpl(interface, builder, call, callable,
                     postInsertBlock->getArguments());

    /// Handle the terminators for each of the new blocks.
    for (auto &newBlock : newBlocks)
      interface.handleTerminator(newBlock.getTerminator(), postInsertBlock);
  }

  // Splice the instructions of the inlined entry block into the insert block.
  inlineBlock->getOperations().splice(inlineBlock->end(),
                                      firstNewBlock->getOperations());
  firstNewBlock->erase();
  return success();
}

static LogicalResult
inlineRegionImpl(InlinerInterface &interface, Region *src, Block *inlineBlock,
                 Block::iterator inlinePoint, ValueRange inlinedOperands,
                 ValueRange resultsToReplace, std::optional<Location> inlineLoc,
                 bool shouldCloneInlinedRegion, Operation *call = nullptr) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  auto *entryBlock = &src->front();
  if (inlinedOperands.size() != entryBlock->getNumArguments())
    return failure();

  // Map the provided call operands to the arguments of the region.
  IRMapping mapper;
  for (unsigned i = 0, e = inlinedOperands.size(); i != e; ++i) {
    // Verify that the types of the provided values match the function argument
    // types.
    BlockArgument regionArg = entryBlock->getArgument(i);
    if (inlinedOperands[i].getType() != regionArg.getType())
      return failure();
    mapper.map(regionArg, inlinedOperands[i]);
  }

  // Call into the main region inliner function.
  return inlineRegionImpl(interface, src, inlineBlock, inlinePoint, mapper,
                          resultsToReplace, resultsToReplace.getTypes(),
                          inlineLoc, shouldCloneInlinedRegion, call);
}

LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint, IRMapping &mapper,
                                 ValueRange resultsToReplace,
                                 TypeRange regionResultTypes,
                                 std::optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  return inlineRegion(interface, src, inlinePoint->getBlock(),
                      ++inlinePoint->getIterator(), mapper, resultsToReplace,
                      regionResultTypes, inlineLoc, shouldCloneInlinedRegion);
}
LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Block *inlineBlock,
                                 Block::iterator inlinePoint, IRMapping &mapper,
                                 ValueRange resultsToReplace,
                                 TypeRange regionResultTypes,
                                 std::optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  return inlineRegionImpl(interface, src, inlineBlock, inlinePoint, mapper,
                          resultsToReplace, regionResultTypes, inlineLoc,
                          shouldCloneInlinedRegion);
}

LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint,
                                 ValueRange inlinedOperands,
                                 ValueRange resultsToReplace,
                                 std::optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  return inlineRegion(interface, src, inlinePoint->getBlock(),
                      ++inlinePoint->getIterator(), inlinedOperands,
                      resultsToReplace, inlineLoc, shouldCloneInlinedRegion);
}
LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Block *inlineBlock,
                                 Block::iterator inlinePoint,
                                 ValueRange inlinedOperands,
                                 ValueRange resultsToReplace,
                                 std::optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  return inlineRegionImpl(interface, src, inlineBlock, inlinePoint,
                          inlinedOperands, resultsToReplace, inlineLoc,
                          shouldCloneInlinedRegion);
}

/// This function inlines a given region, 'src', of a callable operation,
/// 'callable', into the location defined by the given call operation. This
/// function returns failure if inlining is not possible, success otherwise. On
/// failure, no changes are made to the module. 'shouldCloneInlinedRegion'
/// corresponds to whether the source region should be cloned into the 'call' or
/// spliced directly.
LogicalResult mlir::inlineCall(InlinerInterface &interface,
                               CallOpInterface call,
                               CallableOpInterface callable, Region *src,
                               bool shouldCloneInlinedRegion) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();
  auto *entryBlock = &src->front();
  ArrayRef<Type> callableResultTypes = callable.getCallableResults();

  // Make sure that the number of arguments and results matchup between the call
  // and the region.
  SmallVector<Value, 8> callOperands(call.getArgOperands());
  SmallVector<Value, 8> callResults(call->getResults());
  if (callOperands.size() != entryBlock->getNumArguments() ||
      callResults.size() != callableResultTypes.size())
    return failure();

  // Check that argument types are convertible.
  SmallVector<DictionaryAttr> argAttrs = getArgumentAttributes(callable);
  for (auto [argumentType, targetType, argumentAttrs] :
       llvm::zip_equal(call.getArgOperands().getTypes(),
                       entryBlock->getArgumentTypes(), argAttrs)) {
    if (argumentType == targetType)
      continue;
    if (!interface.isTypeConvertible(call, callable, argumentType, targetType,
                                     argumentAttrs, false))
      return failure();
  }

  // Check that result types are convertible.
  SmallVector<DictionaryAttr> resAttrs = getResultAttributes(callable);
  for (auto [resultType, targetType, resultAttrs] :
       llvm::zip_equal(callableResultTypes, call->getResultTypes(), resAttrs)) {
    if (resultType == targetType)
      continue;
    if (!interface.isTypeConvertible(call, callable, resultType, targetType,
                                     resultAttrs, true))
      return failure();
  }

  // Check that it is legal to inline the callable into the call.
  if (!interface.isLegalToInline(call, callable, shouldCloneInlinedRegion))
    return failure();

  IRMapping mapper;
  for (auto [blockArg, operand] :
       llvm::zip(entryBlock->getArguments(), callOperands))
    mapper.map(blockArg, operand);

  // Attempt to inline the call.
  if (failed(inlineRegionImpl(interface, src, call->getBlock(),
                              ++call->getIterator(), mapper, callResults,
                              callableResultTypes, call.getLoc(),
                              shouldCloneInlinedRegion, call)))
    return failure();
  return success();
}
