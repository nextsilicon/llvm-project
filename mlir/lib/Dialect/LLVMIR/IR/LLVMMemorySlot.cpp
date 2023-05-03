//===- LLVMMemorySlot.cpp - MemorySlot interfaces ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MemorySlot-related interfaces for LLVM dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

static Type getAllocaElementType(LLVM::AllocaOp alloca) {
  // This will become trivial once non-opaque pointers are gone.
  return alloca.getElemType().has_value()
             ? *alloca.getElemType()
             : alloca.getResult().getType().getElementType();
}

llvm::SmallVector<MemorySlot> LLVM::AllocaOp::getPromotableSlots() {
  if (!getOperation()->getBlock()->isEntryBlock())
    return {};

  return {MemorySlot{getResult(), getAllocaElementType(*this)}};
}

Value LLVM::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                      OpBuilder &builder) {
  return builder.create<LLVM::UndefOp>(getLoc(), slot.elemType);
}

void LLVM::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                         BlockArgument argument,
                                         OpBuilder &builder) {
  for (Operation *user : getOperation()->getUsers())
    if (auto declareOp = llvm::dyn_cast<LLVM::DbgDeclareOp>(user))
      builder.create<LLVM::DbgValueOp>(declareOp.getLoc(), argument,
                                       declareOp.getVarInfo());
}

void LLVM::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                             Value defaultValue) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  erase();
}

SmallVector<DestructibleMemorySlot> LLVM::AllocaOp::getDestructibleSlots() {
  if (!mlir::matchPattern(getArraySize(), m_One()))
    return {};

  Type elemType = getAllocaElementType(*this);
  auto destructible = dyn_cast<DestructibleTypeInterface>(elemType);
  if (!destructible)
    return {};

  Optional<DenseMap<Attribute, Type>> destructedType =
      destructible.getSubelementIndexMap();
  if (!destructedType)
    return {};

  DenseMap<Attribute, Type> allocaTypeMap;
  for (Attribute index : llvm::make_first_range(destructedType.value()))
    allocaTypeMap.insert({index, LLVM::LLVMPointerType::get(getContext())});

  return {DestructibleMemorySlot{{getResult(), elemType}, {allocaTypeMap}}};
}

DenseMap<Attribute, MemorySlot>
LLVM::AllocaOp::destruct(const DestructibleMemorySlot &slot,
                         SmallPtrSetImpl<Attribute> &usedIndices,
                         OpBuilder &builder) {
  assert(slot.ptr == getResult());
  Type elemType =
      getElemType() ? *getElemType() : getResult().getType().getElementType();

  builder.setInsertionPointAfter(*this);

  DenseMap<Attribute, MemorySlot> slotMap;
  Optional<DenseMap<Attribute, Type>> destructedType =
      cast<DestructibleTypeInterface>(elemType).getSubelementIndexMap();
  for (auto &[index, type] : destructedType.value()) {
    if (usedIndices.contains(index)) {
      auto subAlloca = builder.create<LLVM::AllocaOp>(
          getLoc(), LLVM::LLVMPointerType::get(getContext()), type,
          getArraySize());
      slotMap.try_emplace<MemorySlot>(index, {subAlloca.getResult(), type});
    }
  }

  return slotMap;
}

void LLVM::AllocaOp::handleDestructionComplete(
    const DestructibleMemorySlot &slot) {
  assert(slot.ptr == getResult());
  erase();
}

//===----------------------------------------------------------------------===//
// Interfaces for LoadOp/StoreOp
//===----------------------------------------------------------------------===//

bool LLVM::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

Value LLVM::LoadOp::getStored(const MemorySlot &slot) { return {}; }

bool LLVM::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

Value LLVM::StoreOp::getStored(const MemorySlot &slot) {
  return getAddr() == slot.ptr ? getValue() : Value();
}

bool LLVM::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, there will be enough
  // context to reconstruct the result of the load at removal time, so it can
  // be removed (provided it loads the exact stored value and is not
  // volatile).
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getResult().getType() == slot.elemType && !getVolatile_();
}

DeletionKind LLVM::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

bool LLVM::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, dropping the store is
  // fine, provided we are currently promoting its target value. Don't allow a
  // store OF the slot pointer, only INTO the slot pointer.
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr && getValue().getType() == slot.elemType &&
         !getVolatile_();
}

DeletionKind LLVM::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the stored slot
  // pointer.
  for (Operation *user : slot.ptr.getUsers())
    if (auto declareOp = llvm::dyn_cast<LLVM::DbgDeclareOp>(user))
      builder.create<LLVM::DbgValueOp>(declareOp->getLoc(), getValue(),
                                       declareOp.getVarInfo());
  return DeletionKind::Delete;
}

LogicalResult LLVM::LoadOp::ensureOnlyTypeSafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  return success(getAddr() != slot.ptr || getType() == slot.elemType);
}

LogicalResult LLVM::StoreOp::ensureOnlyTypeSafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  return success(getAddr() != slot.ptr ||
                 getValue().getType() == slot.elemType);
}

//===----------------------------------------------------------------------===//
// Interfaces for discardable OPs
//===----------------------------------------------------------------------===//

/// Conditions the deletion of the operation to the removal of all its uses.
static bool forwardToUsers(Operation *op,
                           SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

bool LLVM::BitcastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::BitcastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::AddrSpaceCastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::AddrSpaceCastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeStartOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::LifetimeStartOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeEndOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::LifetimeEndOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::DbgDeclareOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::DbgDeclareOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

static bool hasAllZeroIndices(LLVM::GEPOp gepOp) {
  return llvm::all_of(gepOp.getIndices(), [](auto index) {
    auto indexAttr = index.template dyn_cast<IntegerAttr>();
    return indexAttr && indexAttr.getValue() == 0;
  });
}

//===----------------------------------------------------------------------===//
// Interfaces for GEPOp
//===----------------------------------------------------------------------===//

bool LLVM::GEPOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  // GEP can be removed as long as it is a no-op and its users can be removed.
  if (!hasAllZeroIndices(*this))
    return false;
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::GEPOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

/// TODO: Support non-opaque pointers.
static Type computeReachedGEPType(LLVM::GEPOp gep) {
  assert(gep.getBase().getType().cast<LLVM::LLVMPointerType>().isOpaque());

  if (gep.getIndices().empty())
    return {};

  // Check the pointer indexing only targets the first element.
  auto firstIndex = gep.getIndices()[0];
  IntegerAttr indexInt = firstIndex.dyn_cast<IntegerAttr>();
  if (!indexInt || indexInt.getInt() != 0)
    return {};

  // Set the initial type currently being used for indexing. This will be
  // updated as the indices get walked over.
  Optional<Type> maybeSelectedType = gep.getElemType();
  if (!maybeSelectedType)
    return {};
  Type selectedType = *maybeSelectedType;

  // Follow the indexed elements in the gep.
  for (const auto &index : llvm::drop_begin(gep.getIndices())) {
    // Ensure the index is static and obtain it.
    IntegerAttr indexInt = index.dyn_cast<IntegerAttr>();
    if (!indexInt)
      return {};

    // Ensure the structure of the type being indexed can be reasoned about.
    assert(!selectedType.isa<LLVM::LLVMPointerType>());
    auto destructible = selectedType.dyn_cast<DestructibleTypeInterface>();
    if (!destructible)
      return {};

    // Follow the type at the index the gep is accessing, making it the new type
    // used for indexing.
    Type field = destructible.getTypeAtIndex(indexInt);
    if (!field)
      return {};
    selectedType = field;
  }

  // When there are no more indices, the type currently being used for indexing
  // is the type of the value pointed at by the returned indexed pointer.
  return selectedType;
}

LogicalResult LLVM::GEPOp::ensureOnlyTypeSafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  if (getBase() != slot.ptr)
    return success();
  if (slot.elemType != getElemType())
    return failure();
  Type reachedType = computeReachedGEPType(*this);
  if (!reachedType)
    return failure();
  mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), reachedType});
  return success();
}

bool LLVM::GEPOp::canRewire(const DestructibleMemorySlot &slot,
                            SmallPtrSetImpl<Attribute> &usedIndices,
                            SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  if (getBase() != slot.ptr || slot.elemType != getElemType())
    return false;
  Type reachedType = computeReachedGEPType(*this);
  if (!reachedType || getIndices().size() < 2)
    return false;
  auto firstLevelIndex = cast<IntegerAttr>(getIndices()[1]);
  assert(slot.elementPtrs.contains(firstLevelIndex));
  if (!slot.elementPtrs.at(firstLevelIndex).isa<LLVM::LLVMPointerType>())
    return false;
  mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), reachedType});
  usedIndices.insert(firstLevelIndex);
  return true;
}

DeletionKind LLVM::GEPOp::rewire(const DestructibleMemorySlot &slot,
                                 DenseMap<Attribute, MemorySlot> &subslots) {
  IntegerAttr firstLevelIndex = getIndices()[1].dyn_cast<IntegerAttr>();
  const MemorySlot &newSlot = subslots.at(firstLevelIndex);

  ArrayRef<int32_t> remainingIndices = getRawConstantIndices().slice(2);

  // If the GEP would become trivial after this transformation, eliminate it.
  if (llvm::all_of(remainingIndices,
                   [](int32_t index) { return index == 0; })) {
    getResult().replaceAllUsesWith(newSlot.ptr);
    return DeletionKind::Delete;
  }

  // Rewire the indices by popping off the second index.
  // Start with a single zero, then add the indices beyond the second.
  SmallVector<int32_t> newIndices(1);
  newIndices.append(remainingIndices.begin(), remainingIndices.end());
  setRawConstantIndices(newIndices);

  // Rewire the pointed type.
  setElemType(newSlot.elemType);

  // Rewire the pointer.
  getBaseMutable().assign(newSlot.ptr);

  return DeletionKind::Keep;
}

//===----------------------------------------------------------------------===//
// Interfaces for destructible types
//===----------------------------------------------------------------------===//

Optional<DenseMap<Attribute, Type>>
LLVM::LLVMStructType::getSubelementIndexMap() {
  Type i32 = IntegerType::get(getContext(), 32);
  DenseMap<Attribute, Type> destructured;
  for (auto const &[index, elemType] : llvm::enumerate(getBody()))
    destructured.insert({IntegerAttr::get(i32, index), elemType});
  return destructured;
}

Type LLVM::LLVMStructType::getTypeAtIndex(Attribute index) {
  auto indexAttr = index.dyn_cast<IntegerAttr>();
  if (!indexAttr || !indexAttr.getType().isInteger(32))
    return {};
  int32_t indexInt = indexAttr.getInt();
  ArrayRef<Type> body = getBody();
  if (indexInt < 0 || body.size() <= static_cast<uint32_t>(indexInt))
    return {};
  return body[indexInt];
}

Optional<DenseMap<Attribute, Type>>
LLVM::LLVMArrayType::getSubelementIndexMap() const {
  constexpr size_t maxArraySizeForDestruction = 16;
  if (getNumElements() > maxArraySizeForDestruction)
    return {};
  int32_t numElements = getNumElements();

  Type i32 = IntegerType::get(getContext(), 32);
  DenseMap<Attribute, Type> destructured;
  for (int32_t index = 0; index < numElements; ++index)
    destructured.insert({IntegerAttr::get(i32, index), getElementType()});
  return destructured;
}

Type LLVM::LLVMArrayType::getTypeAtIndex(Attribute index) const {
  auto indexAttr = index.dyn_cast<IntegerAttr>();
  if (!indexAttr || !indexAttr.getType().isInteger(32))
    return {};
  int32_t indexInt = indexAttr.getInt();
  if (indexInt < 0 || getNumElements() <= static_cast<uint32_t>(indexInt))
    return {};
  return getElementType();
}
