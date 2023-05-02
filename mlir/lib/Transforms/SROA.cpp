//===-- SROA.cpp - Scalar Replacement Of Aggregates -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/SROA.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_SROA
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

Optional<MemorySlotDestructionInfo>
mlir::computeDestructionInfo(DestructibleMemorySlot &slot) {
  assert(isa<DestructibleTypeInterface>(slot.elemType));

  MemorySlotDestructionInfo info;

  SmallVector<MemorySlot> usedSafelyWorklist;

  auto scheduleAsBlockingUse = [&](OpOperand &use) {
    SmallPtrSet<OpOperand *, 4> &blockingUses =
        info.userToBlockingUses.getOrInsertDefault(use.getOwner());
    blockingUses.insert(&use);
  };

  // Initialize the analysis with the immediate users of the slot.
  for (OpOperand &use : slot.ptr.getUses()) {
    if (auto accessor =
            dyn_cast<DestructibleAccessorOpInterface>(use.getOwner())) {
      if (accessor.canRewire(slot, info.usedIndices, usedSafelyWorklist)) {
        info.accessors.push_back(accessor);
        continue;
      }
    }

    // If it cannot be shown that the operation uses the slot safely, maybe it
    // can be promoted out of using the slot?
    scheduleAsBlockingUse(use);
  }

  SmallPtrSet<OpOperand *, 16> dealtWith;
  while (!usedSafelyWorklist.empty()) {
    MemorySlot mustBeUsedSafely = usedSafelyWorklist.pop_back_val();
    for (OpOperand &subslotUse : mustBeUsedSafely.ptr.getUses()) {
      if (dealtWith.contains(&subslotUse))
        continue;
      dealtWith.insert(&subslotUse);
      Operation *subslotUser = subslotUse.getOwner();

      if (auto memOp = dyn_cast<TypeSafeOpInterface>(subslotUser))
        if (succeeded(memOp.ensureOnlyTypeSafeAccesses(mustBeUsedSafely,
                                                       usedSafelyWorklist)))
          continue;

      // If it cannot be shown that the operation uses the slot safely, maybe it
      // can be promoted out of using the slot?
      scheduleAsBlockingUse(subslotUse);
    }
  }

  SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(slot.ptr, &forwardSlice);
  for (Operation *user : forwardSlice) {
    // If the next operation has no blocking uses, everything is fine.
    if (!info.userToBlockingUses.contains(user))
      continue;

    SmallPtrSet<OpOperand *, 4> &blockingUses = info.userToBlockingUses[user];
    auto promotable = dyn_cast<PromotableOpInterface>(user);

    // An operation that has blocking uses must be promoted. If it is not
    // promotable, destruction must fail.
    if (!promotable)
      return {};

    SmallVector<OpOperand *> newBlockingUses;
    // If the operation decides it cannot deal with removing the blocking uses,
    // destruction must fail.
    if (!promotable.canUsesBeRemoved(blockingUses, newBlockingUses))
      return {};

    // Then, register any new blocking uses for coming operations.
    for (OpOperand *blockingUse : newBlockingUses) {
      assert(llvm::is_contained(user->getResults(), blockingUse->get()));

      SmallPtrSetImpl<OpOperand *> &newUserBlockingUseSet =
          info.userToBlockingUses.getOrInsertDefault(blockingUse->getOwner());
      newUserBlockingUseSet.insert(blockingUse);
    }
  }

  return info;
}

void mlir::destructSlot(DestructibleMemorySlot &slot,
                        DestructibleAllocationOpInterface allocator,
                        OpBuilder &builder, MemorySlotDestructionInfo &info) {
  OpBuilder::InsertionGuard guard(builder);

  builder.setInsertionPointToStart(slot.ptr.getParentBlock());
  DenseMap<Attribute, MemorySlot> subslots =
      allocator.destruct(slot, info.usedIndices, builder);

  llvm::SetVector<Operation *> usersToRewire;
  for (auto &[user, _] : info.userToBlockingUses)
    usersToRewire.insert(user);
  for (DestructibleAccessorOpInterface accessor : info.accessors)
    usersToRewire.insert(accessor);
  SetVector<Operation *> sortedUsersToRewire =
      mlir::topologicalSort(usersToRewire);

  llvm::SmallVector<Operation *> toErase;
  for (Operation *toRewire : llvm::reverse(sortedUsersToRewire)) {
    builder.setInsertionPointAfter(toRewire);
    if (auto accessor = dyn_cast<DestructibleAccessorOpInterface>(toRewire)) {
      if (accessor.rewire(slot, subslots) == DeletionKind::Delete)
        toErase.push_back(accessor);
      continue;
    }

    auto promotable = cast<PromotableOpInterface>(toRewire);
    if (promotable.removeBlockingUses(info.userToBlockingUses[promotable],
                                      builder) == DeletionKind::Delete)
      toErase.push_back(promotable);
  }

  for (Operation *toEraseOp : toErase)
    toEraseOp->erase();

  assert(slot.ptr.use_empty() && "at the end of destruction, the original slot "
                                 "pointer should no longer be used");

  allocator.handleDestructionComplete(slot);
}

namespace {

struct SROA : public impl::SROABase<SROA> {
  void runOnOperation() override {
    Operation *scopeOp = getOperation();
    bool changed = false;

    for (Region &region : scopeOp->getRegions()) {
      if (region.getBlocks().empty())
        continue;

      OpBuilder builder(&region.front(), region.front().begin());

      // Destructing a slot can allow for further destruction of other slots,
      // destruction is tried until no destruction succeeds.
      while (true) {
        struct DestructionJob {
          DestructibleAllocationOpInterface allocator;
          DestructibleMemorySlot slot;
          MemorySlotDestructionInfo info;
        };

        std::vector<DestructionJob> toDestruct;

        region.walk([&](DestructibleAllocationOpInterface allocator) {
          for (DestructibleMemorySlot slot : allocator.getDestructibleSlots())
            if (auto info = computeDestructionInfo(slot))
              toDestruct.emplace_back<DestructionJob>(
                  {allocator, std::move(slot), std::move(info.value())});
        });

        if (toDestruct.empty())
          break;

        for (DestructionJob &job : toDestruct)
          destructSlot(job.slot, job.allocator, builder, job.info);

        changed = true;
      }
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
