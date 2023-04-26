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
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_SROA
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

template <typename K, typename V>
static V &getOrInsertDefault(DenseMap<K, V> &map, K key) {
  return map.try_emplace(key).first->second;
}

Optional<MemorySlotDestructionInfo>
mlir::computeDestructionInfo(DestructibleMemorySlot &slot) {
  MemorySlotDestructionInfo info;

  // Initialize the analysis with the immediate users of the slot.
  for (OpOperand &use : slot.slot.ptr.getUses()) {
    if (auto accessor =
            dyn_cast<DestructibleAccessorOpInterface>(use.getOwner())) {
      if (accessor.canRewire(slot, info.usedIndices)) {
        info.accessors.push_back(accessor);
        continue;
      }
    }

    SmallPtrSet<OpOperand *, 4> &blockingUses =
        getOrInsertDefault(info.userToBlockingUses, use.getOwner());
    blockingUses.insert(&use);
  }

  struct AccessCheckJob {
    DestructibleAccessorOpInterface accessor;
    DestructibleMemorySlot memorySlot;
  };

  DenseMap<PromotableMemOpInterface, MemorySlot> shouldBePromotable;
  SmallVector<AccessCheckJob> accessCheckWorklist;
  for (DestructibleAccessorOpInterface accessor : info.accessors)
    accessCheckWorklist.emplace_back<AccessCheckJob>({accessor, slot});

  while (!accessCheckWorklist.empty()) {
    AccessCheckJob job = accessCheckWorklist.pop_back_val();
    for (MaybeDestructibleSubElementMemorySlot &subslot :
         job.accessor.getSubElementMemorySlots(job.memorySlot)) {
      for (OpOperand &subslotUse : subslot.slot.ptr.getUses()) {
        bool shouldAbort =
            TypeSwitch<Operation *, bool>(subslotUse.getOwner())
                .Case<DestructibleAccessorOpInterface>([&](auto accessor) {
                  if (!subslot.destructibleInfo)
                    return true;

                  accessCheckWorklist.emplace_back<AccessCheckJob>(
                      {accessor,
                       DestructibleMemorySlot{
                           subslot.slot, subslot.destructibleInfo.value()}});
                  return false;
                })
                .Case<PromotableMemOpInterface>([&](auto memOp) {
                  Operation *memOpAsOp = memOp;
                  SmallPtrSet<OpOperand *, 4> &blockingUses =
                      getOrInsertDefault(info.userToBlockingUses, memOpAsOp);
                  blockingUses.insert(&subslotUse);

                  shouldBePromotable.insert({memOp, subslot.slot});
                  return false;
                })
                .Case<PromotableOpInterface>([&](auto promotableOp) {
                  Operation *promotableOpAsOp = promotableOp;
                  SmallPtrSet<OpOperand *, 4> &blockingUses =
                      getOrInsertDefault(info.userToBlockingUses,
                                         promotableOpAsOp);
                  blockingUses.insert(&subslotUse);
                  return false;
                })
                .Default([](auto) { return true; });

        if (shouldAbort)
          return {};
      }
    }
  }

  SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(slot.slot.ptr, &forwardSlice);
  for (Operation *user : forwardSlice) {
    // If the next operation has no blocking uses, everything is fine.
    if (!info.userToBlockingUses.contains(user))
      continue;

    // If the operation is a mem op, we just need to check it is promotable if
    // necessary.
    if (auto memOp = dyn_cast<PromotableMemOpInterface>(user)) {
      if (!shouldBePromotable.contains(memOp))
        continue;
      MemorySlot concernedSlot = shouldBePromotable.at(memOp);
      assert(0 && "todo");
    }

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
      assert(llvm::find(user->getResults(), blockingUse->get()) !=
             user->result_end());

      SmallPtrSetImpl<OpOperand *> &newUserBlockingUseSet =
          getOrInsertDefault(info.userToBlockingUses, blockingUse->getOwner());
      newUserBlockingUseSet.insert(blockingUse);
    }
  }

  return info;
}

void mlir::destructSlot(DestructibleMemorySlot &slot,
                        DestructibleAllocationOpInterface allocator,
                        OpBuilder &builder, MemorySlotDestructionInfo &info) {
  OpBuilder::InsertionGuard guard(builder);

  builder.setInsertionPointToStart(slot.slot.ptr.getParentBlock());
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
    if (auto promotable = dyn_cast<PromotableOpInterface>(toRewire)) {
      if (promotable.removeBlockingUses(info.userToBlockingUses[promotable],
                                        builder) == DeletionKind::Delete)
        toErase.push_back(promotable);
      continue;
    }

    auto accessor = cast<DestructibleAccessorOpInterface>(toRewire);
    if (accessor.rewire(slot, subslots) == DeletionKind::Delete)
      toErase.push_back(accessor);
  }

  for (Operation *toEraseOp : toErase)
    toEraseOp->erase();

  assert(slot.slot.ptr.use_empty() &&
         "at the end of destruction, the original slot "
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
      bool justDestructed = true;
      while (justDestructed) {
        justDestructed = false;

        for (Block &block : region) {
          for (Operation &op : block.getOperations()) {
            if (auto allocator =
                    llvm::dyn_cast<DestructibleAllocationOpInterface>(op)) {
              for (DestructibleMemorySlot slot :
                   allocator.getDestructibleSlots()) {
                if (auto info = computeDestructionInfo(slot)) {
                  destructSlot(slot, allocator, builder, info.value());
                  justDestructed = true;
                }
              }
            }
          }
        }

        changed |= justDestructed;
      }
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
