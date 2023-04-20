//===-- SROA.cpp - Scalar Replacement Of Aggregates -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/SROA.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_SROA
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

LogicalResult MemorySlotDestructionAnalyzer::computeDestructionTree() {
  // TODO: document structure

  SmallVector<MemorySlot> dfsWorklist{slot};
  while (!dfsWorklist.empty()) {
    MemorySlot currentSlot = dfsWorklist.pop_back_val();
    bool mustBeLeaf = false;

    for (Operation *user : currentSlot.ptr.getUsers()) {
      if (auto memOp = llvm::dyn_cast<PromotableMemOpInterface>(user))
        mustBeLeaf |=
            memOp.getStored(currentSlot) || memOp.loadsFrom(currentSlot);

      
    }
  }

  return success();
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

      // Promoting a slot can allow for further promotion of other slots,
      // promotion is tried until no promotion succeeds.
      while (true) {
        DominanceInfo &dominance = getAnalysis<DominanceInfo>();

        for (Block &block : region) {
          for (Operation &op : block.getOperations()) {
            if (auto allocator =
                    llvm::dyn_cast<DestructibleAllocationOpInterface>(op)) {
              allocator.getDestructibleSlots();
            }
          }
        }

        changed = true;
        getAnalysisManager().invalidate({});
      }
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
