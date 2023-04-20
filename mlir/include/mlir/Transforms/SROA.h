//===-- SROA.h - Scalar Replacement Of Aggregates ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SROA_H
#define MLIR_TRANSFORMS_SROA_H

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// Computes information for slot destruction leading to promotion. This will
/// compute whether destructing this slot and subsequent new subslots will
/// lead to only promatble slots being generated.
class MemorySlotDestructionAnalyzer {
public:
  MemorySlotDestructionAnalyzer(MemorySlot slot, DominanceInfo &dominance)
      : slot(slot), dominance(dominance) {}

private:
  LogicalResult computeDestructionTree();

  LogicalResult computeBlockingUses();

  MemorySlot slot;
  DominanceInfo &dominance;
};

class MemorySlotDestructor {
public:
  void destructSlot();
private:
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_SROA_H
