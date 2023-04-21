//===-- SROA.h - Scalar Replacement Of Aggregates ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SROA_H
#define MLIR_TRANSFORMS_SROA_H

#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

struct MemorySlotDestructionInfo {
  DenseMap<Operation *, SmallPtrSet<OpOperand *, 4>> userToBlockingUses;
  SmallVector<DestructibleAccessorOpInterface> accessors;
};

/// Computes information for slot destruction leading to promotion. This will
/// compute whether this slot can be destructed. Returns nothing if the slot
/// cannot be destructed.
Optional<MemorySlotDestructionInfo>
computeDestructionInfo(DestructibleMemorySlot &slot);

void destructSlot(DestructibleMemorySlot &slot,
                  DestructibleAllocationOpInterface allocator,
                  OpBuilder &builder, MemorySlotDestructionInfo &info);

} // namespace mlir

#endif // MLIR_TRANSFORMS_SROA_H
