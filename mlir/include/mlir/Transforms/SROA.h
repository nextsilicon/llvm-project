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

struct MemorySlotDestructuringInfo {
  SmallPtrSet<Attribute, 8> usedIndices;
  DenseMap<Operation *, SmallPtrSet<OpOperand *, 4>> userToBlockingUses;
  SmallVector<DestructurableAccessorOpInterface> accessors;
};

/// Computes information for slot destructuring. This will compute whether this
/// slot can be destructured and data to perform the destructuring. Returns
/// nothing if the slot cannot be destructured.
Optional<MemorySlotDestructuringInfo>
computeDestructuringInfo(DestructurableMemorySlot &slot);

/// Performs the destructuring of a destructible slot given associated
/// destructuring information. The provided slot will be destructured in
/// subslots as specified by its allocator.
void destructureSlot(DestructurableMemorySlot &slot,
                     DestructurableAllocationOpInterface allocator,
                     OpBuilder &builder, MemorySlotDestructuringInfo &info);

} // namespace mlir

#endif // MLIR_TRANSFORMS_SROA_H
