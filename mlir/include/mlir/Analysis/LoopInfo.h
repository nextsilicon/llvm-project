//===- LoopInfo.h - LoopInfo analysis for regions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopInfo analysis for MLIR. The LoopInfo is used to
// identify natural loops and determine the loop depth of various nodes of a
// CFG.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_LOOPINFO_H
#define MLIR_ANALYSIS_LOOPINFO_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopInfoImpl.h"

namespace llvm {

// llvm::GraphTrait specializations that are required for the LLVM's generic
// LoopInfo. Note that the const_casts are required because MLIR has no constant
// accessors on IR constructs.

template <>
struct GraphTraits<const mlir::Block *> {
  using ChildIteratorType = mlir::Block::succ_iterator;
  using Node = const mlir::Block;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef node) { return node; }

  static ChildIteratorType child_begin(NodeRef node) {
    return const_cast<mlir::Block *>(node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return const_cast<mlir::Block *>(node)->succ_end();
  }
};

template <>
struct GraphTraits<Inverse<const mlir::Block *>> {
  using ChildIteratorType = mlir::Block::pred_iterator;
  using Node = const mlir::Block;
  using NodeRef = Node *;

  static NodeRef getEntryNode(Inverse<NodeRef> inverseGraph) {
    return inverseGraph.Graph;
  }

  static ChildIteratorType child_begin(NodeRef node) {
    return const_cast<mlir::Block *>(node)->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return const_cast<mlir::Block *>(node)->pred_end();
  }
};
} // namespace llvm

namespace mlir {
class Loop : public llvm::LoopBase<mlir::Block, mlir::Loop> {
private:
  explicit Loop(mlir::Block *block);

  friend class llvm::LoopBase<mlir::Block, Loop>;
  friend class llvm::LoopInfoBase<mlir::Block, Loop>;
};

/// Instantiate a variant of LLVM LoopInfo that works on mlir::Block.
class LoopInfo : public llvm::LoopInfoBase<mlir::Block, mlir::Loop> {
public:
  LoopInfo(const llvm::DominatorTreeBase<mlir::Block, false> &domTree);
};
} // namespace mlir

#endif // MLIR_ANALYSIS_LOOPINFO_H
