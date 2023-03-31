//===- LoopInfo.cpp - LoopInfo analysis for regions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopInfo.h"

using namespace mlir;

::mlir::Loop::Loop(mlir::Block *block)
    : llvm::LoopBase<mlir::Block, Loop>(block) {}

::mlir::LoopInfo::LoopInfo(
    const llvm::DominatorTreeBase<mlir::Block, false> &domTree) {
  analyze(domTree);
}
