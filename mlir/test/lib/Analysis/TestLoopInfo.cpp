//===- TestLoopInfo.cpp - Test loop info analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing the LoopInfo analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopInfo.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// A testing pass that applies the LoopInfo analysis on a region and prints the
/// information it collected to llvm::errs().
struct TestLoopInfo
    : public PassWrapper<TestLoopInfo, InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopInfo)

  StringRef getArgument() const final { return "test-loop-info"; }
  StringRef getDescription() const final {
    return "Test the loop info analysis.";
  }

  void runOnOperation() override;
};
} // namespace

void TestLoopInfo::runOnOperation() {
  auto func = getOperation();
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  Region &region = func.getFunctionBody();

  // Prints the label of the test.
  llvm::errs() << "Testing : " << func.getNameAttr() << "\n";

  // Print all the block identifiers first such that the tests can match them.
  llvm::errs() << "Blocks : ";
  region.front().printAsOperand(llvm::errs());
  for (auto &block : region.getBlocks()) {
    llvm::errs() << ", ";
    block.printAsOperand(llvm::errs());
  }
  llvm::errs() << "\n";

  if (region.getBlocks().size() == 1) {
    llvm::errs() << "no loops\n";
    return;
  }

  llvm::DominatorTreeBase<mlir::Block, false> &domTree =
      domInfo.getDomTree(&region);
  mlir::LoopInfo loopInfo(domTree);

  if (loopInfo.getTopLevelLoops().empty())
    llvm::errs() << "no loops\n";
  else
    loopInfo.print(llvm::errs());
}

namespace mlir {
namespace test {
void registerTestLoopInfoPass() { PassRegistration<TestLoopInfo>(); }
} // namespace test
} // namespace mlir
