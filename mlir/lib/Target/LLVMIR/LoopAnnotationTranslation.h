//===- LoopAnnotationTranslation.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR loop annotations and
// the corresponding LLVMIR metadata representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONTRANSLATION_H_
#define MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONTRANSLATION_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

namespace mlir {
namespace LLVM {
namespace detail {
/// A helper class that converts a LoopAnnotationAttr into a corresponding
/// llvm::MDNode.
class LoopAnnotationConverter {
public:
  LoopAnnotationConverter(LoopAnnotationAttr attr, llvm::LLVMContext &ctx,
                          LLVM::ModuleTranslation &moduleTranslation,
                          Operation *opInst)
      : attr(attr), ctx(ctx), moduleTranslation(moduleTranslation),
        opInst(opInst) {}
  /// Converts this struct's loop annotation into a corresponding LLVMIR
  /// metadata representation.
  llvm::MDNode *convert();

private:
  /// Functions that convert attributes to metadata nodes and add them to the
  /// `mdNodes` vector.
  void addUnitNode(StringRef name);
  void addUnitNode(StringRef name, BoolAttr attr);
  void convertBoolNode(StringRef name, BoolAttr attr, bool negated = false);
  void convertI32Node(StringRef name, IntegerAttr attr);
  void convertFollowupNode(StringRef name, LoopAnnotationAttr attr);

  /// Conversion functions for each of the sub-attributes of the loop
  /// annotation.
  void convertLoopOptions(LoopVectorizeAttr options);
  void convertLoopOptions(LoopInterleaveAttr options);
  void convertLoopOptions(LoopUnrollAttr options);
  void convertLoopOptions(LoopUnrollAndJamAttr options);
  void convertLoopOptions(LoopLICMAttr options);
  void convertLoopOptions(LoopDistributeAttr options);
  void convertLoopOptions(LoopPipelineAttr options);

  LoopAnnotationAttr attr;
  llvm::LLVMContext &ctx;
  LLVM::ModuleTranslation &moduleTranslation;
  Operation *opInst;
  llvm::SmallVector<llvm::Metadata *> mdNodes;
};
} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONTRANSLATION_H_
