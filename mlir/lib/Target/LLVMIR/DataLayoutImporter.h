//===- DataLayoutImporter.h - LLVM to MLIR data layout conversion -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between the LLVMIR data layout and the
// corresponding MLIR representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_DATALAYOUTIMPORT_H_
#define MLIR_LIB_TARGET_LLVMIR_DATALAYOUTIMPORT_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DataLayout;
} // namespace llvm

namespace mlir {
class DataLayoutSpecInterface;
class FloatType;
class Operation;

namespace LLVM {
class LLVMFuncOp;

namespace detail {

class DataLayoutImporter {
public:
  DataLayoutImporter(MLIRContext *context) : context(context) {}

  /// Translates the LLVM `dataLayout` to an MLIR data layout specification. The
  /// method only translates integer, float, pointer, alloca memory space, and
  /// endianess attributes and relies on the default specification found in the
  /// language reference (https://llvm.org/docs/LangRef.html#data-layout) for
  /// types and attributes that are not found in the provided layout.
  DataLayoutSpecInterface
  translateDataLayout(const llvm::DataLayout &dataLayout);

  /// Returns a supported MLIR floating point type of the given bit width or
  /// null if the bit width is not supported.
  FloatType getFloatType(unsigned width) const;

private:
  /// Tries to parse an identifier of letters only and removes the identifier
  /// from the beginning of the string.
  FailureOr<StringRef> tryToParseIdentifier(StringRef &token) const;

  /// Tries to parse an integer parameter and removes the integer from the
  /// beginning of the string.
  FailureOr<unsigned> tryToParseInt(StringRef &token) const;

  /// Tries to parse an integer parameter array.
  FailureOr<SmallVector<unsigned>> tryToParseIntList(StringRef token) const;

  /// Tries to parse the parameters of a type alignment entry.
  FailureOr<DenseIntElementsAttr> tryToParseAlignment(StringRef token) const;

  /// Tries to parse the parameters of a pointer alignment entry.
  FailureOr<DenseIntElementsAttr>
  tryToParsePointerAlignment(StringRef token) const;

  /// Adds a type alignment entry if there is none yet.
  LogicalResult tryToEmplaceAlignmentEntry(Type type, StringRef token);

  /// Adds a pointer alignment entry if there is none yet.
  LogicalResult tryToEmplacePointerAlignmentEntry(Type type, StringRef token);

  /// Adds an endianess entry if there is none yet.
  LogicalResult tryToEmplaceEndianessEntry(StringRef endianess,
                                           StringRef token);

  /// Adds an alloca address space entry if there is none yet.
  LogicalResult tryToEmplaceAllocaAddrSpaceEntry(StringRef token);

  DenseMap<TypeAttr, DataLayoutEntryInterface> typeEntries;
  DenseMap<StringAttr, DataLayoutEntryInterface> idEntries;
  MLIRContext *context;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_DATALAYOUTIMPORT_H_
