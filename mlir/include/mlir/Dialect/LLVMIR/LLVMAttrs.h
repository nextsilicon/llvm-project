//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMATTRS_H_
#define MLIR_DIALECT_LLVMIR_LLVMATTRS_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace LLVM {

/// This class represents the base attribute for all debug info attributes.
class DINodeAttr : public Attribute {
public:
  using Attribute::Attribute;

  // Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info scope.
class DIScopeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a local debug info
/// scope.
class DILocalScopeAttr : public DIScopeAttr {
public:
  using DIScopeAttr::DIScopeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info type.
class DITypeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

namespace detail {
class AttrIDGenAttrStorage;
} // namespace detail

class AttrIDGenAttr : public Attribute::AttrBase<AttrIDGenAttr, Attribute,
                                                 detail::AttrIDGenAttrStorage,
                                                 AttributeTrait::IsMutable> {
public:
  /// Inherit base constructors.
  using Base::Base;

  /// Returns a generator attribute that produces unique attribute identifiers.
  static AttrIDGenAttr get(Attribute attribute);

  /// Returns the next identifier.
  int64_t getNextID();
};

/// Returns an attribute of the given type using the given parameters and
/// replaces the initial builder argument with a unique identifier.
template <typename AttrTy, typename... Args>
AttrTy getDistinct(MLIRContext *context, Args &&...args) {
  auto genAttr = AttrIDGenAttr::get(AttrTy::get(context, 0, args...));
  return AttrTy::get(context, genAttr.getNextID(), std::forward<Args>(args)...);
}

// Inline the LLVM generated Linkage enum and utility.
// This is only necessary to isolate the "enum generated code" from the
// attribute definition itself.
// TODO: this shouldn't be needed after we unify the attribute generation, i.e.
// --gen-attr-* and --gen-attrdef-*.
using cconv::CConv;
using linkage::Linkage;
} // namespace LLVM
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMATTRS_H_
