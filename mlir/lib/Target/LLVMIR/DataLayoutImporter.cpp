//===- DataLayoutImporter.cpp - LLVM to MLIR data layout conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DataLayoutImporter.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DataLayout.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

DataLayoutSpecInterface
DataLayoutImporter::translateDataLayout(const llvm::DataLayout &dataLayout) {
  // Transform the data layout to its string representation and append the
  // default data layout string specified in the language reference
  // (https://llvm.org/docs/LangRef.html#data-layout). The translation then
  // parses the string and ignores the default value if a specific kind occurs
  // in both strings. Additionally, the following default values exist:
  // - non-default address space pointer specifications default to the default
  //   address space pointer specification
  // - the alloca address space defaults to the default address space.
  std::string layoutStr = dataLayout.getStringRepresentation();
  if (!layoutStr.empty())
    layoutStr += "-";
  layoutStr += "e-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-"
               "f16:16:16-f64:64:64-f128:128:128";
  StringRef layout(layoutStr);

  // Split the data layout string into specifications separated by a dash.
  SmallVector<StringRef> tokens;
  layout.split(tokens, '-');

  for (auto token : tokens) {
    FailureOr<StringRef> identifier = tryToParseIdentifier(token);
    if (failed(identifier))
      return nullptr;

    // Parse the endianness.
    if (*identifier == "e") {
      if (failed(tryToEmplaceEndianessEntry(
              DLTIDialect::kDataLayoutEndiannessLittle, token)))
        return nullptr;
      continue;
    }
    if (*identifier == "E") {
      if (failed(tryToEmplaceEndianessEntry(
              DLTIDialect::kDataLayoutEndiannessBig, token)))
        return nullptr;
      continue;
    }
    // Parse alloca address space.
    if (*identifier == "A") {
      if (failed(tryToEmplaceAllocaAddrSpaceEntry(token)))
        return nullptr;
      continue;
    }
    // Parse integer alignment specifications.
    if (*identifier == "i") {
      FailureOr<unsigned> width = tryToParseInt(token);
      if (failed(width))
        return nullptr;

      Type type = IntegerType::get(context, *width);
      if (failed(tryToEmplaceAlignmentEntry(type, token)))
        return nullptr;
      continue;
    }
    // Parse float alignment specifications.
    if (*identifier == "f") {
      FailureOr<unsigned> width = tryToParseInt(token);
      if (failed(width))
        return nullptr;

      Type type = getFloatType(*width);
      if (failed(tryToEmplaceAlignmentEntry(type, token)))
        return nullptr;
      continue;
    }
    // Parse pointer alignment specifications.
    if (*identifier == "p") {
      FailureOr<unsigned> space =
          token.starts_with(":") ? 0 : tryToParseInt(token);
      if (failed(space))
        return nullptr;

      Type type = LLVMPointerType::get(context, *space);
      if (failed(tryToEmplacePointerAlignmentEntry(type, token)))
        return nullptr;
      continue;
    }
  }

  return nullptr;
}

FloatType DataLayoutImporter::getFloatType(unsigned width) const {
  switch (width) {
  case 16:
    return FloatType::getF16(context);
  case 32:
    return FloatType::getF32(context);
  case 64:
    return FloatType::getF64(context);
  case 80:
    return FloatType::getF80(context);
  case 128:
    return FloatType::getF128(context);
  default:
    return nullptr;
  }
}

FailureOr<StringRef>
DataLayoutImporter::tryToParseIdentifier(StringRef &token) const {
  if (token.empty())
    return failure();

  StringRef id = token.take_while([](char c) { return llvm::isAlpha(c); });
  if (id.empty())
    return failure();

  token.consume_front(id);
  return id;
}

FailureOr<unsigned> DataLayoutImporter::tryToParseInt(StringRef &token) const {
  unsigned parameter = 0;
  if (token.consumeInteger(/*Radix=*/10, parameter))
    return failure();
  return parameter;
}

FailureOr<SmallVector<unsigned>>
DataLayoutImporter::tryToParseIntList(StringRef token) const {
  SmallVector<StringRef> tokens;
  token.consume_front(":");
  token.split(tokens, ':');

  // Parse an integer list.
  SmallVector<unsigned> results(tokens.size());
  for (auto [result, token] : llvm::zip(results, tokens))
    if (token.getAsInteger(/*Radix=*/10, result))
      return failure();
  return results;
}

FailureOr<DenseIntElementsAttr>
DataLayoutImporter::tryToParseAlignment(StringRef token) const {
  FailureOr<SmallVector<unsigned>> alignment = tryToParseIntList(token);
  if (failed(alignment))
    return failure();
  if (alignment->empty() || alignment->size() > 2)
    return failure();

  // Set the preferred alignment to the minimal alignment if not available.
  unsigned minimal = (*alignment)[0];
  unsigned preferred = alignment->size() == 1 ? minimal : (*alignment)[1];
  return DenseIntElementsAttr::get(
      VectorType::get({2}, IntegerType::get(context, 32)),
      {minimal, preferred});
}

FailureOr<DenseIntElementsAttr>
DataLayoutImporter::tryToParsePointerAlignment(StringRef token) const {
  FailureOr<SmallVector<unsigned>> alignment = tryToParseIntList(token);
  if (failed(alignment))
    return failure();
  if (alignment->size() < 2 || alignment->size() > 4)
    return failure();

  // Set the idx to the size and the preferred alignment to the minimal
  // alignment if not available.
  unsigned size = (*alignment)[0];
  unsigned minimal = (*alignment)[1];
  unsigned preferred = alignment->size() < 3 ? minimal : (*alignment)[2];
  unsigned idx = alignment->size() < 4 ? size : (*alignment)[3];
  return DenseIntElementsAttr::get(
      VectorType::get({4}, IntegerType::get(context, 32)),
      {size, minimal, preferred, idx});
}

LogicalResult DataLayoutImporter::tryToEmplaceAlignmentEntry(Type type,
                                                             StringRef token) {
  auto typeAttr = TypeAttr::get(type);
  if (typeEntries.count(typeAttr))
    return success();

  FailureOr<DenseIntElementsAttr> params = tryToParseAlignment(token);
  if (failed(params))
    return failure();

  typeEntries.try_emplace(typeAttr, DataLayoutEntryAttr::get(type, *params));
  return success();
}

LogicalResult
DataLayoutImporter::tryToEmplacePointerAlignmentEntry(Type type,
                                                      StringRef token) {
  auto typeAttr = TypeAttr::get(type);
  if (typeEntries.count(typeAttr))
    return success();

  FailureOr<DenseIntElementsAttr> params = tryToParsePointerAlignment(token);
  if (failed(params))
    return failure();

  typeEntries.try_emplace(typeAttr, DataLayoutEntryAttr::get(type, *params));
  return success();
}

LogicalResult
DataLayoutImporter::tryToEmplaceEndianessEntry(StringRef endianess,
                                               StringRef token) {
  auto idAttr = StringAttr::get(context, DLTIDialect::kDataLayoutEndiannessKey);
  if (idEntries.count(idAttr))
    return success();

  if (!token.empty())
    return failure();

  idEntries.try_emplace(idAttr, StringAttr::get(context, endianess));
  return success();
}

LogicalResult
DataLayoutImporter::tryToEmplaceAllocaAddrSpaceEntry(StringRef token) {
  auto idAttr =
      StringAttr::get(context, DLTIDialect::kDataLayoutAllocaMemorySpaceKey);
  if (idEntries.count(idAttr))
    return success();

  FailureOr<unsigned> space = tryToParseInt(token);
  if (failed(space))
    return failure();

  // Only store the address space if it has a non-default value.
  if (*space == 0)
    return success();
  OpBuilder builder(context);
  idEntries.try_emplace(idAttr, builder.getUI32IntegerAttr(*space));
  return success();
}
