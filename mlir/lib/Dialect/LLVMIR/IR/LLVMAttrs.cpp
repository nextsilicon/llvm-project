//===- LLVMAttrs.cpp - LLVM Attributes registration -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attribute details for the LLVM IR dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// LLVMDialect registration
//===----------------------------------------------------------------------===//

void LLVMDialect::registerAttributes() {
  addAttributes<DICompositeTypeMutAttr>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DINodeAttr
//===----------------------------------------------------------------------===//

bool DINodeAttr::classof(Attribute attr) {
  return llvm::isa<DIVoidResultTypeAttr, DIBasicTypeAttr, DICompileUnitAttr,
                   DICompositeTypeAttr, DIDerivedTypeAttr, DIFileAttr,
                   DILexicalBlockAttr, DILexicalBlockFileAttr,
                   DILocalVariableAttr, DISubprogramAttr, DISubrangeAttr,
                   DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DICompositeTypeAttr, DIFileAttr,
                   DILexicalBlockFileAttr, DILocalScopeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DILocalScopeAttr
//===----------------------------------------------------------------------===//

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DISubprogramAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DITypeAttr
//===----------------------------------------------------------------------===//

bool DITypeAttr::classof(Attribute attr) {
  return llvm::isa<DIVoidResultTypeAttr, DIBasicTypeAttr, DICompositeTypeAttr,
                   DICompositeTypeMutAttr, DIDerivedTypeAttr,
                   DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DISubroutineTypeAttr
//===----------------------------------------------------------------------===//

LogicalResult
DISubroutineTypeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             unsigned int callingConventions,
                             ArrayRef<DITypeAttr> types) {
  ArrayRef<DITypeAttr> argumentTypes =
      types.empty() ? types : types.drop_front();
  if (llvm::any_of(argumentTypes, [](DITypeAttr type) {
        return type.isa<DIVoidResultTypeAttr>();
      }))
    return emitError() << "expected subroutine to have non-void argument types";
  return success();
}

//===----------------------------------------------------------------------===//
// DICompositeTypeMutAttr
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {
namespace detail {
// TODO move to LLVMAttersDetail.h?
class DICompositeTypeMutAttrStorage : public AttributeStorage {
public:
  using KeyTy = std::tuple<int64_t, StringAttr, DITypeAttr>;

  DICompositeTypeMutAttrStorage(int64_t id, StringAttr name,
                                DITypeAttr baseType)
      : id(id), name(name), baseType(baseType) {}

  KeyTy getAsKey() const { return KeyTy(id, name, baseType); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key));
  }

  static DICompositeTypeMutAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    int64_t id = std::get<0>(key);
    StringAttr name = std::get<1>(key);
    DITypeAttr baseType = std::get<2>(key);
    return new (allocator.allocate<DICompositeTypeMutAttrStorage>())
        DICompositeTypeMutAttrStorage(id, name, baseType);
  }

  LogicalResult mutate(AttributeStorageAllocator &allocator,
                       ArrayRef<DITypeAttr> elements) {
    if (isMutated)
      return failure();

    ArrayRef<DITypeAttr> allocated = allocator.copyInto(elements);
    elementsPtr = allocated.data();
    elementsSize = allocated.size();
    isMutated = true;
    return success();
  }

  bool operator==(const KeyTy &other) const { return getAsKey() == other; }

  /// Returns the list of types contained in an identified struct.
  ArrayRef<DITypeAttr> getElements() const {
    return ArrayRef<DITypeAttr>(elementsPtr, elementsSize);
  }

  // TODO make stuff blow private
  int64_t id;
  StringAttr name;
  DITypeAttr baseType;

  bool isMutated = false;
  const DITypeAttr *elementsPtr = nullptr;
  size_t elementsSize = 0;
};
} // namespace detail
} // namespace LLVM
} // namespace mlir

DICompositeTypeMutAttr DICompositeTypeMutAttr::get(MLIRContext *context,
                                                   int64_t id, StringAttr name,
                                                   DITypeAttr baseType) {
  return Base::get(context, id, name, baseType);
}

int64_t DICompositeTypeMutAttr::getID() const { return getImpl()->id; }

StringAttr DICompositeTypeMutAttr::getName() const { return getImpl()->name; }

DITypeAttr DICompositeTypeMutAttr::getBaseType() const {
  return getImpl()->baseType;
}

LogicalResult
DICompositeTypeMutAttr::setElements(ArrayRef<DITypeAttr> elements) {
  return Base::mutate(elements);
}

ArrayRef<DITypeAttr> DICompositeTypeMutAttr::getElements() const {
  return getImpl()->getElements();
}

Attribute DICompositeTypeMutAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // A helper function to parse a composite type parameter.
  auto parseParameter =
      [&](StringRef name, StringRef type, bool &seen,
          function_ref<ParseResult()> parseFn) -> ParseResult {
    if (seen) {
      return parser.emitError(parser.getCurrentLocation())
             << "struct has duplicate parameter '" << name << "'";
    }
    if (failed(parseFn())) {
      return parser.emitError(parser.getCurrentLocation())
             << "failed to parse LLVM_DICompositeTypeAttr parameter '" << name
             << "' which is to be a '" << type << "'";
    }
    seen = true;
    return success();
  };

  std::pair<int64_t, bool> id = {0, false};
  std::pair<StringAttr, bool> name = {nullptr, false};
  std::pair<DITypeAttr, bool> baseType = {nullptr, false};
  std::pair<SmallVector<DITypeAttr>, bool> elements = {{}, false};
  do {
    std::string keyword;
    if (failed(parser.parseKeywordOrString(&keyword))) {
      parser.emitError(parser.getCurrentLocation())
          << "expected a parameter name in struct";
      return {};
    }
    if (parser.parseEqual())
      return {};

    if (keyword == "id") {
      if (failed(parseParameter(keyword, "int64_t", id.second, [&]() {
            return parser.parseInteger(id.first);
          })))
        return {};
    } else if (keyword == "name") {
      if (failed(parseParameter(keyword, "StringAttr", name.second, [&]() {
            return parser.parseAttribute(name.first);
          })))
        return {};
    } else if (keyword == "base_type") {
      if (failed(parseParameter(keyword, "DITypeAttr", baseType.second, [&]() {
            return parser.parseAttribute(baseType.first);
          })))
        return {};
    } else if (keyword == "elements") {
      if (failed(parseParameter(
              keyword, "::llvm::ArrayRef<DINodeAttr>", elements.second, [&]() {
                return parser.parseCommaSeparatedList([&]() {
                  return parser.parseAttribute(elements.first.emplace_back());
                });
              })))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation())
          << "expected a parameter name in struct";
      return {};
    }
  } while (succeeded(parser.parseOptionalComma()));

  if (!id.second) {
    parser.emitError(parser.getCurrentLocation())
        << "struct is missing required parameter 'id'";
    return {};
  }
  if (!name.second) {
    parser.emitError(parser.getCurrentLocation())
        << "struct is missing required parameter 'name'";
    return {};
  }

  if (parser.parseGreater())
    return {};

  DICompositeTypeMutAttr attr =
      get(parser.getContext(), id.first, name.first, baseType.first);
  if (elements.second)
    if (failed(attr.setElements(elements.first))) {
      parser.emitError(parser.getCurrentLocation())
          << "cannot mutate 'elements' twice";
      return {};
    }
  return attr;
}

void DICompositeTypeMutAttr::print(AsmPrinter &os) const {

  thread_local SetVector<TypeID> knownAttributes;
  // TODO add stack size assertion?

  os << DICompositeTypeMutAttr::getMnemonic() << "<";
  os << "id = " << getID();
  os << ", name = ";
  os.printStrippedAttrOrType(getName());
  if (getBaseType()) {
    os << ", base_type = ";
    os.printStrippedAttrOrType(getBaseType());
  }
  if (getImpl()->isMutated && knownAttributes.count(getTypeID()) == 0) {
    knownAttributes.insert(getTypeID());

    os << ", elements = ";
    llvm::interleaveComma(getElements(), os, [&](auto typeAttr) {
      os.printStrippedAttrOrType(typeAttr);
    });

    knownAttributes.pop_back();
  }
  os << ">";
}

//===----------------------------------------------------------------------===//
// LoopOptionsAttrBuilder
//===----------------------------------------------------------------------===//

LoopOptionsAttrBuilder::LoopOptionsAttrBuilder(LoopOptionsAttr attr)
    : options(attr.getOptions().begin(), attr.getOptions().end()) {}

template <typename T>
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setOption(LoopOptionCase tag, std::optional<T> value) {
  auto option = llvm::find_if(
      options, [tag](auto option) { return option.first == tag; });
  if (option != options.end()) {
    if (value)
      option->second = *value;
    else
      options.erase(option);
  } else {
    options.push_back(LoopOptionsAttr::OptionValuePair(tag, *value));
  }
  return *this;
}

LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisableLICM(std::optional<bool> value) {
  return setOption(LoopOptionCase::disable_licm, value);
}

/// Set the `interleave_count` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setInterleaveCount(std::optional<uint64_t> count) {
  return setOption(LoopOptionCase::interleave_count, count);
}

/// Set the `disable_unroll` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisableUnroll(std::optional<bool> value) {
  return setOption(LoopOptionCase::disable_unroll, value);
}

/// Set the `disable_pipeline` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisablePipeline(std::optional<bool> value) {
  return setOption(LoopOptionCase::disable_pipeline, value);
}

/// Set the `pipeline_initiation_interval` option to the provided value.
/// If no value is provided the option is deleted.
LoopOptionsAttrBuilder &LoopOptionsAttrBuilder::setPipelineInitiationInterval(
    std::optional<uint64_t> count) {
  return setOption(LoopOptionCase::pipeline_initiation_interval, count);
}

//===----------------------------------------------------------------------===//
// LoopOptionsAttr
//===----------------------------------------------------------------------===//

template <typename T>
static std::optional<T>
getOption(ArrayRef<std::pair<LoopOptionCase, int64_t>> options,
          LoopOptionCase option) {
  auto it =
      lower_bound(options, option, [](auto optionPair, LoopOptionCase option) {
        return optionPair.first < option;
      });
  if (it == options.end())
    return {};
  return static_cast<T>(it->second);
}

std::optional<bool> LoopOptionsAttr::disableUnroll() {
  return getOption<bool>(getOptions(), LoopOptionCase::disable_unroll);
}

std::optional<bool> LoopOptionsAttr::disableLICM() {
  return getOption<bool>(getOptions(), LoopOptionCase::disable_licm);
}

std::optional<int64_t> LoopOptionsAttr::interleaveCount() {
  return getOption<int64_t>(getOptions(), LoopOptionCase::interleave_count);
}

/// Build the LoopOptions Attribute from a sorted array of individual options.
LoopOptionsAttr LoopOptionsAttr::get(
    MLIRContext *context,
    ArrayRef<std::pair<LoopOptionCase, int64_t>> sortedOptions) {
  assert(llvm::is_sorted(sortedOptions, llvm::less_first()) &&
         "LoopOptionsAttr ctor expects a sorted options array");
  return Base::get(context, sortedOptions);
}

/// Build the LoopOptions Attribute from a sorted array of individual options.
LoopOptionsAttr LoopOptionsAttr::get(MLIRContext *context,
                                     LoopOptionsAttrBuilder &optionBuilders) {
  llvm::sort(optionBuilders.options, llvm::less_first());
  return Base::get(context, optionBuilders.options);
}

void LoopOptionsAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getOptions(), printer, [&](auto option) {
    printer << stringifyEnum(option.first) << " = ";
    switch (option.first) {
    case LoopOptionCase::disable_licm:
    case LoopOptionCase::disable_unroll:
    case LoopOptionCase::disable_pipeline:
      printer << (option.second ? "true" : "false");
      break;
    case LoopOptionCase::interleave_count:
    case LoopOptionCase::pipeline_initiation_interval:
      printer << option.second;
      break;
    }
  });
  printer << ">";
}

Attribute LoopOptionsAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  SmallVector<std::pair<LoopOptionCase, int64_t>> options;
  llvm::SmallDenseSet<LoopOptionCase> seenOptions;
  auto parseLoopOptions = [&]() -> ParseResult {
    StringRef optionName;
    if (parser.parseKeyword(&optionName))
      return failure();

    auto option = symbolizeLoopOptionCase(optionName);
    if (!option)
      return parser.emitError(parser.getNameLoc(), "unknown loop option: ")
             << optionName;
    if (!seenOptions.insert(*option).second)
      return parser.emitError(parser.getNameLoc(), "loop option present twice");
    if (failed(parser.parseEqual()))
      return failure();

    int64_t value;
    switch (*option) {
    case LoopOptionCase::disable_licm:
    case LoopOptionCase::disable_unroll:
    case LoopOptionCase::disable_pipeline:
      if (succeeded(parser.parseOptionalKeyword("true")))
        value = 1;
      else if (succeeded(parser.parseOptionalKeyword("false")))
        value = 0;
      else {
        return parser.emitError(parser.getNameLoc(),
                                "expected boolean value 'true' or 'false'");
      }
      break;
    case LoopOptionCase::interleave_count:
    case LoopOptionCase::pipeline_initiation_interval:
      if (failed(parser.parseInteger(value)))
        return parser.emitError(parser.getNameLoc(), "expected integer value");
      break;
    }
    options.push_back(std::make_pair(*option, value));
    return success();
  };
  if (parser.parseCommaSeparatedList(parseLoopOptions) || parser.parseGreater())
    return {};

  llvm::sort(options, llvm::less_first());
  return get(parser.getContext(), options);
}

//===----------------------------------------------------------------------===//
// MemoryEffectsAttr
//===----------------------------------------------------------------------===//

MemoryEffectsAttr MemoryEffectsAttr::get(MLIRContext *context,
                                         ArrayRef<ModRefInfo> memInfoArgs) {
  if (memInfoArgs.empty())
    return MemoryEffectsAttr::get(context, ModRefInfo::ModRef,
                                  ModRefInfo::ModRef, ModRefInfo::ModRef);
  if (memInfoArgs.size() == 3)
    return MemoryEffectsAttr::get(context, memInfoArgs[0], memInfoArgs[1],
                                  memInfoArgs[2]);
  return {};
}

bool MemoryEffectsAttr::isReadWrite() {
  if (this->getArgMem() != ModRefInfo::ModRef)
    return false;
  if (this->getInaccessibleMem() != ModRefInfo::ModRef)
    return false;
  if (this->getOther() != ModRefInfo::ModRef)
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// LLVMDialect
//===----------------------------------------------------------------------===//

Attribute LLVMDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef mnemonic;
  Attribute attr;
  OptionalParseResult result =
      generatedAttributeParser(parser, &mnemonic, type, attr);
  if (result.has_value())
    return attr;

  if (mnemonic == DICompositeTypeMutAttr::getMnemonic())
    return DICompositeTypeMutAttr::parse(parser, type);

  llvm_unreachable("unhandled LLVM attribute kind");
}

void LLVMDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  if (succeeded(generatedAttributePrinter(attr, os)))
    return;

  if (auto composite = dyn_cast<DICompositeTypeMutAttr>(attr))
    composite.print(os);
  else
    llvm_unreachable("unhandled LLVM attribute kind");
}
