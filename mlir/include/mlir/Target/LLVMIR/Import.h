//===- Import.h - LLVM IR To MLIR translation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry point for the LLVM IR to MLIR conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_IMPORT_H
#define MLIR_TARGET_LLVMIR_IMPORT_H

#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

// Forward-declare LLVM classes.
namespace llvm {
class DataLayout;
class Module;
} // namespace llvm

namespace mlir {

class DataLayoutSpecInterface;
class MLIRContext;
class ModuleOp;

/// Translates the LLVM module into an MLIR module living in the given context.
/// The translation supports operations from any dialect that has a registered
/// implementation of the LLVMImportDialectInterface. It returns nullptr if the
/// translation fails and reports errors using the error handler registered with
/// the MLIR context.
OwningOpRef<ModuleOp>
translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                        MLIRContext *context);

/// Translates the given LLVM IR data layout to a dialect data layout spec
/// attribute. Only integer, float, pointer, alloca memory space, and endianess
/// attributes are currently supported.
FailureOr<DataLayoutSpecInterface>
translateDataLayout(const llvm::DataLayout &dataLayout, MLIRContext *context,
                    function_ref<InFlightDiagnostic(StringRef)> emitErrorFn,
                    function_ref<void(StringRef)> emitWarningFn);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_IMPORT_H
