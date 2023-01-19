// RUN: mlir-opt %s | mlir-opt | FileCheck %s

#file = #llvm.di_file<"debuginfo.mlir" in "/test/">
#cu = #llvm.di_compile_unit<
  sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR",
  isOptimized = true, emissionKind = Full
>
#void = #llvm.di_void_result_type
#int0 = #llvm.di_basic_type<
  tag = DW_TAG_base_type, name = "int0"
>
#derived = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #int0>


#comp0 = #llvm.di_composite_type_mut<id = 0, name = "test0", elements = #llvm.di_composite_type_mut<id = 0, name = "test1">>

#comp1 = #llvm.di_composite_type_mut<id = 0, name = "test1", elements = #comp0>

//#comp2 = #llvm.di_composite_type_mut<id = 0, name = "test1", elements = #int0>

// #ptr = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #int0>

// #comp0 = #llvm.di_composite_type<
//   tag = DW_TAG_array_type, name = "array0",
//   line = 10, sizeInBits = 128, alignInBits = 32,
//   elements = #int0
// >

#spType0 = #llvm.di_subroutine_type<
  callingConvention = DW_CC_normal, types = #void, #comp0, #comp1 //, #comp2
>

#sp0 = #llvm.di_subprogram<
  // Omit the optional linkageName parameter.
  compileUnit = #cu, scope = #file, name = "value",
  file = #file, subprogramFlags = "Definition", type = #spType0
>

llvm.func @value() {
  llvm.return
} loc(fused<#sp0>["foo.mlir":1:1])
