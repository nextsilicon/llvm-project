; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK-LABEL:  llvm.func @memset_test
define void @memset_test(i32 %0, ptr noundef nonnull align 8 %1, i8 %2) {
  ; CHECK: "llvm.intr.memset"{{.*}} {llvm.arg_attrs = [{llvm.align = 8 : i64, llvm.nonnull, llvm.noundef}, {llvm.noundef}, {}]}
  call void @llvm.memset.p0.i32(ptr noundef nonnull align 8 %1, i8 noundef %2, i32 %0, i1 false)
  ret void
}

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

define void @abs_test(i32 %0) {
  ; CHECK: "llvm.intr.abs"{{.*}} {llvm.res_attrs = {llvm.zeroext}}
  %res = call zeroext i32 @llvm.abs.i32(i32 %0, i1 false)
  ret void
}

declare i32 @llvm.abs.i32(i32, i1) 
