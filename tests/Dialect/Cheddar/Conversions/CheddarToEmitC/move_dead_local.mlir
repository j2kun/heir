// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar,arith,scf %s | FileCheck %s

// A copy out of a local temporary that is never used again is a move.
// CHECK: func.func @dead_local(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">> {cheddar.support = "context"}, %[[IN:.*]]: !emitc.lvalue<!emitc.opaque<"Ciphertext<word>">>, %[[DST:.*]]: !emitc.lvalue<!emitc.opaque<"Ciphertext<word>">>)
// CHECK: %[[TMP:.*]] = "emitc.variable"
// CHECK: "Neg"(%[[TMP]], %[[IN]])
// CHECK: emitc.verbatim "{} = std::move({});" args %[[DST]], %[[TMP]]
// CHECK-NOT: Copy
func.func @dead_local(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>, %dst: memref<!cheddar.ciphertext>) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %dst : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// The deallocation of the temporary does not count as a use.
// CHECK: func.func @dead_local_dealloc
// CHECK: emitc.verbatim "{} = std::move({});"
// CHECK-NOT: Copy
func.func @dead_local_dealloc(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>, %dst: memref<!cheddar.ciphertext>) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %dst : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  memref.dealloc %tmp : memref<!cheddar.ciphertext>
  return
}

// A temporary that is still read after the copy is deep-copied.
// CHECK: func.func @live_local
// CHECK: Copy
// CHECK-NOT: std::move
func.func @live_local(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>, %dst: memref<!cheddar.ciphertext>) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %dst : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %tmp, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  return
}

// A temporary allocated outside the block of the copy is reused across
// iterations, so it is not dead: deep-copy.
// CHECK: func.func @outer_temporary
// CHECK: Copy
// CHECK-NOT: std::move
func.func @outer_temporary(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>, %dst: memref<!cheddar.ciphertext>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  scf.for %i = %c0 to %n step %c1 {
    cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
    memref.copy %tmp, %dst : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  }
  return
}

// Move-only types without a deep-copy API can be moved out of a dead temporary.
// CHECK: func.func @dead_local_plaintext
// CHECK: emitc.verbatim "{} = std::move({});"
func.func @dead_local_plaintext(%enc: !cheddar.encoder, %msg: memref<4xf64>, %dst: memref<!cheddar.plaintext>) {
  %tmp = memref.alloc() : memref<!cheddar.plaintext>
  cheddar.encode %enc, %msg, %tmp {level = 1 : i64} : (!cheddar.encoder, memref<4xf64>, memref<!cheddar.plaintext>) -> ()
  memref.copy %tmp, %dst : memref<!cheddar.plaintext> to memref<!cheddar.plaintext>
  return
}

// A use nested in a region before the copy keeps the temporary dead afterwards.
// CHECK: func.func @use_in_region_before
// CHECK: emitc.verbatim "{} = std::move({});"
// CHECK-NOT: Copy
func.func @use_in_region_before(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>, %dst: memref<!cheddar.ciphertext>, %c: i1) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  scf.if %c {
    cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  }
  memref.copy %tmp, %dst : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// A use nested in a region after the copy: deep-copy.
// CHECK: func.func @use_in_region_after
// CHECK: Copy
// CHECK-NOT: std::move
func.func @use_in_region_after(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>, %dst: memref<!cheddar.ciphertext>, %c: i1) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %dst : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  scf.if %c {
    cheddar.neg %ctx, %tmp, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  }
  return
}

// A view of the temporary may outlive the copy: deep-copy.
// CHECK: func.func @viewed_temporary
// CHECK: Copy
// CHECK-NOT: std::move
func.func @viewed_temporary(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>, %dst: memref<2x!cheddar.ciphertext>) {
  %tmp = memref.alloc() : memref<2x!cheddar.ciphertext>
  %slot = memref.subview %tmp[0] [1] [1] : memref<2x!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %slot : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %dst : memref<2x!cheddar.ciphertext> to memref<2x!cheddar.ciphertext>
  return
}

// A rank-1 temporary moves element-wise.
// CHECK: func.func @dead_local_array
// CHECK: emitc.verbatim "for (size_t _i = 0; _i < 2; ++_i) {}[_i] = std::move({}[_i]);"
// CHECK-NOT: Copy
func.func @dead_local_array(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<2x!cheddar.ciphertext>, %dst: memref<2x!cheddar.ciphertext>) {
  %tmp = memref.alloc() : memref<2x!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<2x!cheddar.ciphertext>, memref<2x!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %dst : memref<2x!cheddar.ciphertext> to memref<2x!cheddar.ciphertext>
  return
}

// A dead local user interface moves too (unique_ptr), where a copy is an error.
// CHECK: func.func @dead_local_user_interface
// CHECK: emitc.verbatim "{} = std::move({});"
func.func @dead_local_user_interface(%dst: memref<!cheddar.user_interface>) {
  %tmp = memref.alloc() : memref<!cheddar.user_interface>
  memref.copy %tmp, %dst : memref<!cheddar.user_interface> to memref<!cheddar.user_interface>
  return
}

// A self-copy of a dead temporary is erased, not self-moved.
// CHECK: func.func @self_copy_dead_local
// CHECK-NOT: std::move
// CHECK-NOT: Copy
// CHECK: return
func.func @self_copy_dead_local(%ctx: !cheddar.context {cheddar.support = "context"}, %in: memref<!cheddar.ciphertext>) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.copy %tmp, %tmp : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}
