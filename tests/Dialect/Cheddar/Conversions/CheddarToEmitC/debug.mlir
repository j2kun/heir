// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// A `__heir_debug_*` call lowers to `__heir_debug(encoder, ui, ct, "name",
// "metadata")` (`ct, N` for a buffer of N ciphertexts); the external
// declaration is erased.

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!user_interface = !cheddar.user_interface

// The external declaration must NOT survive into the emitted module.
// CHECK-NOT: @__heir_debug_0
func.func private @__heir_debug_0(!encoder, !user_interface, tensor<!ciphertext>)

// CHECK: func.func @debug_chain
// Encoder + scalar ciphertext inputs are tightened to const C++ references; the
// UserInterface stays a pointer.
// CHECK-SAME: !emitc.opaque<"const Encoder<word>&">
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// The debug.validate -> __heir_debug call: encoder, ui, ct operands plus the
// name + metadata baked as trailing string-literal opaque args.
// CHECK: emitc.call_opaque "__heir_debug"
// CHECK-SAME: %arg0, %arg1, %arg2
// CHECK-SAME: heir_val0
// CHECK-SAME: heir_meta0
func.func @debug_chain(%enc: !encoder, %ui: !user_interface, %ct: tensor<!ciphertext>) {
  func.call @__heir_debug_0(%enc, %ui, %ct) {debug.name = "heir_val0", debug.metadata = "heir_meta0"} : (!encoder, !user_interface, tensor<!ciphertext>) -> ()
  return
}

// A rank-1 (1-element array) ciphertext value -- the usual cheddar value rep --
// is a `const Ciphertext<word>[1]` argument, passed with its element count.
// CHECK: func.func @debug_arr
// CHECK: emitc.call_opaque "__heir_debug"
// CHECK-SAME: #emitc.opaque<"1">
// CHECK-SAME: !emitc.array<1x!emitc.opaque<"const Ciphertext<word>">>
func.func private @__heir_debug_1(!encoder, !user_interface, tensor<1x!ciphertext>)
func.func @debug_arr(%enc: !encoder, %ui: !user_interface, %ct: tensor<1x!ciphertext>) {
  func.call @__heir_debug_1(%enc, %ui, %ct) {debug.name = "arr0", debug.metadata = "m"} : (!encoder, !user_interface, tensor<1x!ciphertext>) -> ()
  return
}
