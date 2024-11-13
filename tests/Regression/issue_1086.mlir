// RUN: heir-opt %s --secret-to-cggi | FileCheck %s

// CHECK-LABEL: @trivial_loop
func.func @trivial_loop(%arg0: !secret.secret<memref<2xi3>>, %arg1: !secret.secret<i3>) -> !secret.secret<i3> {
  %c0 = arith.constant 0 : index
  %0 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %arg1) -> (!secret.secret<i3>) {
    %1 = secret.generic ins(%arg0 : !secret.secret<memref<2xi3>>) {
    ^bb0(%arg4: memref<2xi3>):
      %4 = memref.load %arg4[%c0] : memref<2xi3>
      secret.yield %4 : i3
    } -> !secret.secret<i3>
    %2 = secret.generic {
      %alloc = memref.alloc() : memref<3xi1>
      secret.yield %alloc : memref<3xi1>
    } -> !secret.secret<memref<3xi1>>
    %3 = secret.cast %2 : !secret.secret<memref<3xi1>> to !secret.secret<i3>
    affine.yield %3 : !secret.secret<i3>
  }
  return %0 : !secret.secret<i3>
}

// CHECK-LABEL: @sum
func.func @sum(%arg0: !secret.secret<memref<2xi3>>) -> !secret.secret<i3> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i3 = arith.constant 0 : i3
  %0 = secret.conceal %c0_i3 : i3 -> !secret.secret<i3>
  %1 = affine.for %arg1 = 0 to 2 iter_args(%arg2 = %0) -> (!secret.secret<i3>) {
    %2 = secret.cast %arg0 : !secret.secret<memref<2xi3>> to !secret.secret<memref<6xi1>>
    %3 = secret.generic ins(%2 : !secret.secret<memref<6xi1>>) {
    ^bb0(%arg3: memref<6xi1>):
      %5 = memref.load %arg3[%c1] : memref<6xi1>
      %6 = memref.load %arg3[%c2] : memref<6xi1>
      %7 = comb.truth_table %true, %5, %6 -> 1 : ui8
      %alloc = memref.alloc() : memref<3xi1>
      memref.store %7, %alloc[%c0] : memref<3xi1>
      memref.store %7, %alloc[%c1] : memref<3xi1>
      memref.store %7, %alloc[%c2] : memref<3xi1>
      secret.yield %alloc : memref<3xi1>
    } -> !secret.secret<memref<3xi1>>
    %4 = secret.cast %3 : !secret.secret<memref<3xi1>> to !secret.secret<i3>
    affine.yield %4 : !secret.secret<i3>
  }
  return %1 : !secret.secret<i3>
}
