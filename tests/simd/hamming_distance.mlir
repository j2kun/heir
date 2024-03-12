// RUN: heir-opt --secretize=entry-function=hamming --wrap-generic --canonicalize --cse \
// RUN:   --full-loop-unroll --insert-rotate --cse --canonicalize \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @hamming
// CHECK: secret.generic
// CHECK: arith.subi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.addi
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: secret.yield

func.func @hamming(%arg0: tensor<4xi16> {secret.secret}, %arg1: tensor<4xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %arg2 = 0 to 4 iter_args(%arg6 = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%arg2] : tensor<4xi16>
    %2 = tensor.extract %arg1[%arg2] : tensor<4xi16>
    %3 = arith.subi %1, %2 : i16
    %4 = tensor.extract %arg0[%arg2] : tensor<4xi16>
    %5 = tensor.extract %arg1[%arg2] : tensor<4xi16>
    %6 = arith.subi %4, %5 : i16
    %7 = arith.muli %3, %6 : i16
    %8 = arith.addi %arg6, %7 : i16
    affine.yield %8 : i16
  }
  return %0 : i16

  %output = arith.cmpi slt, %input, %const_128 : i1
}

func.func @hamming(%arg0: !secret.secret<tensor<4xi16>>, %arg1: !secret.secret<tensor<4xi16>>) -> !secret.secret<i16> {
  %c0_i16 = arith.constant 0 : i16
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<4xi16>>, !secret.secret<tensor<4xi16>>) {
  ^bb0(%arg2: tensor<4xi16>, %arg3: tensor<4xi16>):
    %1 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %c0_i16) -> (i16) {
      %extracted = tensor.extract %arg2[%arg4] : tensor<4xi16>
      %extracted_0 = tensor.extract %arg3[%arg4] : tensor<4xi16>
      %2 = arith.subi %extracted, %extracted_0 : i16
      %extracted_1 = tensor.extract %arg2[%arg4] : tensor<4xi16>
      %extracted_2 = tensor.extract %arg3[%arg4] : tensor<4xi16>
      %3 = arith.subi %extracted_1, %extracted_2 : i16
      %4 = arith.muli %2, %3 : i16
      %5 = arith.addi %arg5, %4 : i16
      affine.yield %5 : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}


func.func @hamming(%arg0: !secret.secret<tensor<4xi16>>, %arg1: !secret.secret<tensor<4xi16>>) -> !secret.secret<i16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<4xi16>>, !secret.secret<tensor<4xi16>>) {
  ^bb0(%arg2: tensor<4xi16>, %arg3: tensor<4xi16>):
    %extracted = tensor.extract %arg2[%c0] : tensor<4xi16>
    %extracted_0 = tensor.extract %arg3[%c0] : tensor<4xi16>
    %1 = arith.subi %extracted, %extracted_0 : i16
    %2 = arith.muli %1, %1 : i16
    %extracted_1 = tensor.extract %arg2[%c1] : tensor<4xi16>
    %extracted_2 = tensor.extract %arg3[%c1] : tensor<4xi16>
    %3 = arith.subi %extracted_1, %extracted_2 : i16
    %4 = arith.muli %3, %3 : i16
    %5 = arith.addi %2, %4 : i16
    %extracted_3 = tensor.extract %arg2[%c2] : tensor<4xi16>
    %extracted_4 = tensor.extract %arg3[%c2] : tensor<4xi16>
    %6 = arith.subi %extracted_3, %extracted_4 : i16
    %7 = arith.muli %6, %6 : i16
    %8 = arith.addi %5, %7 : i16
    %extracted_5 = tensor.extract %arg2[%c3] : tensor<4xi16>
    %extracted_6 = tensor.extract %arg3[%c3] : tensor<4xi16>
    %9 = arith.subi %extracted_5, %extracted_6 : i16
    %10 = arith.muli %9, %9 : i16
    %11 = arith.addi %8, %10 : i16
    secret.yield %11 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}


func.func @hamming(%arg0: !secret.secret<tensor<4xi16>>, %arg1: !secret.secret<tensor<4xi16>>) -> !secret.secret<i16> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<4xi16>>, !secret.secret<tensor<4xi16>>) {
  ^bb0(%arg2: tensor<4xi16>, %arg3: tensor<4xi16>):
    %1 = arith.subi %arg2, %arg3 : tensor<4xi16>
    %2 = arith.muli %1, %1 : tensor<4xi16>
    %3 = tensor_ext.rotate %2, %c3 : tensor<4xi16>, index
    %4 = arith.addi %3, %2 : tensor<4xi16>
    %5 = tensor_ext.rotate %4, %c2 : tensor<4xi16>, index
    %6 = arith.addi %5, %3 : tensor<4xi16>
    %7 = arith.addi %6, %2 : tensor<4xi16>
    %extracted = tensor.extract %7[%c3] : tensor<4xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
