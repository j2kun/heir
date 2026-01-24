// RUN: heir-opt --populate-scale-bgv %s | FileCheck %s

// This test ensures that the scale is large enough to avoid overflow.

// CHECK-NOT: scale = -{{[0-9]+}}

#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and slot = 0 }">
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [134250497, 2148728833, 2148794369, 2149810177, 4398047232001], P = [4398048575489, 4398048706561], plaintextModulus = 65537>, scheme.bgv} {
  func.func @simple_sum(%arg0: !secret.secret<tensor<1x32xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 4>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 32 = 0 and 0 <= i0 <= 31 and 0 <= slot <= 31 }">>}) -> (!secret.secret<tensor<1x32xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>, tensor_ext.original_type = #original_type}) {
    %c15 = arith.constant 15 : index
    %cst = arith.constant dense<[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]> : tensor<1x32xi16>
    %cst_0 = arith.constant dense<[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]> : tensor<1x32xi16>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %0 = mgmt.init %cst_0 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>
    %1 = mgmt.init %cst_0 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x32xi16>
    %2 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x32xi16>
    %3 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x32xi16>
    %4 = secret.generic(%arg0: !secret.secret<tensor<1x32xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 4>}) {
    ^body(%input0: tensor<1x32xi16>):
      %5 = tensor_ext.rotate %input0, %c16 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>, index
      %6 = arith.addi %input0, %5 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>
      %7 = tensor_ext.rotate %6, %c8 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>, index
      %8 = arith.addi %6, %7 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>
      %9 = tensor_ext.rotate %8, %c4 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>, index
      %10 = arith.addi %8, %9 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>
      %11 = tensor_ext.rotate %10, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>, index
      %12 = arith.addi %10, %11 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>
      %13 = tensor_ext.rotate %12, %c1 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>, index
      %14 = arith.addi %12, %13 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>
      %15 = arith.muli %14, %0 {mgmt.mgmt = #mgmt.mgmt<level = 4>} : tensor<1x32xi16>
      %16 = mgmt.modreduce %15 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x32xi16>
      %17 = arith.muli %16, %1 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x32xi16>
      %18 = tensor_ext.rotate %17, %c15 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x32xi16>, index
      %19 = mgmt.modreduce %18 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x32xi16>
      %20 = arith.muli %19, %2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x32xi16>
      %21 = mgmt.modreduce %20 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x32xi16>
      %22 = arith.muli %21, %3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x32xi16>
      %23 = tensor_ext.rotate %22, %c16 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x32xi16>, index
      %24 = mgmt.modreduce %23 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x32xi16>
      secret.yield %24 : tensor<1x32xi16>
    } -> (!secret.secret<tensor<1x32xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %4 : !secret.secret<tensor<1x32xi16>>
  }
}


