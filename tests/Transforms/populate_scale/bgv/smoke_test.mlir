// RUN: heir-opt --populate-scale-bgv %s

// Test backpropagating the scale analysis to a plaintext operand of a mul op.
//
// Generated by
//
// bazel run //tools:heir-opt -- \
//  --mlir-to-bgv='ciphertext-degree=32' --scheme-to-openfhe='entry-function=simple_sum' \
//  $GIT_ROOT/tests/Transforms/mlir_to_openfhe_bgv/simple_sum.mlir --mlir-print-ir-after-failure
//
// and copying the IR at the failing step

#alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 32), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [67239937, 34359754753], P = [34359771137], plaintextModulus = 65537>, scheme.bgv} {
  func.func @simple_sum(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #tensor_ext.layout<map = (d0) -> (d0 mod 32)>>}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.original_type = #original_type}) {
    %c31 = arith.constant 31 : index
    %c1_i16 = arith.constant 1 : i16
    %cst = arith.constant dense<0> : tensor<32xi16>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %inserted = tensor.insert %c1_i16 into %cst[%c31] : tensor<32xi16>

    // the plaintext operand in question
    %0 = mgmt.init %inserted {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>

    %1 = secret.generic(%arg0 : !secret.secret<tensor<32xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) {
    ^body(%input0: tensor<32xi16>):
      %2 = tensor_ext.rotate %input0, %c16 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>, index
      %3 = arith.addi %input0, %2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>
      %4 = tensor_ext.rotate %3, %c8 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>, index
      %5 = arith.addi %3, %4 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>
      %6 = tensor_ext.rotate %5, %c4 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>, index
      %7 = arith.addi %5, %6 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>
      %8 = tensor_ext.rotate %7, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>, index
      %9 = arith.addi %7, %8 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>
      %10 = tensor_ext.rotate %9, %c1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>, index
      %11 = arith.addi %9, %10 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>

      // The mul op in question
      %12 = arith.muli %0, %11 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>

      %13 = tensor_ext.rotate %12, %c31 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<32xi16>, index
      %14 = mgmt.modreduce %13 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<32xi16>
      secret.yield %14 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %1 : !secret.secret<tensor<32xi16>>
  }
}
