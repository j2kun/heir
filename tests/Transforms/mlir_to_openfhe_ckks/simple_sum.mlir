// RUN: heir-opt --mlir-to-ckks='ciphertext-degree=32' --scheme-to-openfhe='entry-function=simple_sum' %s | FileCheck %s

// CHECK: @simple_sum
// CHECK-COUNT-6: openfhe.rot
// CHECK: return
func.func @simple_sum(%arg0: tensor<32xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %i = 0 to 32 iter_args(%sum_iter = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%i] : tensor<32xi16>
    %2 = arith.addi %1, %sum_iter : i16
    affine.yield %2 : i16
  }
  return %0 : i16
}
