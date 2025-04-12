// bazel run //tools:heir-opt -- "--annotate-module=backend=openfhe scheme=bgv" "--mlir-to-bgv=ciphertext-degree=1024" ~/fhe/heir/matvec.mlir --mlir-print-ir-after-all --mlir-print-ir-tree-dir=/tmp/mlir_demo

func.func @matvec(%arg0 : tensor<16xf32> {secret.secret}) -> tensor<16xf32> {
  %matrix = arith.constant dense<1.0> : tensor<16x16xf32>
  %matrix2 = arith.constant dense<2.0> : tensor<16x16xf32>

  %matvec0_out = arith.constant dense<0.0> : tensor<16xf32>
  %matvec1_out = arith.constant dense<0.0> : tensor<16xf32>
  %relu0_out = arith.constant dense<0.0> : tensor<16xf32>
  %relu1_out = arith.constant dense<0.0> : tensor<16xf32>
  %zero = arith.constant dense<0.0> : tensor<16xf32>

  %0 = linalg.matvec ins(%matrix, %arg0 : tensor<16x16xf32>, tensor<16xf32>) outs(%matvec0_out : tensor<16xf32>) -> tensor<16xf32>
  %relu0 = linalg.max ins(%0, %zero : tensor<16xf32>, tensor<16xf32>) outs(%relu0_out : tensor<16xf32>) -> tensor<16xf32>
  %1 = linalg.matvec ins(%matrix, %relu0 : tensor<16x16xf32>, tensor<16xf32>) outs(%matvec1_out : tensor<16xf32>) -> tensor<16xf32>
  %relu1 = linalg.max ins(%1, %zero : tensor<16xf32>, tensor<16xf32>) outs(%relu1_out : tensor<16xf32>) -> tensor<16xf32>

  return %relu1 : tensor<16xf32>
}
