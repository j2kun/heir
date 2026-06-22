// RUN: tamagoyaki-demo --insert-equivalent-conv-layouts %s | FileCheck %s

// Single-channel convs (linalg.conv_2d) are also handled. They have a single
// MatvecDiagonal layout (no row-interchange variant), so the class holds the
// one materialized variant plus the original.

// CHECK: func.func @conv2d_single
// CHECK: linalg.conv_2d {{.*}}secret.kernel = #{{.*}}tensor_ext.layout
// CHECK: linalg.conv_2d
// CHECK-NOT: secret.kernel
// CHECK: %[[CLASS:.*]] = equivalence.class %{{[0-9]+}}, %{{[0-9]+}} : tensor<4x4xf32>
// CHECK: return %[[CLASS]]

func.func @conv2d_single(%arg0: tensor<6x6xf32>, %filter: tensor<3x3xf32>) -> tensor<4x4xf32> {
  %0 = tensor.empty() : tensor<4x4xf32>
  %1 = linalg.conv_2d
    ins(%arg0, %filter : tensor<6x6xf32>, tensor<3x3xf32>)
    outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
