// RUN: tamagoyaki-demo --insert-equivalent-conv-layouts %s | FileCheck %s

// The pass also handles 1-D multichannel convolutions.

// CHECK: func.func @conv1d
// CHECK-COUNT-2: linalg.conv_1d_ncw_fcw {{.*}}secret.kernel = #{{.*}}tensor_ext.layout
// The kept original joins the class as a third member; the result is the class.
// CHECK: linalg.conv_1d_ncw_fcw
// CHECK-NOT: secret.kernel
// CHECK: %[[CLASS:.*]] = equivalence.class %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : tensor<1x8x2xf32>
// CHECK: return %[[CLASS]]

func.func @conv1d(%arg0: tensor<1x1x4xf32>, %filter: tensor<8x1x2xf32>) -> tensor<1x8x2xf32> {
  %0 = tensor.empty() : tensor<1x8x2xf32>
  %1 = linalg.conv_1d_ncw_fcw
    {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
    ins(%arg0, %filter : tensor<1x1x4xf32>, tensor<8x1x2xf32>)
    outs(%0 : tensor<1x8x2xf32>) -> tensor<1x8x2xf32>
  return %1 : tensor<1x8x2xf32>
}
