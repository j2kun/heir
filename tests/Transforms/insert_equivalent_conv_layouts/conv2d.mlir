// RUN: tamagoyaki-demo --insert-equivalent-conv-layouts %s | FileCheck %s --check-prefix=KEEP
// RUN: tamagoyaki-demo --insert-equivalent-conv-layouts=remove-original=true %s | FileCheck %s --check-prefix=REMOVE

// Three layout variants are materialized for the convolution: the MatvecDiagonal
// kernel with and without the pixel-shuffle row-interchange, and the naive
// MatvecNaive packing. Each is a copy of the conv carrying a kernel and a result
// layout, fed by operand conversions and followed by a conversion back to the
// common layout. The variant results are grouped into an equivalence.class, and
// users of the original read the class result.

// KEEP-DAG: name = "MatvecDiagonal"
// KEEP-DAG: name = "MatvecNaive"
// KEEP-LABEL: func.func @conv2d
// KEEP-COUNT-3: linalg.conv_2d_nchw_fchw {{.*}}secret.kernel = #{{.*}}tensor_ext.layout
// The original, layout-free convolution is kept (no kernel) ...
// KEEP: linalg.conv_2d_nchw_fchw
// KEEP-NOT: secret.kernel
// ... and joins the class as a fourth member; the result is the class result.
// KEEP: %[[CLASS:.*]] = equivalence.class %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : tensor<1x4x5x5xf32>
// KEEP: return %[[CLASS]]

// REMOVE-DAG: name = "MatvecDiagonal"
// REMOVE-DAG: name = "MatvecNaive"
// REMOVE-LABEL: func.func @conv2d
// REMOVE-COUNT-3: linalg.conv_2d_nchw_fchw {{.*}}secret.kernel = #{{.*}}tensor_ext.layout
// Exactly the three variants remain; the original conv is erased.
// REMOVE-NOT: linalg.conv_2d_nchw_fchw
// REMOVE: %[[CLASS:.*]] = equivalence.class %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : tensor<1x4x5x5xf32>
// REMOVE: return %[[CLASS]]

func.func @conv2d(%arg0: tensor<1x1x10x10xf32>, %filter: tensor<4x1x2x2xf32>) -> tensor<1x4x5x5xf32> {
  %0 = tensor.empty() : tensor<1x4x5x5xf32>
  %1 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%arg0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>)
    outs(%0 : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
  return %1 : tensor<1x4x5x5xf32>
}
