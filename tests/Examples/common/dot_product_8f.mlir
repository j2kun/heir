func.func @dot_product_all(%arg0: tensor<4x512xf32> {secret.secret}, %arg1: tensor<512xf32> {secret.secret}) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %result_init = tensor.splat %zero : tensor<4xf32>
  %0 = affine.for %data_index = 0 to 4 iter_args(%result_iter = %result_init) {
    %item = tensor.extract_slice %arg0[%data_index, 0] [1, 512] [1, 1] : tensor<4x512xf32> to tensor<512xf32>
    %0 = affine.for %i = 0 to 512 iter_args(%iter = %zero) -> (f32) {
      %1 = tensor.extract %arg1[%i] : tensor<512xf32>
      %2 = tensor.extract %item[%i] : tensor<512xf32>
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %iter, %3 : f32
      affine.yield %4 : f32
    }
    %update = tensor.insert %0 into %result_iter[%data_index] : tensor<4xf32>
    affine.yield %update : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}
