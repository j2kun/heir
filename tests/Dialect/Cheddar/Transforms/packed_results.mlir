// RUN: heir-opt --cse --cheddar-bufferize %s | FileCheck %s

// CSE shares both scalar and packed empty tensors. The two independently
// encoded results must still be produced directly in separate output slots.
// CHECK: func.func @packed_results(
// CHECK-SAME: %[[OUT0:[a-zA-Z0-9_]+]]: memref<1x!plaintext> {bufferize.result}, %[[OUT1:[a-zA-Z0-9_]+]]: memref<1x!plaintext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: %[[SLOT0:[a-zA-Z0-9_]+]] = memref.subview %[[OUT0]][0] [1] [1]
// CHECK: cheddar.encode {{.*}}, %[[SLOT0]] {level = 1 : i64}
// CHECK-NOT: memref.alloc
// CHECK: %[[SLOT1:[a-zA-Z0-9_]+]] = memref.subview %[[OUT1]][0] [1] [1]
// CHECK: cheddar.encode {{.*}}, %[[SLOT1]] {level = 0 : i64}
// CHECK-NEXT: return
func.func @packed_results(%encoder: !cheddar.encoder, %input0: tensor<4xf32>, %input1: tensor<4xf32>) -> (tensor<1x!cheddar.plaintext>, tensor<1x!cheddar.plaintext>) {
  %scalar0 = tensor.empty() : tensor<!cheddar.plaintext>
  %pt0 = cheddar.encode %encoder, %input0, %scalar0 {level = 1 : i64} : (!cheddar.encoder, tensor<4xf32>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  %scalar1 = tensor.empty() : tensor<!cheddar.plaintext>
  %pt1 = cheddar.encode %encoder, %input1, %scalar1 {level = 0 : i64} : (!cheddar.encoder, tensor<4xf32>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  %packed0 = tensor.empty() : tensor<1x!cheddar.plaintext>
  %result0 = tensor.insert_slice %pt0 into %packed0[0] [1] [1] : tensor<!cheddar.plaintext> into tensor<1x!cheddar.plaintext>
  %packed1 = tensor.empty() : tensor<1x!cheddar.plaintext>
  %result1 = tensor.insert_slice %pt1 into %packed1[0] [1] [1] : tensor<!cheddar.plaintext> into tensor<1x!cheddar.plaintext>
  return %result0, %result1 : tensor<1x!cheddar.plaintext>, tensor<1x!cheddar.plaintext>
}
