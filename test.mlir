// 2x2 Matrix Multiplication: C = A * B
// A, B: input matrices (secret)
// C: result matrix (secret, initially zeros)
func.func @matmul_2x2_fully(
    %A: tensor<2x2xi32> {secret.secret},
    %B: tensor<2x2xi32> {secret.secret},
    %C: tensor<2x2xi32> {secret.secret}
) {

  // Triple-nested affine loops: C[i][j] = sum_k(A[i][k] * B[k][j])
  %result = affine.for %i = 0 to 2 iter_args(%acc_i = %C) -> (tensor<2x2xi32>) {
    %result_j = affine.for %j = 0 to 2 iter_args(%acc_j = %acc_i) -> (tensor<2x2xi32>) {
      %sum = affine.for %k = 0 to 2 iter_args(%acc_k = %acc_j) -> (tensor<2x2xi32>) {

        // Extract A[i][k] and B[k][j]
        %a_ik = tensor.extract %A[%i, %k] : tensor<2x2xi32>
        %b_kj = tensor.extract %B[%k, %j] : tensor<2x2xi32>

        // Extract current result: C[i][j]
        %c_ij = tensor.extract %acc_k[%i, %j] : tensor<2x2xi32>

        // Compute: C[i][j] += A[i][k] * B[k][j]
        %prod = arith.muli %a_ik, %b_kj : i32
        %new_c = arith.addi %c_ij, %prod : i32

        // Update C[i][j]
        %updated = tensor.insert %new_c into %acc_k[%i, %j] : tensor<2x2xi32>

        affine.yield %updated : tensor<2x2xi32>
      }
      affine.yield %sum : tensor<2x2xi32>
    }
    affine.yield %result_j : tensor<2x2xi32>
  }

  return
}
