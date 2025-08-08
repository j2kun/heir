// RUN: heir-opt %s | FileCheck %s

// CHECK: @test_arith_syntax
module {
  pipedef.pipeline {
      name = "mlirToSecretArithmetic",
      cli_flag = "mlir-to-secret-arithmetic",
      description = "Convert a func using standard MLIR dialects to secret dialect with arithmetic ops.",
  } {
    %enableArithmetization = pipedef.option {
      name = "enable-arithmetization",
      description = "Enable arithmetization of MLIR operations; if false, input is assumed to be arithmetized",
      default_value = "true"
    } : i1
    %ciphertextDegree = pipedef.option {
      name = "ciphertext-degree",
      description = "The degree of the polynomials to use for ciphertexts; equivalently, the number of messages that can be packed into a single ciphertext.",
      default_value = 1024 : i64
    } : i64
  }
}
