#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_TD_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_TD_

include "mlir/Pass/PassBase.td"

def SecretToCKKS : Pass<"secret-to-ckks"> {
  let summary = "Lower `secret` to `ckks` dialect.";

  let description = [{
    This pass lowers an IR with `secret.generic` blocks containing arithmetic
    operations to operations on ciphertexts with the CKKS dialect.

    The pass assumes that the `secret.generic` regions have been distributed
    through arithmetic operations so that only one ciphertext operation appears
    per generic block. It also requires that `canonicalize` was run so that
    non-secret values used are removed from the `secret.generic`'s block
    arguments.

    The pass requires that all types are tensors of a uniform shape matching the
    dimension of the ciphertext space specified my `poly-mod-degree`.
  }];

  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::heir::rns::RNSDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];

  let options = [
    Option<"polyModDegree", "poly-mod-degree", "int",
           /*default=*/"1024", "Default degree of the cyclotomic polynomial "
           "modulus to use for ciphertext space.">,
  ];
}

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_TD_
