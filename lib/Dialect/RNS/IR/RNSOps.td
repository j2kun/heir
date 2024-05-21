#ifndef LIB_DIALECT_RNS_IR_RNSOPS_TD_
#define LIB_DIALECT_RNS_IR_RNSOPS_TD_

include "lib/Dialect/RNS/IR/RNSDialect.td"
include "lib/Dialect/RNS/IR/RNSTypes.td"
include "mlir/IR/OpBase.td"

class RNS_Op<string mnemonic, list<Trait> traits = []> :
        Op<RNS_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::rns";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def RNS_DecomposeOp : RNS_Op<"decompose", [
    Pure,
]> {
  let summary = "Decompose a value into RNS components";
  let description = [{
  Example:

  ```mlir
  // Convert a coefficient modulus into three smaller residue rings with prime coefficients
  !poly_ty = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=1001 : i32>>

  !poly_ty_1 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=7 : i32>>
  !poly_ty_2 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=11 : i32>>
  !poly_ty_3 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=13 : i32>>
  !ty = !rns.rns<!poly_ty_1, !poly_ty_2, !poly_ty_3>

  %output = rns.decompose %input : !poly_ty -> !rns<!poly_ty_1, !poly_ty_2, !poly_ty_3>
  ```
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs RNSType:$output);
}

def RNS_RecomposeOp : RNS_Op<"recompose", [
    Pure,
]> {
  let summary = "Recompose a value into RNS components";
  let description = [{
  The inverse operation of `rns.decompose`.

  Example:

  ```mlir
  !poly_ty = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=1001 : i32>>

  !poly_ty_1 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=7 : i32>>
  !poly_ty_2 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=11 : i32>>
  !poly_ty_3 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=13 : i32>>
  !ty = !rns.rns<!poly_ty_1, !poly_ty_2, !poly_ty_3>

  %output = rns.recompose %input : !rns<!poly_ty_1, !poly_ty_2, !poly_ty_3> -> !poly_ty
  ```
  }];
  let arguments = (ins RNSType:$input);
  let results = (outs AnyType:$output);
}


#endif  // LIB_DIALECT_RNS_IR_RNSOPS_TD_
