#ifndef LIB_DIALECT_RNS_IR_RNSTYPES_TD_
#define LIB_DIALECT_RNS_IR_RNSTYPES_TD_

include "lib/Dialect/RNS/IR/RNSDialect.td"
include "lib/Dialect/RNS/IR/RNSTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class RNS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RNS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def RNSType : RNS_Type<"RNS", "rns"> {
  let summary = "a residue number system representation";
  let description = [{
  An RNS type represents the decomposition of a single type into a set of
  residues relative to a list of basis types.

  The basis types are required to implement RNSBasisTypeInterface in order to
  verify compatibility of types used in the same RNS basis.

  Example:

  ```mlir
  !poly_ty_1 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=3721063133 : i32>
  !poly_ty_2 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=2737228591 : i32>
  !poly_ty_3 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=3180146689 : i32>
  !ty = !rns.rns<!poly_ty_1, !poly_ty_2, !poly_ty_3>
  ```
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Type">:$basisTypes);
  let assemblyFormat = "`<` $basisTypes `>`";
  let genVerifyDecl = 1;
}

#endif  // LIB_DIALECT_RNS_IR_RNSTYPES_TD_
