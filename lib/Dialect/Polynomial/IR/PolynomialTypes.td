#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_

include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"
include "mlir/IR/OpAsmInterface.td"

class Polynomial_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Polynomial_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Polynomial_PolynomialType : Polynomial_Type<"Polynomial", "polynomial", [OpAsmTypeInterface]> {
  let summary = "An element of a polynomial ring.";
  let description = [{
    A type for polynomials in a polynomial quotient ring.
  }];
  let parameters = (ins Polynomial_RingAttr:$ring);
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // OpAsmTypeInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "poly";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def PolynomialLike: TypeOrValueSemanticsContainer<Polynomial_PolynomialType, "polynomial-like">;


#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTYPES_TD_
