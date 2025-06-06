#ifndef LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_TD_
#define LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_TD_

include "JaxiteWordDialect.td"
include "JaxiteWordTypes.td"

include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class JaxiteWord_Op<string mnemonic, list<Trait> traits = []> :
        Op<JaxiteWord_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::jaxiteword";
}

// Define the JaxiteWord_MakeCiphertext operation
def MakeCiphertextOp : JaxiteWord_Op<"make_ciphertext", [Pure]> {
  let summary = [{Create a JaxiteWord ciphertext.}];
  let description = [{
    Creates a JaxiteWord ciphertext with the given dimensions.
  }];

  let arguments = (ins I32:$polys, I32:$towers, I32:$degrees);
  let results = (outs JaxiteWord_Ciphertext:$ciphertext);

  let assemblyFormat = "$polys `,` $towers `,` $degrees attr-dict `:` type($ciphertext)";
}

def AddOp : JaxiteWord_Op<"add", [AllTypesMatch<["value_a", "value_b", "result"]>,Commutative,Pure]> {
  let description = [{
    The operation computed by this function is homomorphic addition.
  }];
  let arguments = (ins JaxiteWord_Ciphertext:$value_a,
    JaxiteWord_Ciphertext:$value_b,
    JaxiteWord_ModulusList:$modulus_list
  );
  let results = (outs JaxiteWord_Ciphertext:$result);
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_TD_
