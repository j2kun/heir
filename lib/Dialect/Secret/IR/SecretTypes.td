#ifndef LIB_DIALECT_SECRET_IR_SECRETTYPES_TD_
#define LIB_DIALECT_SECRET_IR_SECRETTYPES_TD_

include "SecretDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Secret_Type<string name, string typeMnemonic>
    : TypeDef<Secret_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Secret : Secret_Type<"Secret", "secret"> {
  let summary = "A secret value";

  let description = [{
    A generic wrapper around another MLIR type, representing an encrypted value
    but not specifying the manner of encryption.

    When the wrapped type is a tensor, a layout attribute specifies how the
    tensor's entries are mapped to one or more ciphertexts, and this may be
    managed by compiler-internal optimization passes.
  }];

  let parameters = (ins
    "Type":$valueType,
    OptionalParameter<"Attribute">:$layout
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
        "Type":$valueType,
        CArg<"Attribute", "{}">:$layout
    ), [{
      return $_get(valueType.getContext(), valueType, layout);
    }]>
  ];
  let assemblyFormat = "`<` $valueType (`,` $layout^)? `>`";
}

#endif  // LIB_DIALECT_SECRET_IR_SECRETTYPES_TD_
