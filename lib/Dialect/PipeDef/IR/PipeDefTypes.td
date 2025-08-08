#ifndef LIB_DIALECT_PIPEDEF_IR_PIPEDEFTYPES_TD_
#define LIB_DIALECT_PIPEDEF_IR_PIPEDEFTYPES_TD_

include "lib/Dialect/PipeDef/IR/PipeDefDialect.td"
include "lib/Dialect/PipeDef/IR/PipeDefAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class PipeDef_Type<string name, string typeMnemonic>
    : TypeDef<PipeDef_Dialect, name> {
  let mnemonic = typeMnemonic;
}


#endif  // LIB_DIALECT_PIPEDEF_IR_PIPEDEFTYPES_TD_
