#ifndef LIB_DIALECT_PIPEDEF_IR_PIPEDEFOPS_TD_
#define LIB_DIALECT_PIPEDEF_IR_PIPEDEFOPS_TD_

include "lib/Dialect/PipeDef/IR/PipeDefDialect.td"
include "lib/Dialect/PipeDef/IR/PipeDefTypes.td"
include "mlir/IR/OpBase.td"

class PipeDef_Op<string mnemonic, list<Trait> traits = []> :
        Op<PipeDef_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::pipedef";
}

#endif  // LIB_DIALECT_PIPEDEF_IR_PIPEDEFOPS_TD_
