#ifndef LIB_DIALECT_PIPEDEF_IR_PIPEDEFDIALECT_TD_
#define LIB_DIALECT_PIPEDEF_IR_PIPEDEFDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def PipeDef_Dialect : Dialect {
  let name = "pipedef";
  let description = [{
    The `pipedef` dialect is an IR for defining MLIR pipelines
  }];

  let cppNamespace = "::mlir::heir::pipedef";
}

#endif  // LIB_DIALECT_PIPEDEF_IR_PIPEDEFDIALECT_TD_
