#ifndef LIB_DIALECT_PIPEDEF_IR_PIPEDEFOPS_H_
#define LIB_DIALECT_PIPEDEF_IR_PIPEDEFOPS_H_

#include "lib/Dialect/PipeDef/IR/PipeDefDialect.h"
#include "lib/Dialect/PipeDef/IR/PipeDefTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/PipeDef/IR/PipeDefOps.h.inc"

#endif  // LIB_DIALECT_PIPEDEF_IR_PIPEDEFOPS_H_
