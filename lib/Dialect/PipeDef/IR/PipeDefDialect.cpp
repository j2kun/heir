#include "lib/Dialect/PipeDef/IR/PipeDefDialect.h"

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define PipeDefOps

#include "lib/Dialect/PipeDef/IR/PipeDefAttributes.h"
#include "lib/Dialect/PipeDef/IR/PipeDefOps.h"
#include "lib/Dialect/PipeDef/IR/PipeDefTypes.h"

// Generated definitions
#include "lib/Dialect/PipeDef/IR/PipeDefDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/PipeDef/IR/PipeDefAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/PipeDef/IR/PipeDefTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/PipeDef/IR/PipeDefOps.cpp.inc"

namespace mlir {
namespace heir {
namespace pipedef {

void PipeDefDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/PipeDef/IR/PipeDefAttributes.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/PipeDef/IR/PipeDefTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/PipeDef/IR/PipeDefOps.cpp.inc"
      >();
}

}  // namespace pipedef
}  // namespace heir
}  // namespace mlir
