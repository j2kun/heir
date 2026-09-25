#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h.inc"

namespace mlir {
namespace heir {
namespace cheddar {

// Argument attribute naming the kind of a lowered support argument (a context,
// encoder, key, ...). Its value is the support type's mnemonic, so later
// stages read the role off the argument instead of recovering it from the
// type, or after EmitC conversion from a C++ type name.
constexpr ::llvm::StringLiteral kSupportArgAttrName = "cheddar.support";

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_
