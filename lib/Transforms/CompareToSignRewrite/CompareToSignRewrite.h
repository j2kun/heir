#ifndef LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_H_
#define LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_H_

#include "lib/Dialect/MathExt/IR/MathExtDialect.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"    // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project
namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/CompareToSignRewrite/CompareToSignRewrite.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/CompareToSignRewrite/CompareToSignRewrite.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_COMPARETOSIGNREWRITE_COMPARETOSIGNREWRITE_H_
