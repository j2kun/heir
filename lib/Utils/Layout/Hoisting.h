#ifndef LIB_UTILS_LAYOUT_HOISTING_H_
#define LIB_UTILS_LAYOUT_HOISTING_H_

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

namespace mlir {
namespace heir {

// FIXME: add good docstring
presburger::IntegerRelation hoistConversionThroughMatvec(
    const presburger::IntegerRelation& matrixLayout,
    const presburger::IntegerRelation& fromVecLayout,
    const presburger::IntegerRelation& toVecLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_HOISTING_H_
