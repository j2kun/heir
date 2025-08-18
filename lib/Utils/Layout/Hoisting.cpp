#include "lib/Utils/Layout/Hoisting.h"

#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

namespace mlir {
namespace heir {

using presburger::IntegerRelation;

presburger::IntegerRelation hoistConversionThroughMatvec(
    const presburger::IntegerRelation& matrixLayout,
    const presburger::IntegerRelation& fromVecLayout,
    const presburger::IntegerRelation& toVecLayout) {
  return matrixLayout;
}

}  // namespace heir
}  // namespace mlir
