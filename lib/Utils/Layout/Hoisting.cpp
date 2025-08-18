#include "lib/Utils/Layout/Hoisting.h"

#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

presburger::IntegerRelation hoistConversionThroughMatvec(
    const IntegerRelation& matrixLayout, const IntegerRelation& fromVecLayout,
    const IntegerRelation& toVecLayout) {
  // The intuition for this function is that the conversion from fromVecLayout
  // to toVecLayout implies some transformation of the slot ordering of the
  // packed vector. The kernels we support have matrix layouts for which the
  // packed slots of a ciphertext track the columns of the packed vector.
  // So we need to apply the same transformation of the packed vector slots
  // to the packed matrix slots.
  //
  // This function works in two steps:
  //
  // 1. Compute the inferred re-packing relation of vector slots (i.e., a
  // relation (ct, slots) -> (ct, slots))
  //
  // 2. Compose the transformation from (1) with the (ct, slot) dims of the
  // matrix packing.

  // llvm::outs() << "fromVecLayout:\n";
  // llvm::outs().flush();
  // fromVecLayout.dump();
  // llvm::outs() << "toVecLayout:\n";
  // llvm::outs().flush();
  // toVecLayout.dump();

  IntegerRelation fromClone(fromVecLayout);
  fromClone.inverse();
  // llvm::outs() << "fromCloneInverse:\n";
  // llvm::outs().flush();
  // fromClone.dump();

  fromClone.compose(toVecLayout);
  fromClone.removeRedundantConstraints();
  fromClone.simplify();

  // llvm::outs() << "fromClone:\n";
  // llvm::outs().flush();
  // fromClone.dump();

  IntegerRelation result(matrixLayout);
  result.applyRange(fromClone);
  return result;
}

}  // namespace heir
}  // namespace mlir
