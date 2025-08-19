#include "lib/Utils/Layout/Hoisting.h"

#include <iostream>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

// ISL
#include "include/isl/ast.h"             // from @isl
#include "include/isl/ast_build.h"       // from @isl
#include "include/isl/ast_type.h"        // from @isl
#include "include/isl/constraint.h"      // from @isl
#include "include/isl/ctx.h"             // from @isl
#include "include/isl/local_space.h"     // from @isl
#include "include/isl/map.h"             // from @isl
#include "include/isl/map_type.h"        // from @isl
#include "include/isl/set.h"             // from @isl
#include "include/isl/space.h"           // from @isl
#include "include/isl/space_type.h"      // from @isl
#include "include/isl/union_map.h"       // from @isl
#include "include/isl/union_map_type.h"  // from @isl

namespace mlir {
namespace heir {

using presburger::BoundType;
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

  IntegerRelation fromClone(fromVecLayout);
  IntegerRelation toClone(toVecLayout);

  // Project out the ciphertext dimension, though this will need to change
  // when we get a larger vector than can fit in one ciphertext
  std::optional<int64_t> ctUpperBound =
      fromClone.getConstantBound64(BoundType::UB, 1);
  std::optional<int64_t> ctLowerBound =
      fromClone.getConstantBound64(BoundType::LB, 1);
  fromClone.projectOut(1);
  toClone.projectOut(1);
  fromClone.inverse();

  fromClone.compose(toClone);
  fromClone.removeRedundantConstraints();
  fromClone.simplify();

  // Put the ct dim back in with same bound constraints as original
  unsigned ctDomainVar = fromClone.insertVar(VarKind::Domain, 0, 1);
  unsigned ctRangeVar = fromClone.insertVar(VarKind::Range, 0, 1);
  if (ctUpperBound.has_value()) {
    fromClone.addBound(BoundType::UB, ctDomainVar, *ctUpperBound);
    fromClone.addBound(BoundType::UB, ctRangeVar, *ctUpperBound);
  }
  if (ctLowerBound.has_value()) {
    fromClone.addBound(BoundType::LB, ctDomainVar, *ctLowerBound);
    fromClone.addBound(BoundType::LB, ctRangeVar, *ctLowerBound);
  }

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* islRel = convertRelationToBasicMap(fromClone, ctx);
  char* resultStr = isl_basic_map_to_str(islRel);
  std::string actual(resultStr);
  free(resultStr);
  std::cout << actual << std::endl;
  isl_basic_map_free(islRel);
  isl_ctx_free(ctx);

  IntegerRelation result(matrixLayout);
  result.applyRange(fromClone);
  return result;
}

}  // namespace heir
}  // namespace mlir
