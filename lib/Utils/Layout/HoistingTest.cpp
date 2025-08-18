#include <cmath>
#include <cstdint>
#include <optional>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Hoisting.h"
#include "lib/Utils/Layout/Parser.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;

TEST(UtilsTest, DiagonalLayout) {
  // MLIRContext context;

  // // Diagonalize a 4x8 matrix into a 4x64 matrix.
  // RankedTensorType matrixType =
  //     RankedTensorType::get({4, 8}, IndexType::get(&context));
  // RankedTensorType diagonalizedType =
  //     RankedTensorType::get({4, 64}, IndexType::get(&context));
  // IntegerRelation diagonalRelation =
  //     getDiagonalLayoutRelation(matrixType, diagonalizedType);

  // diagonalRelation.simplify();
  // for (unsigned int i = 0; i < 4; ++i) {
  //   for (unsigned int j = 0; j < 64; ++j) {
  //     auto maybeExists =
  //         diagonalRelation.containsPointNoLocal({j % 4, (i + j) % 8, i, j});
  //     EXPECT_TRUE(maybeExists.has_value());
  //   }
  // }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
