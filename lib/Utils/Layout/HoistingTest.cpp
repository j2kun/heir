#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "lib/Utils/Layout/Parser.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;
using testing::Eq;

void debugFailure(const IntegerRelation &expected,
                  const IntegerRelation &actual) {
  llvm::outs() << "Expected:\n";
  expected.print(llvm::outs());
  llvm::outs() << "\nActual:\n";
  actual.print(llvm::outs());

  llvm::outs() << "Printing a sample point from expected:\n";
  auto maybeSample = expected.findIntegerSample();
  EXPECT_TRUE(maybeSample.has_value());
  SmallVector<DynamicAPInt, 8> sample = maybeSample.value();

  llvm::outs() << "Sample point in expected (with locals): \n";
  for (auto &value : sample) {
    value.print(llvm::outs());
    llvm::outs() << ", ";
  }
  llvm::outs() << "\n";
  llvm::outs().flush();

  llvm::outs() << "Codegen for actual:\n";
  auto result = generateLoopNestAsCStr(actual);
  if (failed(result)) {
    llvm::outs() << "Failed to generate code for actual relation.\n";
  } else {
    std::string actualCode = result.value();
    llvm::outs() << actualCode << "\n";
  }

  llvm::outs() << "Codegen for expected:\n";
  result = generateLoopNestAsCStr(expected);
  if (failed(result)) {
    llvm::outs() << "Failed to generate code for expected relation.\n";
  } else {
    std::string expectedCode = result.value();
    llvm::outs() << expectedCode << "\n";
  }

  llvm::outs().flush();
}

TEST(UtilsTest, DiagonalLayout) {
  MLIRContext context;
  IntegerRelation fromVecLayout = relationFromString(
      "(d, ct, slot) : "
      "((d - slot) mod 8 == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation toVecLayout = relationFromString(
      "(d, ct, slot) : "
      "((d - (slot + 3)) mod 8 == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation matrixLayout = relationFromString(
      "(row, col, ct, slot) : "
      "((slot - row) mod 8 == 0, "
      "(ct + slot - col) mod 8 == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);
  IntegerRelation expected = relationFromString(
      "(row, col, ct, slot) : "
      "(((slot + 3) - row) mod 8 == 0, "
      "(ct + (slot + 3) - col) mod 8 == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);

  IntegerRelation actual =
      hoistConversionThroughMatvec(matrixLayout, fromVecLayout, toVecLayout);

  auto actualVolume = actual.computeVolume();
  auto expectedVolume = actual.computeVolume();

  EXPECT_THAT(expectedVolume, Eq(actualVolume));

  // Pick any value to check if it exists in the expected and actual relation.
  auto maybeSample = actual.findIntegerSample();
  EXPECT_TRUE(maybeSample.has_value());
  SmallVector<DynamicAPInt, 8> sample = maybeSample.value();

  llvm::outs() << "Sample point in actual (with locals): \n";
  for (auto &value : sample) {
    value.print(llvm::outs());
    llvm::outs() << ", ";
  }
  llvm::outs() << "\n";
  llvm::outs().flush();

  // Copy first four values which gives domain/range values
  SmallVector<DynamicAPInt, 4> point(sample.begin(), sample.begin() + 4);
  auto maybeExists = expected.containsPointNoLocal(point);

  if (!maybeExists.has_value()) {
    llvm::outs() << "Printing toVec layout codegen for debugging:\n";
    auto result = generateLoopNestAsCStr(toVecLayout);
    if (failed(result)) {
      llvm::outs() << "Failed to generate code for actual relation.\n";
    } else {
      std::string actualCode = result.value();
      llvm::outs() << actualCode << "\n";
    }
    llvm::outs().flush();

    debugFailure(expected, actual);
    FAIL() << "Expected point not found in actual relation.";
  }
  EXPECT_TRUE(maybeExists.has_value());

  // Spot check some points
  std::vector<std::pair<int, int>> expectedDomain = {
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
  };
  std::vector<std::pair<int, int>> expectedRange = {
      {0, 5}, {0, 6}, {0, 7}, {0, 0}, {7, 1}, {0, 1}, {1, 1}, {2, 1},
  };

  for (const auto &[domain, range] : llvm::zip(expectedDomain, expectedRange)) {
    const auto &[row, col] = domain;
    const auto &[ct, slot] = range;
    auto maybeExists = actual.containsPointNoLocal({row, col, ct, slot});
    if (!maybeExists.has_value()) {
      llvm::outs() << "Failed to find point (" << row << ", " << col << ", "
                   << ct << ", " << slot << ") in actual relation.\n";
      debugFailure(expected, actual);
      FAIL() << "Expected point not found in actual relation.";
    }
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
