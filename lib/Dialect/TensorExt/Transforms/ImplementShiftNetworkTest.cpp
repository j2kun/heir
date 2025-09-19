#include <iostream>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/TestingUtils.h"

namespace mlir {
namespace heir {
namespace tensor_ext {
namespace {

using kernel::LiteralValue;

std::vector<std::vector<int>> manuallyApplyMapping(
    const Mapping& mapping, const std::vector<std::vector<int>>& input,
    int64_t ctSize) {
  std::vector<std::vector<int>> output(input.size(),
                                       std::vector<int>(ctSize, 0));
  for (const auto& entry : mapping) {
    output[entry.target.ct][entry.target.slot] =
        input[entry.source.ct][entry.source.slot];
  }
  return output;
}

void simulateShiftNetwork(const Mapping& mapping, const ShiftScheme& scheme,
                          int64_t numCiphertexts, int64_t ciphertextSize) {
  // print the rotation groups
  std::cout << "Rotation groups:\n";
  for (const auto& row : scheme.rotationGroups) {
    for (const auto& slot : row) {
      std::cout << "(" << slot.ct << "," << slot.slot << ") ";
    }
    std::cout << "\n";
  }

  SmallVector<LiteralValue> inputLeaves;
  std::vector<std::vector<int>> input;
  input.reserve(numCiphertexts);
  inputLeaves.reserve(numCiphertexts);
  // row-major values as input
  for (int64_t i = 0; i < numCiphertexts; i++) {
    std::vector<int> oneInput(ciphertextSize);
    for (int64_t j = 0; j < ciphertextSize; j++) {
      oneInput[j] = i * ciphertextSize + j;
    }
    input.push_back(oneInput);
    inputLeaves.push_back(LiteralValue(oneInput));
  }

  // print the input
  std::cout << "Input:\n";
  for (const auto& row : input) {
    for (const auto& val : row) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  auto expected = manuallyApplyMapping(mapping, input, ciphertextSize);
  auto dag =
      implementShiftNetwork(inputLeaves, mapping, scheme, ciphertextSize);
  std::vector<LiteralValue> actual = multiEvalKernel(dag);

  std::vector<std::vector<int>> combinedActual;
  combinedActual.reserve(numCiphertexts);
  for (const LiteralValue& val : actual) {
    combinedActual.push_back(std::get<std::vector<int>>(val.getTensor()));
  }

  EXPECT_EQ(combinedActual, expected);
}

TEST(ImplementShiftNetworkTest, TestTrivial) {
  int64_t numCts = 1;
  int64_t ctSize = 8;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 0));
  VosVosErkinShiftNetworks shiftNetworks;
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 1);
}

TEST(ImplementShiftNetworkTest, TestFig3) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 13));
  mapping.add(CtSlot(0, 1), CtSlot(0, 8));
  mapping.add(CtSlot(0, 2), CtSlot(0, 4));
  mapping.add(CtSlot(0, 3), CtSlot(0, 0));
  mapping.add(CtSlot(0, 4), CtSlot(0, 11));
  mapping.add(CtSlot(0, 5), CtSlot(0, 7));
  mapping.add(CtSlot(0, 6), CtSlot(0, 14));
  mapping.add(CtSlot(0, 7), CtSlot(0, 5));
  mapping.add(CtSlot(0, 8), CtSlot(0, 15));
  mapping.add(CtSlot(0, 9), CtSlot(0, 3));
  mapping.add(CtSlot(0, 10), CtSlot(0, 12));
  mapping.add(CtSlot(0, 11), CtSlot(0, 6));
  mapping.add(CtSlot(0, 12), CtSlot(0, 10));
  mapping.add(CtSlot(0, 13), CtSlot(0, 2));
  mapping.add(CtSlot(0, 14), CtSlot(0, 9));
  mapping.add(CtSlot(0, 15), CtSlot(0, 1));
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  EXPECT_EQ(scheme.rotationGroups.size(), 3);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}

TEST(ImplementShiftNetworkTest, TestFullReplication) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 0));
  mapping.add(CtSlot(0, 0), CtSlot(0, 1));
  mapping.add(CtSlot(0, 0), CtSlot(0, 2));
  mapping.add(CtSlot(0, 0), CtSlot(0, 3));
  mapping.add(CtSlot(0, 0), CtSlot(0, 4));
  mapping.add(CtSlot(0, 0), CtSlot(0, 5));
  mapping.add(CtSlot(0, 0), CtSlot(0, 6));
  mapping.add(CtSlot(0, 0), CtSlot(0, 7));
  mapping.add(CtSlot(0, 0), CtSlot(0, 8));
  mapping.add(CtSlot(0, 0), CtSlot(0, 9));
  mapping.add(CtSlot(0, 0), CtSlot(0, 10));
  mapping.add(CtSlot(0, 0), CtSlot(0, 11));
  mapping.add(CtSlot(0, 0), CtSlot(0, 12));
  mapping.add(CtSlot(0, 0), CtSlot(0, 13));
  mapping.add(CtSlot(0, 0), CtSlot(0, 14));
  mapping.add(CtSlot(0, 0), CtSlot(0, 15));
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  EXPECT_EQ(scheme.rotationGroups.size(), 1);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}

TEST(ImplementShiftNetworkTest, TestTwoReplication) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 14), CtSlot(0, 0));
  mapping.add(CtSlot(0, 14), CtSlot(0, 1));
  mapping.add(CtSlot(0, 14), CtSlot(0, 2));
  mapping.add(CtSlot(0, 14), CtSlot(0, 3));
  mapping.add(CtSlot(0, 14), CtSlot(0, 4));
  mapping.add(CtSlot(0, 14), CtSlot(0, 5));
  mapping.add(CtSlot(0, 14), CtSlot(0, 6));
  mapping.add(CtSlot(0, 14), CtSlot(0, 7));
  mapping.add(CtSlot(0, 15), CtSlot(0, 8));
  mapping.add(CtSlot(0, 15), CtSlot(0, 9));
  mapping.add(CtSlot(0, 15), CtSlot(0, 10));
  mapping.add(CtSlot(0, 15), CtSlot(0, 11));
  mapping.add(CtSlot(0, 15), CtSlot(0, 12));
  mapping.add(CtSlot(0, 15), CtSlot(0, 13));
  mapping.add(CtSlot(0, 15), CtSlot(0, 14));
  mapping.add(CtSlot(0, 15), CtSlot(0, 15));
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  EXPECT_EQ(scheme.rotationGroups.size(), 2);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}

TEST(ImplementShiftNetworkTest, TestTwoReplicationAlternateShiftOrder) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 14), CtSlot(0, 0));
  mapping.add(CtSlot(0, 14), CtSlot(0, 1));
  mapping.add(CtSlot(0, 14), CtSlot(0, 2));
  mapping.add(CtSlot(0, 14), CtSlot(0, 3));
  mapping.add(CtSlot(0, 14), CtSlot(0, 4));
  mapping.add(CtSlot(0, 14), CtSlot(0, 5));
  mapping.add(CtSlot(0, 14), CtSlot(0, 6));
  mapping.add(CtSlot(0, 14), CtSlot(0, 7));
  mapping.add(CtSlot(0, 15), CtSlot(0, 8));
  mapping.add(CtSlot(0, 15), CtSlot(0, 9));
  mapping.add(CtSlot(0, 15), CtSlot(0, 10));
  mapping.add(CtSlot(0, 15), CtSlot(0, 11));
  mapping.add(CtSlot(0, 15), CtSlot(0, 12));
  mapping.add(CtSlot(0, 15), CtSlot(0, 13));
  mapping.add(CtSlot(0, 15), CtSlot(0, 14));
  mapping.add(CtSlot(0, 15), CtSlot(0, 15));
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping, {8, 4, 2, 1});
  EXPECT_EQ(scheme.rotationGroups.size(), 1);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}

TEST(ImplementShiftNetworkTest, TestSwapTwoCiphertexts) {
  int64_t numCts = 2;
  int64_t ctSize = 4;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(1, 0));
  mapping.add(CtSlot(0, 1), CtSlot(1, 1));
  mapping.add(CtSlot(0, 2), CtSlot(1, 2));
  mapping.add(CtSlot(0, 3), CtSlot(1, 3));
  mapping.add(CtSlot(1, 0), CtSlot(0, 0));
  mapping.add(CtSlot(1, 1), CtSlot(0, 1));
  mapping.add(CtSlot(1, 2), CtSlot(0, 2));
  mapping.add(CtSlot(1, 3), CtSlot(0, 3));
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  EXPECT_EQ(scheme.rotationGroups.size(), 1);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}

TEST(ImplementShiftNetworkTest, TestReorderThreeCiphertexts) {
  int64_t numCts = 3;
  int64_t ctSize = 4;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(2, 0));
  mapping.add(CtSlot(0, 1), CtSlot(2, 1));
  mapping.add(CtSlot(0, 2), CtSlot(2, 2));
  mapping.add(CtSlot(0, 3), CtSlot(2, 3));
  mapping.add(CtSlot(1, 0), CtSlot(0, 0));
  mapping.add(CtSlot(1, 1), CtSlot(0, 1));
  mapping.add(CtSlot(1, 2), CtSlot(0, 2));
  mapping.add(CtSlot(1, 3), CtSlot(0, 3));
  mapping.add(CtSlot(2, 0), CtSlot(1, 0));
  mapping.add(CtSlot(2, 1), CtSlot(1, 1));
  mapping.add(CtSlot(2, 2), CtSlot(1, 2));
  mapping.add(CtSlot(2, 3), CtSlot(1, 3));
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  EXPECT_EQ(scheme.rotationGroups.size(), 1);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}

TEST(ImplementShiftNetworkTest, TestSingleRotSplit) {
  int64_t numCts = 3;
  int64_t ctSize = 4;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 1));
  mapping.add(CtSlot(0, 1), CtSlot(0, 2));
  mapping.add(CtSlot(0, 2), CtSlot(0, 3));
  mapping.add(CtSlot(0, 3), CtSlot(1, 0));
  mapping.add(CtSlot(1, 0), CtSlot(1, 1));
  mapping.add(CtSlot(1, 1), CtSlot(1, 2));
  mapping.add(CtSlot(1, 2), CtSlot(1, 3));
  mapping.add(CtSlot(1, 3), CtSlot(2, 0));
  mapping.add(CtSlot(2, 0), CtSlot(2, 1));
  mapping.add(CtSlot(2, 1), CtSlot(2, 2));
  mapping.add(CtSlot(2, 2), CtSlot(2, 3));
  mapping.add(CtSlot(2, 3), CtSlot(0, 0));
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  EXPECT_EQ(scheme.rotationGroups.size(), 1);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}

}  // namespace
}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
