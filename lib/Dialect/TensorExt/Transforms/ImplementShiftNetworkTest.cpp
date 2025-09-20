#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

namespace mlir {
namespace heir {
namespace tensor_ext {
namespace {

TEST(ImplementShiftNetworkTest, TestTrivial) {
  int64_t numCts = 1;
  int64_t ctSize = 8;
  Mapping mapping;
  mapping.add(CtSlot(0, 0), CtSlot(0, 0));
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts);
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 1);
}

TEST(ImplementShiftNetworkTest, TestFig3) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping;
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
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts);
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 3);
}

TEST(ImplementShiftNetworkTest, TestFullReplication) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping;
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
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts);
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 1);
}

TEST(ImplementShiftNetworkTest, TestTwoReplication) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping;
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
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts);
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 2);
}

TEST(ImplementShiftNetworkTest, TestTwoReplicationAlternateShiftOrder) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping;
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
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts, {8, 4, 2, 1});
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 1);
}

TEST(ImplementShiftNetworkTest, TestSwapTwoCiphertexts) {
  int64_t numCts = 2;
  int64_t ctSize = 4;
  Mapping mapping;
  mapping.add(CtSlot(0, 0), CtSlot(1, 0));
  mapping.add(CtSlot(0, 1), CtSlot(1, 1));
  mapping.add(CtSlot(0, 2), CtSlot(1, 2));
  mapping.add(CtSlot(0, 3), CtSlot(1, 3));
  mapping.add(CtSlot(1, 0), CtSlot(0, 0));
  mapping.add(CtSlot(1, 1), CtSlot(0, 1));
  mapping.add(CtSlot(1, 2), CtSlot(0, 2));
  mapping.add(CtSlot(1, 3), CtSlot(0, 3));
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts);
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 1);
}

TEST(ImplementShiftNetworkTest, TestReorderThreeCiphertexts) {
  int64_t numCts = 3;
  int64_t ctSize = 4;
  Mapping mapping;
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
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts);
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 1);
}

TEST(ImplementShiftNetworkTest, TestSingleRotSplit) {
  int64_t numCts = 3;
  int64_t ctSize = 4;
  Mapping mapping;
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
  VosVosErkinShiftNetworks shiftNetworks(ctSize, numCts);
  EXPECT_EQ(shiftNetworks.findShiftScheme(mapping).rotationGroups.size(), 1);
}

}  // namespace
}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
