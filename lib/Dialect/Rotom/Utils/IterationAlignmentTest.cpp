#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/IterationAlignment.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::OpSpec;
using rotom::RotomDialect;

//===----------------------------------------------------------------------===//
// Oracle: alignment decided from what the layouts mean (their ISL relations),
// independent of the structural code under test.
//===----------------------------------------------------------------------===//

using Position = std::pair<int64_t, int64_t>;  // (ct, slot)

// The operand index held at every occupied position, or nullopt if the layout
// does not lower or names an axis beyond `rank`.
std::optional<std::map<Position, std::vector<int64_t>>> placements(
    LayoutAttr layout, int64_t rank) {
  FailureOr<std::string> isl =
      rotom::RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  if (failed(isl)) return std::nullopt;
  auto relation = getIntegerRelationFromIslStr(*isl);
  if (failed(relation)) return std::nullopt;
  // The relation's domain variables are the traversed axes, ascending.
  std::vector<int64_t> axes;
  for (Attribute attr : layout.getDims()) {
    auto d = cast<DimAttr>(attr);
    if (!d.isGap() && !d.isReplicate() && !llvm::is_contained(axes, d.getDim()))
      axes.push_back(d.getDim());
  }
  llvm::sort(axes);
  if (relation->getNumDomainVars() != axes.size()) return std::nullopt;
  PointPairCollector collector(axes.size(), /*rangeDims=*/2);
  enumeratePoints(*relation, collector);
  std::map<Position, std::vector<int64_t>> out;
  for (const auto& [domain, range] : collector.points) {
    std::vector<int64_t> index(rank, 0);
    for (size_t a = 0; a < axes.size(); ++a) {
      if (axes[a] >= rank) return std::nullopt;
      index[axes[a]] = domain[a];
    }
    out[{range[0], range[1]}] = index;
  }
  return out;
}

int64_t axisExtent(LayoutAttr layout, int64_t axis) {
  return rotom::axisPieces(layout.getDims(), axis).extent;
}

// Aligned iff (1) no position pairs elements that disagree on a shared
// iteration index and (2) every point of the iteration space is computed at
// some position.
bool oracleAligned(const OpSpec& spec, LayoutAttr lhs, LayoutAttr rhs) {
  auto l = placements(lhs, spec.lhsAxisToIter.size());
  auto r = placements(rhs, spec.rhsAxisToIter.size());
  if (!l || !r) return false;
  std::vector<int64_t> extent(spec.numIters, 1);
  std::vector<bool> extentSet(spec.numIters, false);
  auto setExtents = [&](LayoutAttr layout, ArrayRef<int64_t> axisToIter) {
    for (auto [axis, iter] : llvm::enumerate(axisToIter)) {
      int64_t e = axisExtent(layout, axis);
      if (extentSet[iter] && extent[iter] != e) return false;
      extent[iter] = e;
      extentSet[iter] = true;
    }
    return true;
  };
  if (!setExtents(lhs, spec.lhsAxisToIter) ||
      !setExtents(rhs, spec.rhsAxisToIter)) {
    return false;
  }
  std::set<std::vector<int64_t>> covered;
  for (const auto& [pos, lhsIndex] : *l) {
    auto it = r->find(pos);
    if (it == r->end()) continue;
    std::vector<int64_t> point(spec.numIters, -1);
    auto assign = [&](const std::vector<int64_t>& index,
                      ArrayRef<int64_t> axisToIter) {
      for (auto [axis, iter] : llvm::enumerate(axisToIter)) {
        if (point[iter] >= 0 && point[iter] != index[axis]) return false;
        point[iter] = index[axis];
      }
      return true;
    };
    if (!assign(lhsIndex, spec.lhsAxisToIter) ||
        !assign(it->second, spec.rhsAxisToIter)) {
      return false;
    }
    covered.insert(point);
  }
  int64_t total = 1;
  for (int64_t e : extent) total *= e;
  return static_cast<int64_t>(covered.size()) == total;
}

class IterationAlignmentTest : public ::testing::Test {
 protected:
  IterationAlignmentTest() { context.loadDialect<RotomDialect>(); }

  DimAttr dim(int64_t dim, int64_t size, int64_t stride = 1) {
    return DimAttr::get(&context, dim, size, stride);
  }

  LayoutAttr layout(ArrayRef<DimAttr> dims, int64_t n,
                    ArrayRef<int64_t> rolls = {}) {
    return LayoutAttr::getCanonical(&context, dims, n, rolls);
  }

  // Checks the structural answer and the oracle agree, then returns it.
  bool aligned(const OpSpec& spec, LayoutAttr lhs, LayoutAttr rhs) {
    bool structural = rotom::isAligned(spec, lhs, rhs);
    EXPECT_EQ(structural, oracleAligned(spec, lhs, rhs))
        << "structural and oracle disagree";
    return structural;
  }

  MLIRContext context;
};

//===----------------------------------------------------------------------===//
// OpSpec
//===----------------------------------------------------------------------===//

TEST_F(IterationAlignmentTest, MatmulSpecDerivesReductionAndResultAxes) {
  using V = SmallVector<int64_t>;
  struct Case {
    int64_t lhsRank, rhsRank;
    V lhs, rhs, result;
  };
  // Iteration indices: batch..., i, j, k.
  for (const Case& c : {
           Case{2, 2, {0, 2}, {2, 1}, {0, 1}},
           Case{2, 1, {0, 1}, {1}, {0}},
           Case{1, 2, {1}, {1, 0}, {0}},
           Case{1, 1, {0}, {0}, {}},
           Case{3, 3, {0, 1, 3}, {0, 3, 2}, {0, 1, 2}},
           Case{3, 2, {0, 1, 3}, {3, 2}, {0, 1, 2}},
       }) {
    OpSpec spec = OpSpec::matmul(c.lhsRank, c.rhsRank);
    EXPECT_TRUE(succeeded(spec.verify()));
    EXPECT_EQ(spec.lhsAxisToIter, c.lhs);
    EXPECT_EQ(spec.rhsAxisToIter, c.rhs);
    EXPECT_EQ(spec.resultAxisToIter, c.result);
    // Exactly the contraction index is reduced.
    for (int64_t it = 0; it < spec.numIters; ++it) {
      EXPECT_EQ(spec.isReduced(it), it == spec.numIters - 1);
    }
  }
}

TEST_F(IterationAlignmentTest, SpecVerifierRejectsInconsistentSpecs) {
  OpSpec spec = OpSpec::matmul(2, 2);
  spec.rhsAxisToIter.push_back(7);  // out of range
  EXPECT_TRUE(failed(spec.verify()));
  OpSpec dup = OpSpec::elementwise(2);
  dup.lhsAxisToIter = {0, 0};
  EXPECT_TRUE(failed(dup.verify()));
}

//===----------------------------------------------------------------------===//
// Alignment: cases ported from LayoutAlignmentTest
//===----------------------------------------------------------------------===//

TEST_F(IterationAlignmentTest, MatmulOperands) {
  OpSpec spec = OpSpec::matmul(2, 2);
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4)}, 16);
  EXPECT_TRUE(aligned(spec, lhs, layout({dim(0, 4), dim(-1, 4)}, 16)));
  EXPECT_FALSE(aligned(spec, lhs, layout({dim(-1, 4), dim(0, 4)}, 16)));
}

TEST_F(IterationAlignmentTest, MatVecOperands) {
  OpSpec spec = OpSpec::matmul(2, 1);
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4)}, 16);
  EXPECT_TRUE(aligned(spec, lhs, layout({dim(0, 4), dim(-1, 4)}, 16)));
  EXPECT_FALSE(aligned(spec, lhs, layout({dim(-1, 4), dim(0, 4)}, 16)));
}

TEST_F(IterationAlignmentTest, BlockMatmulOperands) {
  OpSpec spec = OpSpec::matmul(3, 3);
  LayoutAttr lhs = layout({dim(0, 2), dim(2, 4), dim(1, 4)}, 32);
  EXPECT_TRUE(
      aligned(spec, lhs, layout({dim(0, 2), dim(1, 4), dim(-1, 4)}, 32)));
  EXPECT_FALSE(
      aligned(spec, lhs, layout({dim(-1, 2), dim(1, 4), dim(0, 4)}, 32)));
}

TEST_F(IterationAlignmentTest, ContractionRollMustMatch) {
  OpSpec spec = OpSpec::matmul(2, 2);
  LayoutAttr lhsDiag = layout({dim(1, 4), dim(0, 4)}, 16, {0, 1});
  LayoutAttr rhsDiag = layout({dim(0, 4), dim(-1, 4)}, 16, {0, 1});
  LayoutAttr rhsPlain = layout({dim(0, 4), dim(-1, 4)}, 16);
  EXPECT_TRUE(aligned(spec, lhsDiag, rhsDiag));
  EXPECT_FALSE(aligned(spec, lhsDiag, rhsPlain));
}

TEST_F(IterationAlignmentTest, RollOntoReplicationIsExempt) {
  OpSpec spec = OpSpec::matmul(2, 2);
  LayoutAttr lhsRolled = layout({dim(1, 4), dim(0, 4)}, 16, {1, 0});
  EXPECT_TRUE(aligned(spec, lhsRolled, layout({dim(0, 4), dim(-1, 4)}, 16)));
}

// Equivalent spellings lift to the same digits, so no merge step is needed.
TEST_F(IterationAlignmentTest, SplitAndMergedSpellingsAlign) {
  OpSpec spec = OpSpec::elementwise(1);
  LayoutAttr whole = layout({dim(0, 16)}, 16);
  LayoutAttr split = layout({dim(0, 2, /*stride=*/8), dim(0, 8)}, 16);
  EXPECT_TRUE(aligned(spec, whole, split));
}

//===----------------------------------------------------------------------===//
// Alignment: cases the current LayoutAlignment gets wrong
//===----------------------------------------------------------------------===//

// LayoutAlignment accepts [0:2:1][0:2:2][R:4] here (stride order inside a
// multi-piece run is unchecked); the digit form compares bit places.
TEST_F(IterationAlignmentTest, StrideOrderWithinSplitAxis) {
  OpSpec spec = OpSpec::matmul(2, 1);
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4)}, 16);
  EXPECT_TRUE(
      aligned(spec, lhs, layout({dim(0, 2, 2), dim(0, 2, 1), dim(-1, 4)}, 16)));
  EXPECT_FALSE(
      aligned(spec, lhs, layout({dim(0, 2, 1), dim(0, 2, 2), dim(-1, 4)}, 16)));
}

// LayoutAlignment indexes rhsToLhs[1] out of bounds here.
TEST_F(IterationAlignmentTest, AxisOutsideSpecIsRejected) {
  OpSpec spec = OpSpec::matmul(2, 1);
  LayoutAttr lhs = layout({dim(-1, 4), dim(0, 4)}, 16);
  LayoutAttr rhs = layout({dim(1, 4), dim(0, 4)}, 16);
  EXPECT_FALSE(rotom::isAligned(spec, lhs, rhs));
}

// The two sides split differently ([R:16] faces two i pieces) and the lhs
// roll names a piece position the rhs does not have. LayoutAlignment's
// rollExempt indexes the rhs piece list with the lhs position.
TEST_F(IterationAlignmentTest, RollOnSplitAxisFacingReplication) {
  OpSpec spec = OpSpec::matmul(2, 2);
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4, 4), dim(0, 4, 1)}, 64,
                          /*rolls=*/{2, 0});
  LayoutAttr rhs = layout({dim(0, 4), dim(-1, 16)}, 64);
  EXPECT_TRUE(aligned(spec, lhs, rhs));
}

// Exhaustive check against the oracle: every 2x2-digit ordering of A(4x4)
// against every ordering of x(4) with its 4-way broadcast, at n = 16.
TEST_F(IterationAlignmentTest, MatVecAgreesWithOracleExhaustively) {
  OpSpec spec = OpSpec::matmul(2, 1);
  // Each digit is (axis, stride) of a size-2 piece; -1 is replication.
  std::vector<std::pair<int64_t, int64_t>> aDigits = {
      {0, 2}, {0, 1}, {1, 2}, {1, 1}};
  std::vector<std::pair<int64_t, int64_t>> xDigits = {
      {0, 2}, {0, 1}, {-1, 1}, {-1, 1}};
  auto build = [&](const std::vector<std::pair<int64_t, int64_t>>& ds) {
    SmallVector<DimAttr> dims;
    for (auto [axis, stride] : ds) dims.push_back(dim(axis, 2, stride));
    return layout(dims, 16);
  };
  llvm::sort(aDigits);
  llvm::sort(xDigits);
  int numAligned = 0, numCases = 0;
  do {
    LayoutAttr lhs = build(aDigits);
    auto xs = xDigits;
    do {
      LayoutAttr rhs = build(xs);
      numAligned += aligned(spec, lhs, rhs);
      ++numCases;
    } while (std::next_permutation(xs.begin(), xs.end()));
  } while (std::next_permutation(aDigits.begin(), aDigits.end()));
  EXPECT_EQ(numCases, 24 * 12);
  // Exactly one x ordering fits each A ordering.
  EXPECT_EQ(numAligned, 24);
}

//===----------------------------------------------------------------------===//
// Output layout
//===----------------------------------------------------------------------===//

TEST_F(IterationAlignmentTest, MatmulOutputConsumesTheSummationIndex) {
  OpSpec spec = OpSpec::matmul(2, 2);
  LayoutAttr lhs = layout({dim(-1, 4), dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr rhs = layout({dim(1, 4), dim(0, 4), dim(-1, 4)}, 16);
  auto pair = rotom::align(spec, lhs, rhs);
  ASSERT_TRUE(succeeded(pair));
  auto out = rotom::outputLayout(spec, *pair);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(*out, layout({dim(1, 4), dim(-2, 4), dim(0, 4)}, 16));

  // j rolled by the replication facing i: the roll survives as j by i.
  LayoutAttr rolledRhs =
      layout({dim(1, 4), dim(0, 4), dim(-1, 4)}, 16, /*rolls=*/{0, 2});
  auto rolledPair = rotom::align(spec, lhs, rolledRhs);
  ASSERT_TRUE(succeeded(rolledPair));
  auto rolledOut = rotom::outputLayout(spec, *rolledPair);
  ASSERT_TRUE(succeeded(rolledOut));
  EXPECT_EQ(*rolledOut,
            layout({dim(1, 4), dim(-2, 4), dim(0, 4)}, 16, /*rolls=*/{0, 2}));
}

// The reference diagonal matmul: k on the ciphertexts is summed away, and the
// rhs's j-by-i roll is what remains.
TEST_F(IterationAlignmentTest, DiagonalMatmulYieldsOneRolledCiphertext) {
  OpSpec spec = OpSpec::matmul(2, 2);
  LayoutAttr lhs = layout({dim(1, 4), dim(-1, 4), dim(0, 4)}, 16, {0, 1});
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4), dim(-1, 4)}, 16, {0, 1, 1, 2});
  auto pair = rotom::align(spec, lhs, rhs);
  ASSERT_TRUE(succeeded(pair));
  auto out = rotom::outputLayout(spec, *pair);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(*out, layout({dim(1, 4), dim(0, 4)}, 16, /*rolls=*/{0, 1}));
}

// LayoutAlignment turns the paired batch axis into a gap here.
TEST_F(IterationAlignmentTest, BlockMatmulOutputKeepsTheBatchAxis) {
  OpSpec spec = OpSpec::matmul(3, 3);
  LayoutAttr lhs = layout({dim(0, 2), dim(2, 4), dim(1, 4)}, 32);
  LayoutAttr rhs = layout({dim(0, 2), dim(1, 4), dim(-1, 4)}, 32);
  auto pair = rotom::align(spec, lhs, rhs);
  ASSERT_TRUE(succeeded(pair));
  auto out = rotom::outputLayout(spec, *pair);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(*out, layout({dim(0, 2), dim(-2, 4), dim(1, 4)}, 32));
}

// LayoutAlignment keeps the rhs's axis id for j (1), colliding with C's i.
// C[b, i, j] numbers j as axis 2.
TEST_F(IterationAlignmentTest, BatchedTimesPlainOutputRenumbersAxes) {
  OpSpec spec = OpSpec::matmul(3, 2);
  // A(b=0, i=1, k=2), B(k=0, j=1).
  LayoutAttr lhs = layout({dim(-1, 2), dim(0, 2), dim(2, 2), dim(1, 2)}, 16);
  LayoutAttr rhs = layout({dim(1, 2), dim(-1, 2), dim(0, 2), dim(-1, 2)}, 16);
  auto pair = rotom::align(spec, lhs, rhs);
  ASSERT_TRUE(succeeded(pair));
  auto out = rotom::outputLayout(spec, *pair);
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ(*out, layout({dim(2, 2), dim(0, 2), dim(-2, 2), dim(1, 2)}, 16));
}

// A roll that shifts by the summed index has no output layout; it is refused
// instead of silently dropped.
TEST_F(IterationAlignmentTest, OutputRefusesRollByReducedIndex) {
  OpSpec spec = OpSpec::matmul(2, 2);
  // i (exempt: B lacks i) rolled by k, then k summed in the slots.
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4)}, 16, /*rolls=*/{1, 0});
  LayoutAttr rhs = layout({dim(0, 4), dim(-1, 4)}, 16);
  auto pair = rotom::align(spec, lhs, rhs);
  ASSERT_TRUE(succeeded(pair));
  EXPECT_TRUE(failed(rotom::outputLayout(spec, *pair)));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
