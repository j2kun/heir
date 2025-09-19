#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ADT/FrozenVector.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/Graph/Graph.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"          // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

#define DEBUG_TYPE "implement-shift-network"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_IMPLEMENTSHIFTNETWORK
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

SmallVector<int64_t> defaultShiftOrder(int64_t n) {
  SmallVector<int64_t> result;
  int64_t maxLog2 = APInt(64, n).getActiveBits();
  if (isPowerOfTwo(n)) maxLog2 -= 1;
  for (int64_t i = 0; i < maxLog2; i++) result.push_back(1 << i);
  return result;
}

// Convert an input->output index mapping to a canonical left-shift amount for
// a given tensor size.
// Example: 1 -> 13 with a 64-size tensor should produce a rotation of 52
// Example: 13 -> 1 with a 64-size tensor should produce a rotation of 12
inline int64_t normalizeShift(int64_t input, int64_t output,
                              int64_t tensorSize) {
  int64_t shift = (output - input) % tensorSize;
  shift = -shift;  // Account for leftward rotations
  if (shift < 0) {
    shift += tensorSize;
  }
  return shift;
}

int64_t ShiftStrategy::getVirtualShift(const CtSlot& source,
                                       const CtSlot& target) const {
  int64_t sourceIndex = source.ct * ciphertextSize + source.slot;
  int64_t targetIndex = target.ct * ciphertextSize + target.slot;
  return normalizeShift(sourceIndex, targetIndex, virtualCiphertextSize);
}

SmallVector<ShiftRound> ShiftStrategy::evaluate(const Mapping& mapping) const {
  // First compute the virtual shifts needed for each source slot
  SmallVector<SourceShift> sourceShifts;
  sourceShifts.reserve(mapping.size());
  for (const MappingEntry& entry : mapping) {
    int64_t shift = getVirtualShift(entry.source, entry.target);
    SmallVector<int64_t> neededShifts;
    sourceShifts.push_back({entry.source, shift});
  }

  // Compute the corresponding table of positions after each rotation,
  // akin to the table in Figure 3 of the Vos-Vos-Erkin paper, including the
  // first column of values that have not yet been rotated.
  SmallVector<ShiftRound> rounds;
  rounds.reserve(shiftOrder.size() + 1);
  ShiftRound initialRound;
  for (const SourceShift& ss : sourceShifts) {
    initialRound.positions[ss] = ss.source;
    initialRound.rotationAmount = 0;
  }
  rounds.push_back(initialRound);

  for (auto rotationAmount : shiftOrder) {
    auto lastRoundPositions = rounds.back().positions;
    DenseMap<SourceShift, CtSlot> currentRoundPosns;

    for (const SourceShift& key : sourceShifts) {
      assert(lastRoundPositions.contains(key) &&
             "Expected to find source in last round positions");
      CtSlot currentPos = lastRoundPositions[key];
      int64_t currentVirtualSlot =
          currentPos.ct * ciphertextSize + currentPos.slot;

      CtSlot nextPosition = currentPos;
      if (rotationAmount & key.shift) {
        currentVirtualSlot =
            (currentVirtualSlot + rotationAmount) % virtualCiphertextSize;
        nextPosition = CtSlot{currentVirtualSlot / ciphertextSize,
                              currentVirtualSlot % ciphertextSize};
      }
      currentRoundPosns[key] = nextPosition;
    }

    rounds.push_back({currentRoundPosns, rotationAmount});
  }

  return rounds;
}

ShiftScheme VosVosErkinShiftNetworks::findShiftScheme(const Mapping& mapping) {
  if (schemeCache.count(mapping)) {
    return schemeCache[mapping];
  }

  // FIXME: try many shift orders and pick the best
  ShiftStrategy strategy(ciphertextSize, numCiphertexts, shiftOrder);
  strategy.evaluate(mapping);

  // Create a graph whose vertices are the input indices to permute, and
  // whose edges are conflicts: an edge being present means the two indices
  // cannot participate in the same rotation group.
  graph::UndirectedGraph<int64_t> conflictGraph;
  for (int64_t i = 0; i < ciphertextSize; i++) {
    conflictGraph.addVertex(i);
  }
  for (const auto& [roundNum, round] : llvm::enumerate(strategy.getRounds())) {
    if (roundNum == 0) continue;

    auto posns = round.positions;
    for (auto it1 = posns.begin(); it1 != posns.end(); ++it1) {
      for (auto it2 = std::next(it1); it2 != posns.end(); ++it2) {
        const SourceShift& ss1 = it1->first;
        const SourceShift& ss2 = it2->first;
        if (ss1.source != ss2.source && it1->second == it2->second) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Round " << roundNum << ": collision between " << "{"
                     << ss1.source.ct << "," << ss1.source.slot << "}"
                     << " and " << "{" << ss2.source.ct << ","
                     << ss2.source.slot << "}" << " at " << "{"
                     << it1->second.ct << "," << it1->second.slot << "}\n");
          conflictGraph.addEdge(ss1.source.slot, ss2.source.slot);
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Conflict graph:\n";
    for (int64_t vertex : conflictGraph.getVertices()) {
      llvm::dbgs() << "  " << vertex << ": ";
      for (int64_t neighbor : conflictGraph.edgesIncidentTo(vertex)) {
        llvm::dbgs() << neighbor << " ";
      }
      llvm::dbgs() << "\n";
    }
  });

  graph::GreedyGraphColoring<int64_t> colorer;
  std::unordered_map<int64_t, int> coloring = colorer.color(conflictGraph);

  SmallVector<RotationGroup> resultRotationGroups;
  resultRotationGroups.reserve(5);
  for (const auto& entry : coloring) {
    int64_t index = entry.first;
    int64_t color = entry.second;
    if (color >= resultRotationGroups.size()) {
      resultRotationGroups.resize(color + 1);
    }
    resultRotationGroups[color].insert(index);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Splitting mapping into rotation groups:\n";
    for (int i = 0; i < resultRotationGroups.size(); i++) {
      llvm::dbgs() << "Group " << i << ": ";
      llvm::SmallVector<int64_t> group = llvm::SmallVector<int64_t>(
          resultRotationGroups[i].begin(), resultRotationGroups[i].end());
      llvm::sort(group);
      for (int64_t index : group) {
        llvm::dbgs() << index << " ";
      }
      llvm::dbgs() << "\n";
    }
  });

  ShiftScheme scheme{resultRotationGroups, strategy};
  schemeCache[mapping] = scheme;
  return schemeCache[mapping];
}

// Create a tensor with zeros everywhere except for the indices specified in
// the input `indices` vector.
Value createMask(TypedValue<RankedTensorType> tensor,
                 const SmallVector<int64_t>& indices, IRRewriter& rewriter) {
  auto elementType = tensor.getType().getElementType();
  SmallVector<Attribute> maskAttrs(tensor.getType().getDimSize(0),
                                   rewriter.getIntegerAttr(elementType, 0));
  for (int64_t index : indices) {
    maskAttrs[index] = rewriter.getIntegerAttr(elementType, 1);
  }

  auto denseAttr = DenseElementsAttr::get(tensor.getType(), maskAttrs);
  auto constant =
      arith::ConstantOp::create(rewriter, tensor.getLoc(), denseAttr);
  return constant.getResult();
}

SmallVector<Value> rotateOneGroup(ArrayRef<Value> initialCiphertexts,
                                  ArrayRef<SourceShift> sourceShifts,
                                  ArrayRef<ShiftRound> rounds,
                                  const RotationGroup& group,
                                  IRRewriter& rewriter) {
  SmallVector<Value> results;

  return results;
}

LogicalResult convertPermuteOp(PermuteOp op,
                               VosVosErkinShiftNetworks& shiftNetworks,
                               int64_t ciphertextSize) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  IRRewriter rewriter(op.getContext());
  RankedTensorType tensorTy = op.getInput().getType();
  int64_t numCiphertexts = tensorTy.getDimSize(0);

  // Populate the mapping with (source, target) pairs
  // This require enumerating over the relation for the op
  auto mappingAttr = dyn_cast<NewLayoutAttr>(op.getPermutation());
  if (!mappingAttr) {
    return failure();
  }
  Mapping mapping;
  PointPairCollector collector(2, 2);
  enumeratePoints(mappingAttr.getIntgerRelation(), collector);

  // Put the data from collector into Mapping. Probably can be more efficient
  // here by avoiding a copy and making a custom PointPairCollector that
  // directly adds to mapping.
  for (const auto& [source, target] : collector.points) {
    CtSlot sourceSlot{source[0], source[1]};
    CtSlot targetSlot{target[0], target[1]};
    mapping.add(sourceSlot, targetSlot);
  }

  ShiftScheme scheme = shiftNetworks.findShiftScheme(mapping);
  auto rotationGroups = scheme.rotationGroups;

  SmallVector<Value> reconstructedTensorResults;

  assert(!rotationGroups.empty() &&
         "Shift network must have at least one group");

  rewriter.setInsertionPointAfter(op);

  // Decompose the tensor of ciphertexts into individual values
  SmallVector<Value> ciphertexts;
  for (int64_t i = 0; i < numCiphertexts; i++)
    ciphertexts.push_back(
        rewriter.create<tensor::ExtractOp>(op.getLoc(), op.getInput(), i));

  // Create arith.constant zero initializers for the results
  SmallVector<Value> resultCiphertexts;
  for (int64_t i = 0; i < numCiphertexts; i++)
    resultCiphertexts.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(tensorTy.getElementType())));

  std::optional<Value> result = std::nullopt;
  [[maybe_unused]] int groupIndex = 0;
  for (const RotationGroup& group : rotationGroups) {
    LLVM_DEBUG(llvm::dbgs()
               << "Implementing rotations for group " << groupIndex++ << "\n");

    // Re-compute the subset of SourceShifts needed for this group
    SmallVector<SourceShift> sourceShifts;
    for (const MappingEntry& entry : mapping) {
      if (group.contains(entry.source)) {
        int64_t shift =
            scheme.strategy.getVirtualShift(entry.source, entry.target);
        sourceShifts.push_back({entry.source, shift});
      }
    }

    SmallVector<Value> perGroupResult =
        rotateOneGroup(ciphertexts, sourceShifts, scheme.strategy.getRounds(),
                       group, rewriter);

    reconstructedTensorResults.push_back(tensor::FromElementsOp::create(
        rewriter, tensorTy, perGroupResult, op.getLoc()));
  }

  //   # add all the results together
  //   final_result = [Ciphertext([0] * len(x)) for x in input]
  //   for result in group_results:
  //       for i, ct in enumerate(result):
  //           if ct:
  //               final_result[i] += ct

  rewriter.replaceOp(op, result.value());
  return success();
}

struct ImplementShiftNetwork
    : impl::ImplementShiftNetworkBase<ImplementShiftNetwork> {
  using ImplementShiftNetworkBase::ImplementShiftNetworkBase;

  void runOnOperation() override {
    VosVosErkinShiftNetworks shiftNetworks{ciphertextSize};

    getOperation()->walk([&](PermuteOp op) {
      if (failed(convertPermuteOp(op, shiftNetworks, ciphertextSize))) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
