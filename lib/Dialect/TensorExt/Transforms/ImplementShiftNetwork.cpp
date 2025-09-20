#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Utils/Graph/Graph.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

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

void ShiftStrategy::evaluate(const Mapping& mapping) {
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
            (currentVirtualSlot - rotationAmount + virtualCiphertextSize) %
            virtualCiphertextSize;
        nextPosition = CtSlot{currentVirtualSlot / ciphertextSize,
                              currentVirtualSlot % ciphertextSize};
      }
      currentRoundPosns[key] = nextPosition;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After rotation " << rotationAmount << ":\n";
      for (const auto& [ss, pos] : currentRoundPosns) {
        llvm::dbgs() << "  (" << ss.source.ct << "," << ss.source.slot << ")["
                     << ss.shift << "] -> (" << pos.ct << "," << pos.slot << ")"
                     << "\n";
      }
    });

    rounds.push_back({currentRoundPosns, rotationAmount});
  }
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
  graph::UndirectedGraph<CtSlot> conflictGraph;
  for (const MappingEntry& entry : mapping) {
    conflictGraph.addVertex(entry.source);
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
          conflictGraph.addEdge(ss1.source, ss2.source);
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Conflict graph:\n";
    for (CtSlot vertex : conflictGraph.getVertices()) {
      llvm::dbgs() << vertex.ct << "," << vertex.slot << " <-> {";
      for (CtSlot neighbor : conflictGraph.edgesIncidentTo(vertex)) {
        llvm::dbgs() << neighbor.ct << "," << neighbor.slot << "; ";
      }
      llvm::dbgs() << "}\n";
    }
  });

  graph::GreedyGraphColoring<CtSlot> colorer;
  std::unordered_map<CtSlot, int> coloring = colorer.color(conflictGraph);

  SmallVector<RotationGroup> resultRotationGroups;
  resultRotationGroups.reserve(5);
  for (const auto& entry : coloring) {
    CtSlot source = entry.first;
    int64_t color = entry.second;
    if (color >= resultRotationGroups.size()) {
      resultRotationGroups.resize(color + 1);
    }
    resultRotationGroups[color].insert(source);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Splitting mapping into rotation groups:\n";
    for (int i = 0; i < resultRotationGroups.size(); i++) {
      llvm::dbgs() << "Group " << i << ": ";
      llvm::SmallVector<CtSlot> group = llvm::SmallVector<CtSlot>(
          resultRotationGroups[i].begin(), resultRotationGroups[i].end());
      llvm::sort(group);
      for (CtSlot source : group) {
        llvm::dbgs() << "(" << source.ct << "," << source.slot << ") ";
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
Value createMask(RankedTensorType tensorTy, const SmallVector<int64_t>& indices,
                 ImplicitLocOpBuilder& b) {
  auto elementType = tensorTy.getElementType();
  SmallVector<Attribute> maskAttrs(tensorTy.getDimSize(0),
                                   b.getIntegerAttr(elementType, 0));
  for (int64_t index : indices) {
    maskAttrs[index] = b.getIntegerAttr(elementType, 1);
  }

  auto denseAttr = DenseElementsAttr::get(tensorTy, maskAttrs);
  auto constant = arith::ConstantOp::create(b, denseAttr);
  return constant.getResult();
}

//  Apply a virtual rotation to a real list of ciphertexts.
//
//  A virtual ciphertext is a flattening of a list of ciphertexts. When this
//  materializes to a set of rotations of the real ciphertexts, we need to
//  track the movement of slots between ciphertexts, and decompose the virtual
//  rotation into a set of real rotations and extra masks.
//
//  For example, we have to deal with ciphertexts which are rotated in such a
//  way that they overlap two subsequent ciphertexts in the larger "virtual"
//  ciphertext. E.g. if we have size 8 and two slots 3, 7 are rotated left by
//  -2:
//
//   ct0: . . . x . . . y
//   ct1: . . . . . . . .
//
//  then after their rotation if the desired target for slot 7 is ct1 slot 2,
//  we have the following reality
//
//   ct0: . y . . . x . .
//   ct1: . . . . . . . .
//
//  and we need to mask the position of y to add it to ct1, while masking out x
//  to keep it with ct0.
SmallVector<std::optional<Value>> applyVirtualRotation(
    ArrayRef<Value> input, int64_t rotation,
    const SmallVector<SmallVector<int64_t>>& rotateMasks,
    ImplicitLocOpBuilder& b) {
  int64_t numCiphertexts = input.size();
  RankedTensorType ctType = cast<RankedTensorType>(input[0].getType());
  int64_t ciphertextSize = ctType.getDimSize(0);

  // We need to identify the (possibly two) target ciphertexts for each input
  // ciphertext that was rotated.
  //
  // If there is only one target---i.e., if the rotation was exactly the power
  // of two matching the ciphertext size---we can update the target with the
  // rotated ciphertexts and be done.
  if (rotation % ciphertextSize == 0) {
    SmallVector<Value> masked;
    masked.reserve(numCiphertexts);
    for (const auto& [ct, mask] : llvm::zip(input, rotateMasks)) {
      // Eagerly skip masking if possible
      auto [allZero, allOne] = allZeroAllOne(mask);
      if (allZero) {
        masked.push_back(
            arith::ConstantOp::create(b, ctType, b.getZeroAttr(ctType)));
      } else if (allOne) {
        masked.push_back(ct);
      } else {
        masked.push_back(makeAppropriatelyTypedMulOp(
                             b, b.getLoc(), ct, createMask(ctType, mask, b))
                             ->getResult(0));
      }
    }

    int64_t ciphertextShift = rotation / ciphertextSize;
    SmallVector<std::optional<Value>> result;
    result.reserve(numCiphertexts);
    // We are left-rotating, so ciphertext `source` maps to `target = source -
    // ciphertextShift`
    for (int64_t target = 0; target < numCiphertexts; target++) {
      int64_t source = target + ciphertextShift + numCiphertexts;
      source = source % numCiphertexts;
      result.push_back(masked[source]);
    }
    return result;
  }

  // If there are two targets, we need to add additional masks at the split.
  // Note we are rotating left, so the split is the slot whose rotated position
  // is zero.
  //
  // Nb., there is a choice here:
  //
  //  1. Mask first, then rotate together, then mask twice to separate the
  //     two targets.
  //  2. Split mask first, mask twice, then rotate twice.
  //
  // Option (1) requires one rotation, but three ct-pt muls (depth 2), and
  // option (2) requires two rotations, but only two ct-pt muls (depth 1). Not
  // sure which is better. This func implements (2).
  SmallVector<std::optional<Value>> results;
  results.resize(numCiphertexts);

  int64_t minSlot = 0;
  int64_t maxSlot = ciphertextSize - 1;
  int64_t boundarySlot = rotation;

  for (int64_t ctIndex = 0; ctIndex < numCiphertexts; ctIndex++) {
    const Value& ct = input[ctIndex];
    const SmallVector<int64_t>& mask = rotateMasks[ctIndex];

    // Determine the two ciphertext targets for each input ciphertext. Rotating
    // left, so ciphertext zero wraps around to numCiphertexts-1. Easiest to do
    // it in the virtual coordinate system.
    //
    // Fist compute target1 and target2, the ciphertext indices that the rotated
    // ciphertext will straddle.
    int64_t virtualN = numCiphertexts * ciphertextSize;
    int64_t minVirtual = ctIndex * ciphertextSize + minSlot;
    int64_t maxVirtual = ctIndex * ciphertextSize + maxSlot;
    int64_t minRotated = (minVirtual - rotation + virtualN) % virtualN;
    int64_t maxRotated = (maxVirtual - rotation + virtualN) % virtualN;
    int64_t target1 = minRotated / ciphertextSize;
    int64_t target2 = maxRotated / ciphertextSize;

    assert((target1 + 1) % numCiphertexts == target2 &&
           "Expected targets to be adjacent mod numCiphertexts");

    // Split each of the input masks into two masks, one for the pre-split and
    // one for the post-split.
    SmallVector<int64_t> mask1(ciphertextSize, 0);
    SmallVector<int64_t> mask2(ciphertextSize, 0);
    for (int64_t i = 0; i < ciphertextSize; i++) {
      if (i <= boundarySlot) {
        mask1[i] = mask[i];
      } else {
        mask2[i] = mask[i];
      }
    }

    // Apply the split masks to the input and rotate
    std::optional<Value> rotated1;
    {
      auto [allZero, _] = allZeroAllOne(mask1);
      if (allZero) {
        rotated1 = std::nullopt;
      } else {
        Value masked1 = makeAppropriatelyTypedMulOp(
                            b, b.getLoc(), ct, createMask(ctType, mask1, b))
                            ->getResult(0);
        rotated1 = tensor_ext::RotateOp::create(
                       b, masked1,
                       arith::ConstantOp::create(b, b.getIndexType(),
                                                 b.getIndexAttr(rotation)))
                       ->getResult(0);
      }
    }

    std::optional<Value> rotated2;
    {
      auto [allZero, _] = allZeroAllOne(mask2);
      if (allZero) {
        rotated2 = std::nullopt;
      } else {
        Value masked2 = makeAppropriatelyTypedMulOp(
                            b, b.getLoc(), ct, createMask(ctType, mask2, b))
                            ->getResult(0);
        rotated1 = tensor_ext::RotateOp::create(
                       b, masked2,
                       arith::ConstantOp::create(b, b.getIndexType(),
                                                 b.getIndexAttr(rotation)))
                       ->getResult(0);
      }
    }

    if (rotated1.has_value()) {
      if (results[target1].has_value()) {
        results[target1] = makeAppropriatelyTypedAddOp(
                               b, b.getLoc(), *results[target1], *rotated1)
                               ->getResult(0);
      } else {
        results[target1] = *rotated1;
      }
    }

    if (rotated2.has_value()) {
      if (results[target2].has_value()) {
        results[target2] = makeAppropriatelyTypedAddOp(
                               b, b.getLoc(), *results[target2], *rotated2)
                               ->getResult(0);
      } else {
        results[target2] = *rotated2;
      }
    }
  }

  return results;
}

LogicalResult convertPermuteOp(PermuteOp op,
                               VosVosErkinShiftNetworks& shiftNetworks,
                               int64_t ciphertextSize) {
  LLVM_DEBUG(llvm::dbgs() << "Converting layout op: " << op << "\n");
  ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
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
  enumeratePoints(mappingAttr.getIntegerRelation(), collector);

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

  assert(!rotationGroups.empty() &&
         "Shift network must have at least one group");

  b.setInsertionPointAfter(op);

  // Decompose the tensor of ciphertexts into individual values. This is done
  // by extracting the slice of a tensor<kxN> corresponding to one row.
  //
  // Also needed to be compatible with the ArithmeticDag interface and to
  // avoid extract slice ops in the middle of the dag.
  SmallVector<kernel::SSAValue> ciphertexts;
  for (int64_t i = 0; i < numCiphertexts; i++) {
    auto one = b.getIndexAttr(1);
    SmallVector<OpFoldResult> offsets = {b.getIndexAttr(i), b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {one,
                                       b.getIndexAttr(tensorTy.getDimSize(1))};
    SmallVector<OpFoldResult> strides = {one, one};
    auto slice = tensor::ExtractSliceOp::create(
        b, op.getLoc(),
        RankedTensorType::get({tensorTy.getDimSize(1)},
                              tensorTy.getElementType()),
        op.getInput(), offsets, sizes, strides);
    ciphertexts.push_back(kernel::SSAValue(slice.getResult()));
  }

  // Create arith.constant zero initializers for the results
  SmallVector<Value> resultCiphertexts;
  for (int64_t i = 0; i < numCiphertexts; i++)
    resultCiphertexts.push_back(b.create<arith::ConstantOp>(
        op.getLoc(), b.getZeroAttr(tensorTy.getElementType())));

  SmallVector<std::shared_ptr<kernel::ArithmeticDagNode<kernel::SSAValue>>>
      resultNodes = implementShiftNetwork(ciphertexts, mapping, scheme);

  // FIXME: add this multi-visitor
  IRMaterializingMultiVisitor visitor(b, op.getValue().getType());
  SmallVector<Value> result = visitor.visit(resultNodes);

  auto combinedResult =
      tensor::FromElementsOp::create(b, op.getLoc(), tensorTy, result);

  op.replaceAllUsesWith(combinedResult);
  op.erase();
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
