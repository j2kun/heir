#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_ROTATIONGROUPKERNEL_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_ROTATIONGROUPKERNEL_H_

#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Utils.h"

namespace mlir {
namespace heir {
namespace tensor_ext {

using kernel::AbstractValue;
using kernel::ArithmeticDagNode;

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
template <typename T>
std::enable_if_t<
    std::is_base_of<AbstractValue, T>::value,
    SmallVector<std::optional<std::shared_ptr<ArithmeticDagNode<T>>>>>
applyVirtualRotation(ArrayRef<T> input, int64_t rotation,
                     const SmallVector<SmallVector<int64_t>>& rotateMasks) {
  using NodeTy = ArithmeticDagNode<T>;
  using ValueTy = std::shared_ptr<NodeTy>;

  int64_t numCiphertexts = input.size();
  // FIXME: figure out how to get the ciphertextSize from the input
  // RankedTensorType ctType = cast<RankedTensorType>(input[0].getType());
  int64_t ciphertextSize = ctType.getDimSize(0);

  // We need to identify the (possibly two) target ciphertexts for each input
  // ciphertext that was rotated.
  //
  // If there is only one target---i.e., if the rotation was exactly the power
  // of two matching the ciphertext size---we can update the target with the
  // rotated ciphertexts and be done.
  if (rotation % ciphertextSize == 0) {
    SmallVector<ValueTy> masked;
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
    SmallVector<std::optional<ValueTy>> result;
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
  SmallVector<std::optional<ValueTy>> results;
  results.resize(numCiphertexts);

  int64_t minSlot = 0;
  int64_t maxSlot = ciphertextSize - 1;
  int64_t boundarySlot = rotation;

  for (int64_t ctIndex = 0; ctIndex < numCiphertexts; ctIndex++) {
    const ValueTy& ct = input[ctIndex];
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
    std::optional<ValueTy> rotated1;
    {
      auto [allZero, _] = allZeroAllOne(mask1);
      if (allZero) {
        rotated1 = std::nullopt;
      } else {
        ValueTy masked1 = makeAppropriatelyTypedMulOp(
                              b, b.getLoc(), ct, createMask(ctType, mask1, b))
                              ->getResult(0);
        rotated1 = tensor_ext::RotateOp::create(
                       b, masked1,
                       arith::ConstantOp::create(b, b.getIndexType(),
                                                 b.getIndexAttr(rotation)))
                       ->getResult(0);
      }
    }

    std::optional<ValueTy> rotated2;
    {
      auto [allZero, _] = allZeroAllOne(mask2);
      if (allZero) {
        rotated2 = std::nullopt;
      } else {
        ValueTy masked2 = makeAppropriatelyTypedMulOp(
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

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 SmallVector<std::shared_ptr<ArithmeticDagNode<T>>>>
rotateOneGroup(ArrayRef<T> initialCiphertexts,
               ArrayRef<SourceShift> sourceShifts, ArrayRef<ShiftRound> rounds,
               const RotationGroup& group) {
  using NodeTy = ArithmeticDagNode<T>;
  using ValueTy = std::shared_ptr<NodeTy>;

  RankedTensorType ctType =
      cast<RankedTensorType>(initialCiphertexts[0].getType());
  int64_t numCiphertexts = initialCiphertexts.size();
  int64_t ciphertextSize = ctType.getDimSize(0);
  SmallVector<ValueTy> current;
  for (const T& ct : initialCiphertexts) {
    current.push_back(NodeTy::leaf(ct));
  }

  DenseSet<int64_t> touchedCiphertexts;

  for (const auto& [roundNum, round] : llvm::enumerate(rounds)) {
    if (roundNum == 0) continue;

    // Need two masks, one to select the sources in this group that need to
    // be rotated, and one to preserve the values at fixed positions.
    SmallVector<CtSlot> rotatePositions;
    SmallVector<CtSlot> fixedPositions;
    for (const SourceShift& ss : sourceShifts) {
      if (!group.contains(ss.source)) continue;
      CtSlot currentPos = rounds[roundNum - 1].positions.at(ss);
      if (ss.shift & round.rotationAmount) {
        rotatePositions.push_back(currentPos);
      } else {
        fixedPositions.push_back(currentPos);
      }
    }

    SmallVector<SmallVector<int64_t>> fixedMasks(
        numCiphertexts, SmallVector<int64_t>(ciphertextSize, 0));
    for (CtSlot ctSlot : fixedPositions) {
      fixedMasks[ctSlot.ct][ctSlot.slot] = 1;
    }

    // skip masking if possible
    SmallVector<std::optional<ValueTy>> fixedCurrent;
    fixedCurrent.reserve(numCiphertexts);
    for (const auto& [ct, fixedMask] : llvm::zip(current, fixedMasks)) {
      auto [allZero, allOne] = allZeroAllOne(fixedMask);
      if (allZero) {
        fixedCurrent.push_back(std::nullopt);
      } else if (allOne) {
        fixedCurrent.push_back(ct);
      } else {
        // FIXME; createMask as constant dag node
        ValueTy mask = createMask(ctType, fixedMask);
        fixedCurrent.push_back(NodeTy::mul(ct, mask));
      }
    }

    SmallVector<std::optional<ValueTy>> rotatedCurrent(numCiphertexts);
    rotatedCurrent.reserve(numCiphertexts);
    if (!rotatePositions.empty()) {
      SmallVector<SmallVector<int64_t>> rotateMasks(
          numCiphertexts, SmallVector<int64_t>(ciphertextSize, 0));
      for (CtSlot ctSlot : rotatePositions) {
        rotateMasks[ctSlot.ct][ctSlot.slot] = 1;
      }
      rotatedCurrent =
          applyVirtualRotation(current, round.rotationAmount, rotateMasks);
    }

    // Combine the rotated and fixed parts to form the new current. However,
    // note that a round may involve a subset of sources which are just
    // fixed (no rotations). In this case, there may be some ciphertexts
    // which have no fixed or rotated entries. We call these "untouched",
    // and we need to ensure that the final summation zeroes them out.
    for (int64_t i = 0; i < numCiphertexts; i++) {
      std::optional<ValueTy> fixed = fixedCurrent[i];
      std::optional<ValueTy> rotated = rotatedCurrent[i];
      if (!fixed.has_value() && !rotated.has_value()) continue;

      touchedCiphertexts.insert(i);
      if (!fixed.has_value()) {
        current[i] = *rotated;
      } else if (!rotated.has_value()) {
        current[i] = *fixed;
      } else {
        current[i] = NodeTy::add(*fixed, *rotated);
      }
    }
  }

  // Add up the results, skipping "untouched" ciphertexts which are copies
  // of the input.
  for (int64_t i = 0; i < numCiphertexts; i++) {
    if (!touchedCiphertexts.contains(i)) {
      // FIXME: create zero tensor constant
      current[i] = NodeTy::constant(0);
    }
  }

  return current;
}

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 SmallVector<std::shared_ptr<ArithmeticDagNode<T>>>>
implementShiftNetwork(SmallVector<T>& ciphertexts, const Mapping& mapping,
                      const ShiftScheme& scheme) {
  using NodeTy = ArithmeticDagNode<T>;
  using ValueTy = std::shared_ptr<NodeTy>;

  auto rotationGroups = scheme.rotationGroups;
  SmallVector<SmallVector<ValueTy>> groupResults;
  [[maybe_unused]] int groupIndex = 0;
  for (const RotationGroup& group : rotationGroups) {
    // Compute the subset of SourceShifts needed for this group
    SmallVector<SourceShift> sourceShifts;
    for (const MappingEntry& entry : mapping) {
      if (group.contains(entry.source)) {
        int64_t shift =
            scheme.strategy.getVirtualShift(entry.source, entry.target);
        sourceShifts.push_back({entry.source, shift});
      }
    }

    SmallVector<ValueTy> perGroupResult = rotateOneGroup(
        ciphertexts, sourceShifts, scheme.strategy.getRounds(), group);
    groupResults.push_back(perGroupResult);
  }

  // Add all the per-group results together
  SmallVector<ValueTy> summedResults = groupResults[0];
  summedResults.resize(ciphertexts.size());
  for (const SmallVector<Value>& groupResult : llvm::drop_begin(groupResults)) {
    for (int i = 0; i < summedResults.size(); i++) {
      summedResults[i] = NodeTy::add(summedResults[i], groupResult[i]);
    }
  }

  return summedResults;
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_ROTATIONGROUPKERNEL_H_
