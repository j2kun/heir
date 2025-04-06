#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using tensor_ext::LayoutAttr;

bool containsDim(ArrayRef<int64_t> dims, int64_t dim) {
  return llvm::any_of(dims, [dim](int64_t d) { return d == dim; });
}

Value expandDims(Value value, LayoutAttr layout, ImplicitLocOpBuilder &b,
                 const std::function<void(Operation *)> &createdOpCallback) {
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();

  // Scalars get a special case: just splat the value as a single slot
  // in each inserted dimension.
  if (!isa<RankedTensorType>(value.getType())) {
    // TODO(#1662): improve scalar layout materialization
    // This is broken, at least in that the resulting generic doesn't lower
    // to the right loop because the iteration domain is based on the size of
    // this tensor (<1xi16>) vs the output tensor. Maybe special case the
    // caller of this helper, so this helper only does tensors?
    SmallVector<int64_t> shape(1, alignment.getInsertedDims().size());
    auto splatOp = b.create<tensor::SplatOp>(
        RankedTensorType::get(shape, value.getType()), value);
    createdOpCallback(splatOp);
    return splatOp.getResult();
  }

  // Tensors are handled via tensor.expand_shape
  RankedTensorType dataSemanticType = cast<RankedTensorType>(value.getType());
  // It's a bit weird, but to make an expand shape op we have to group the
  // output indices in dataSemanticType.getRank() many groups where the 1's
  // are all grouped with axes from the dataSemanticType. But the 1's can
  // show up before or after the data semantic tensor's dims, so we
  // eagerly consume unit dims before and after each data semantic dim.
  SmallVector<int64_t> newSizes;
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices nextGroup;
  int64_t ciphertextIndex = 0, groupIndex = 0;
  while (groupIndex < dataSemanticType.getRank()) {
    // Process all the unit dims.
    while (containsDim(alignment.getInsertedDims(), ciphertextIndex)) {
      newSizes.push_back(1);
      nextGroup.push_back(ciphertextIndex);
      ++ciphertextIndex;
    }

    // Now process exactly one data dim.
    newSizes.push_back(dataSemanticType.getDimSize(groupIndex));
    nextGroup.push_back(ciphertextIndex);
    ++ciphertextIndex;

    // Now process any more unit dims.
    while (containsDim(alignment.getInsertedDims(), ciphertextIndex)) {
      newSizes.push_back(1);
      nextGroup.push_back(ciphertextIndex);
      ++ciphertextIndex;
    }

    reassociation.push_back(nextGroup);
    nextGroup.clear();
    ++groupIndex;
  }
  RankedTensorType expandedType =
      RankedTensorType::get(newSizes, dataSemanticType.getElementType());
  auto expandOp =
      b.create<tensor::ExpandShapeOp>(expandedType, value, reassociation);
  createdOpCallback(expandOp);
  return expandOp.getResult();
}

Value applyPadding(Value value, LayoutAttr layout, ImplicitLocOpBuilder &b,
                   const std::function<void(Operation *)> &createdOpCallback) {
  RankedTensorType dataSemanticType = cast<RankedTensorType>(value.getType());
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();
  // Note padding is asserted to be present, and paddingValue is enforced
  // to be present whenever padding is present due to attribute verifier.
  auto padValueOp = b.create<arith::ConstantOp>(alignment.getPaddingValue());

  SmallVector<int64_t> newSizes;
  SmallVector<OpFoldResult> lows;
  SmallVector<OpFoldResult> highs;
  for (int i = 0; i < dataSemanticType.getRank(); ++i) {
    newSizes.push_back(dataSemanticType.getDimSize(i) +
                       alignment.getPadding()[i]);
    lows.push_back(b.getIndexAttr(0));
    highs.push_back(b.getIndexAttr(alignment.getPadding()[i]));
  }
  RankedTensorType expandedType =
      RankedTensorType::get(newSizes, dataSemanticType.getElementType());
  auto padOp = b.create<tensor::PadOp>(expandedType, value, lows, highs,
                                       padValueOp, /*nofold=*/false);

  createdOpCallback(padOp);
  b.setInsertionPointAfter(padOp);
  return padOp.getResult();
}

FailureOr<Value> maybeReplicateAlongAxis(
    tensor_ext::AssignLayoutOp op, Value value, int axis,
    int64_t outputAxisSize, ImplicitLocOpBuilder &b,
    const std::function<void(Operation *)> &createdOpCallback) {
  RankedTensorType mostRecentType = cast<RankedTensorType>(value.getType());
  int64_t dataDimSize = mostRecentType.getDimSize(axis);

  if (outputAxisSize % dataDimSize != 0 && dataDimSize % outputAxisSize != 0) {
    auto diag = op.emitError()
                << "Before replication, tensor size must divide or be a "
                   "multiple of data "
                   "size, or else repetition will not make sense!";
    diag.attachNote()
        << "For dim " << axis << ", target dim size was " << outputAxisSize
        << ", but input size (after optional dim insertion and padding) was "
        << dataDimSize;
    return diag;
  }

  if (dataDimSize < outputAxisSize) {
    // Concatenate appropriately
    SmallVector<int64_t> newSizes =
        SmallVector<int64_t>(mostRecentType.getShape());
    newSizes[axis] = outputAxisSize;
    RankedTensorType expandedShape =
        RankedTensorType::get(newSizes, mostRecentType.getElementType());

    int64_t numIters = outputAxisSize / dataDimSize;
    SmallVector<Value> repeatedInputs(numIters, value);
    auto concatOp = b.create<tensor::ConcatOp>(op.getLoc(), expandedShape,
                                               /*axis=*/axis, repeatedInputs);
    createdOpCallback(concatOp);
    return concatOp.getResult();
  }
  return value;
}

FailureOr<Value> implementAssignLayout(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  Type dataSemanticType = op.getResult().getType();
  RankedTensorType ciphertextSemanticType = cast<RankedTensorType>(
      isa<RankedTensorType>(dataSemanticType)
          ? materializeLayout(cast<RankedTensorType>(dataSemanticType),
                              op.getLayout(), ciphertextSize)
          : materializeScalarLayout(dataSemanticType, op.getLayout(),
                                    ciphertextSize));
  LLVM_DEBUG(llvm::dbgs() << "Converting AssignLayoutOp to use result type "
                          << ciphertextSemanticType << "\n");
  Value input = op.getValue();
  LayoutAttr layout = op.getLayout();

  // Not all aspects of a replication attribute may be applied. In some rare
  // cases, the input type may already be materialized and no work is
  // required. So this tracks the value that is the result of the most
  // recently applied operation in the process, and the final output value to
  // replace this op with.
  Value mostRecentOutput = input;

  // Apply the semantics of the replication attribute in order before
  // applying the layout.
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();

  if (alignment) {
    // 1. Insert unit dimensions via tensor.expand_shape
    if (alignment.getInsertedDims() && !alignment.getInsertedDims().empty()) {
      mostRecentOutput =
          expandDims(mostRecentOutput, layout, builder, createdOpCallback);
    }

    // 2. Add padding to the end of each axis via tensor.pad
    if (alignment.getPadding() && !alignment.getPadding().empty()) {
      mostRecentOutput =
          applyPadding(mostRecentOutput, layout, builder, createdOpCallback);
    }

    // 3. Replicate the input tensor along each axis via tensor.concat
    for (int i = 0; i < alignment.getOut().size(); ++i) {
      FailureOr<Value> res = maybeReplicateAlongAxis(
          op, mostRecentOutput, i, alignment.getOut()[i], builder,
          createdOpCallback);
      if (failed(res)) return res;
      mostRecentOutput = res.value();
    }
  }

  // At this point, we could try to guarantee that the replicated data tensor
  // has the same number of elements as the ciphertext tensor, but in general
  // this is not required. You could just waste slots, though there is a
  // concern that some kernels that rely on replication may not work as
  // expected. So in this case we emit a warning.
  LLVM_DEBUG({
    RankedTensorType mostRecentType =
        cast<RankedTensorType>(mostRecentOutput.getType());
    if (mostRecentType.getNumElements() !=
        ciphertextSemanticType.getNumElements()) {
      op.emitWarning()
          << "Data type (after replication and padding) " << mostRecentType
          << " has fewer entries than ciphertext type "
          << ciphertextSemanticType
          << ". This may indicate unused slots, or may lead to unexpected "
             "behavior for some kernels that require data replication to "
             "operate properly.";
    }
  });

  // 4. Apply the layout
  if (!layout.getMap().isIdentity()) {
    // Materialize encoding via linalg.generic.
    //
    // Nb., rather than use tensor.empty(), start with constant zeros which
    // plays better with secret.generic lowerings. This implies that any
    // unused values in the layout will default to zero, which seems both
    // like a safe default and the kind of thing that a user could
    // unexpectedly become dependent on.
    auto emptyOp = builder.create<mlir::arith::ConstantOp>(
        builder.getZeroAttr(ciphertextSemanticType));
    createdOpCallback(emptyOp);

    SmallVector<utils::IteratorType> iteratorTypes(
        op.getLayout().getMap().getNumDims(), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps = {
        // The first map corresponds to how the iteration indices map to the
        // input tensor indices. This is the identity because the loop is
        // mapping the input values to ciphertext slots.
        AffineMap::getMultiDimIdentityMap(layout.getMap().getNumDims(),
                                          op.getContext()),
        // The first map is the actual layout, mapping input tensor indices
        // to ciphertext slots.
        layout.getMap()};
    auto materializeLayoutOp = builder.create<linalg::GenericOp>(
        /*resultTypes=*/emptyOp.getResult().getType(),
        /*inputs=*/mostRecentOutput,
        /*outputs=*/emptyOp.getResult(), indexingMaps, iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // Do nothing, which just assigns the input to the output slot.
          auto yieldOp =
              nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
          createdOpCallback(yieldOp);
        });

    createdOpCallback(materializeLayoutOp);
    mostRecentOutput = materializeLayoutOp.getResult(0);
  }

  return mostRecentOutput;
};

}  // namespace heir
}  // namespace mlir
