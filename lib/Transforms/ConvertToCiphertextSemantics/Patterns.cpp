#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
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

Value applyPadding(Value value, LayoutAttr layout, ImplicitLocOpBuilder &b,
                   const std::function<void(Operation *)> &createdOpCallback) {
  LLVM_DEBUG(llvm::dbgs() << "Applying padding...\n");
  // The input has already been broadcast, which implies data was replicated,
  // so we can set the regions that should be padded with the padded value
  // as a series of insert_slice ops
  RankedTensorType broadcastType = cast<RankedTensorType>(value.getType());
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();
  Value current = value;
  int64_t rank = broadcastType.getRank();
  for (int i = 0; i < rank; ++i) {
    auto paddingAmount = alignment.getPadding()[i];
    int64_t paddingOffset = broadcastType.getDimSize(i) - paddingAmount;
    auto padSlice = b.create<arith::ConstantOp>(
        RankedTensorType::get({paddingAmount}, broadcastType.getElementType()),
        alignment.getPaddingValue());
    createdOpCallback(padSlice);

    SmallVector<int64_t> offsets(rank, 0);
    SmallVector<int64_t> sizes(rank, 1);
    SmallVector<int64_t> strides(rank, 1);
    offsets[i] = paddingOffset;
    sizes[i] = paddingAmount;

    auto insertSliceOp = b.create<tensor::InsertSliceOp>(
        padSlice, current, ArrayRef<Value>{}, ArrayRef<Value>{},
        ArrayRef<Value>{}, offsets, sizes, strides);
    createdOpCallback(insertSliceOp);

    current = insertSliceOp.getResult();
  }

  return current;
}

FailureOr<Value> implementAssignLayoutForTensor(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  RankedTensorType dataSemanticType =
      cast<RankedTensorType>(op.getValue().getType());
  RankedTensorType ciphertextSemanticType = cast<RankedTensorType>(
      materializeLayout(dataSemanticType, op.getLayout(), ciphertextSize));
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
    // 1. Broadcast to the out shape
    if (alignment.getIn() != alignment.getOut()) {
      RankedTensorType outType = RankedTensorType::get(
          alignment.getOut(), ciphertextSemanticType.getElementType());
      auto zero =
          builder.create<mlir::arith::ConstantOp>(builder.getZeroAttr(outType));
      auto broadcastOp = builder.create<linalg::BroadcastOp>(
          mostRecentOutput, zero.getResult(), alignment.getInsertedDims());
      mostRecentOutput = broadcastOp.getResults()[0];
      createdOpCallback(broadcastOp);
      createdOpCallback(zero);
    }

    // 2. Apply padding by setting the padded regions to the padded value
    if (alignment.getPadding() && !alignment.getPadding().empty()) {
      mostRecentOutput =
          applyPadding(mostRecentOutput, layout, builder, createdOpCallback);
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

  // 3. Apply the layout
  if (!layout.getMap().isIdentity()) {
    // Materialize encoding via linalg.generic.
    //
    // FIXME: We should have some sort of replication here
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
}

FailureOr<Value> implementAssignLayoutForScalar(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  RankedTensorType ciphertextSemanticType =
      cast<RankedTensorType>(materializeScalarLayout(
          op.getResult().getType(), op.getLayout(), ciphertextSize));
  LLVM_DEBUG(
      llvm::dbgs() << "Converting AssignLayoutOp for scalar to use result type "
                   << ciphertextSemanticType << "\n");

  LayoutAttr layout = op.getLayout();
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();
  Value scalar = op.getValue();

  // Common case: no padding, all replication: the entire encoding can be
  // reduced to a single splat.
  if (alignment.getPadding().empty()) {
    auto splatOp =
        builder.create<tensor::SplatOp>(ciphertextSemanticType, scalar);
    createdOpCallback(splatOp);
    return splatOp.getResult();
  }

  // TODO(#1662): improve scalar layout materialization
  return failure();
}

FailureOr<Value> implementAssignLayout(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  if (isa<RankedTensorType>(op.getResult().getType())) {
    return implementAssignLayoutForTensor(op, ciphertextSize, builder,
                                          createdOpCallback);
  }

  return implementAssignLayoutForScalar(op, ciphertextSize, builder,
                                        createdOpCallback);
};

}  // namespace heir
}  // namespace mlir
