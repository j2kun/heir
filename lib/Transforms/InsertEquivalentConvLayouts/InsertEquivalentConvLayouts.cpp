#include "lib/Transforms/InsertEquivalentConvLayouts/InsertEquivalentConvLayouts.h"

#include <cstdint>

#include "EquivalenceDialect.h"  // from @tamagoyaki
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/Layout/ConvolutionLayoutBuilder.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/SmallPtrSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_INSERTEQUIVALENTCONVLAYOUTS
#include "lib/Transforms/InsertEquivalentConvLayouts/InsertEquivalentConvLayouts.h.inc"

using tensor_ext::LayoutAttr;

namespace {

// Returns the row-major layout attribute for `type`, used as the canonical
// layout of a value that does not yet carry one.
LayoutAttr rowMajorLayout(MLIRContext* ctx, RankedTensorType type,
                          int64_t ciphertextSize) {
  return LayoutAttr::getFromIntegerRelation(
      ctx, getRowMajorLayoutRelation(type, ciphertextSize));
}

}  // namespace

struct InsertEquivalentConvLayouts
    : impl::InsertEquivalentConvLayoutsBase<InsertEquivalentConvLayouts> {
  using InsertEquivalentConvLayoutsBase::InsertEquivalentConvLayoutsBase;

  // Materializes every valid layout for `op` and groups the results in an
  // equivalence.class. ConvOpTy is one of the multichannel linalg conv ops.
  template <typename ConvOpTy>
  void processConv(ConvOpTy op) {
    MLIRContext* ctx = op.getContext();
    auto dataType = cast<RankedTensorType>(op.getInputs().front().getType());
    auto filterType = cast<RankedTensorType>(op.getInputs().back().getType());
    auto outputType = cast<RankedTensorType>(op->getResult(0).getType());

    // A layout-free value is treated as living in the canonical row-major
    // layout; that is the `from` side of each operand conversion and the `to`
    // side the variants are converted back to so they are interchangeable.
    LayoutAttr dataFrom = rowMajorLayout(ctx, dataType, ciphertextSize);
    LayoutAttr filterFrom = rowMajorLayout(ctx, filterType, ciphertextSize);
    LayoutAttr resultTo = rowMajorLayout(ctx, outputType, ciphertextSize);

    // Enumerate the candidate layouts. Only valid ones are kept.
    SmallVector<ConvolutionLayout> configs;
    for (bool interchangeRows : {false, true}) {
      FailureOr<ConvolutionLayout> config = getMatvecDiagonalConvolutionLayout(
          op, ciphertextSize, interchangeRows);
      if (succeeded(config)) configs.push_back(*config);
    }
    if (configs.empty()) return;

    OpBuilder builder(op);
    SmallVector<Value> classInputs;
    for (const ConvolutionLayout& config : configs) {
      FailureOr<ConvolutionLayoutOps> ops = buildConvolutionWithLayout(
          op, dataFrom, filterFrom, resultTo, config);
      if (failed(ops)) continue;
      // Insert the detached chain (defs before uses) right before `op`.
      builder.insert(ops->dataConvert);
      builder.insert(ops->filterConvert);
      builder.insert(ops->conv);
      builder.insert(ops->resultConvert);
      classInputs.push_back(ops->resultConvert.getResult());
    }
    if (classInputs.empty()) return;

    // Unless it is removed, the original convolution is itself an equivalent
    // implementation, so it joins the class as another member.
    Value origResult = op->getResult(0);
    if (!removeOriginal) classInputs.push_back(origResult);

    // The class consumes the original result, so it must come after `op`.
    builder.setInsertionPointAfter(op);
    auto classOp = equivalence::ClassOp::create(
        builder, op.getLoc(), outputType, classInputs, /*leader=*/Value(),
        /*min_cost_index=*/IntegerAttr());

    // In either case, route the original convolution's users through the class
    // result, skipping the class's own use of it when the original is a member.
    if (removeOriginal) {
      origResult.replaceAllUsesWith(classOp.getResult());
      op->erase();
    } else {
      SmallPtrSet<Operation*, 1> except{classOp};
      origResult.replaceAllUsesExcept(classOp.getResult(), except);
    }
  }

  void runOnOperation() override {
    // Collect first so that erasing the originals does not disturb the walk.
    SmallVector<Operation*> convs;
    getOperation()->walk([&](Operation* op) {
      if (isa<linalg::Conv2DNchwFchwOp, linalg::Conv1DNcwFcwOp>(op))
        convs.push_back(op);
    });

    for (Operation* op : convs) {
      if (auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(op))
        processConv(conv);
      else if (auto conv = dyn_cast<linalg::Conv1DNcwFcwOp>(op))
        processConv(conv);
    }
  }
};

}  // namespace heir
}  // namespace mlir
