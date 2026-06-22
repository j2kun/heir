#include "lib/Utils/Layout/ConvolutionLayoutBuilder.h"

#include <cstdint>

#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv2DNchwFchwOp op, int64_t ciphertextSize, bool interchangeRows) {
  MLIRContext* ctx = op.getContext();
  auto dataType = cast<RankedTensorType>(op.getInputs().front().getType());
  auto filterType = cast<RankedTensorType>(op.getInputs().back().getType());
  auto outputType = cast<RankedTensorType>(op->getResult(0).getType());

  SmallVector<int64_t> strides(op.getStrides().getValues<int64_t>().begin(),
                               op.getStrides().getValues<int64_t>().end());

  IntegerRelation dataRelation =
      getRowMajorLayoutRelation(dataType, ciphertextSize);
  FailureOr<IntegerRelation> filterRelation =
      get2dConvChwFchwFilterDiagonalizedRelation(filterType, dataType, strides,
                                                 /*padding=*/0, ciphertextSize,
                                                 interchangeRows);
  if (failed(filterRelation)) return failure();
  IntegerRelation resultRelation = get2dConvResultRelation(
      outputType, strides, /*padding=*/0, ciphertextSize, interchangeRows);

  ConvolutionLayout layout;
  layout.dataLayout = LayoutAttr::getFromIntegerRelation(ctx, dataRelation);
  layout.filterLayout =
      LayoutAttr::getFromIntegerRelation(ctx, *filterRelation);
  layout.resultLayout = LayoutAttr::getFromIntegerRelation(ctx, resultRelation);
  layout.kernel = secret::KernelAttr::get(ctx, KernelName::MatvecDiagonal,
                                          /*force=*/false);
  // Matches the schedule the layout-propagation pass uses for the diagonalized
  // filter conversion.
  layout.filterDomainSchedule = {0, 1};
  return layout;
}

FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv1DNcwFcwOp op, int64_t ciphertextSize, bool interchangeRows) {
  MLIRContext* ctx = op.getContext();
  auto dataType = cast<RankedTensorType>(op.getInputs().front().getType());
  auto filterType = cast<RankedTensorType>(op.getInputs().back().getType());
  auto outputType = cast<RankedTensorType>(op->getResult(0).getType());

  int64_t stride = op.getStrides().getValues<int64_t>().begin()[0];

  IntegerRelation dataRelation =
      getRowMajorLayoutRelation(dataType, ciphertextSize);
  FailureOr<IntegerRelation> filterRelation =
      get1dConvCwFcwFilterDiagonalizedRelation(filterType, dataType, stride,
                                               /*padding=*/0, ciphertextSize,
                                               interchangeRows);
  if (failed(filterRelation)) return failure();
  IntegerRelation resultRelation = get1dConvResultRelation(
      outputType, stride, /*padding=*/0, ciphertextSize, interchangeRows);

  ConvolutionLayout layout;
  layout.dataLayout = LayoutAttr::getFromIntegerRelation(ctx, dataRelation);
  layout.filterLayout =
      LayoutAttr::getFromIntegerRelation(ctx, *filterRelation);
  layout.resultLayout = LayoutAttr::getFromIntegerRelation(ctx, resultRelation);
  layout.kernel = secret::KernelAttr::get(ctx, KernelName::MatvecDiagonal,
                                          /*force=*/false);
  layout.filterDomainSchedule = {0, 1};
  return layout;
}

// Shared implementation for the single-channel conv ops, which differ only in
// their op type: data row-major, filter diagonalized (no row-interchange
// variant), result row-major, MatvecDiagonal kernel.
template <typename ConvOpTy>
static FailureOr<ConvolutionLayout> getSingleChannelLayout(
    ConvOpTy op, int64_t ciphertextSize) {
  MLIRContext* ctx = op.getContext();
  auto dataType = cast<RankedTensorType>(op.getInputs().front().getType());
  auto filterType = cast<RankedTensorType>(op.getInputs().back().getType());
  auto outputType = cast<RankedTensorType>(op->getResult(0).getType());

  FailureOr<IntegerRelation> filterRelation = getConvFilterDiagonalizedRelation(
      filterType, dataType, /*padding=*/0, ciphertextSize);
  if (failed(filterRelation)) return failure();

  ConvolutionLayout layout;
  layout.dataLayout = LayoutAttr::getFromIntegerRelation(
      ctx, getRowMajorLayoutRelation(dataType, ciphertextSize));
  layout.filterLayout =
      LayoutAttr::getFromIntegerRelation(ctx, *filterRelation);
  layout.resultLayout = LayoutAttr::getFromIntegerRelation(
      ctx, getRowMajorLayoutRelation(outputType, ciphertextSize));
  layout.kernel = secret::KernelAttr::get(ctx, KernelName::MatvecDiagonal,
                                          /*force=*/false);
  return layout;
}

FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv1DOp op, int64_t ciphertextSize) {
  return getSingleChannelLayout(op, ciphertextSize);
}

FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv2DOp op, int64_t ciphertextSize) {
  return getSingleChannelLayout(op, ciphertextSize);
}

namespace {

// Creates a detached convert_layout op (built via OperationState so it is never
// inserted into a block) that re-packs `value` from `fromLayout` to `toLayout`,
// stamping the target layout as a `tensor_ext.layout` attribute.
ConvertLayoutOp createDetachedConvertLayout(OpBuilder& builder, Location loc,
                                            Value value, LayoutAttr fromLayout,
                                            LayoutAttr toLayout,
                                            ArrayRef<int64_t> domainSchedule) {
  OperationState state(loc, ConvertLayoutOp::getOperationName());
  ConvertLayoutOp::build(builder, state, value, fromLayout, toLayout,
                         builder.getDenseI64ArrayAttr(domainSchedule));
  auto convertOp = cast<ConvertLayoutOp>(Operation::create(state));
  convertOp->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName, toLayout);
  return convertOp;
}

}  // namespace

FailureOr<ConvolutionLayoutOps> buildConvolutionWithLayout(
    Operation* convOp, LayoutAttr dataFromLayout, LayoutAttr filterFromLayout,
    LayoutAttr resultToLayout, const ConvolutionLayout& layout) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(convOp);
  if (!dpsOp || dpsOp.getNumDpsInputs() != 2 || convOp->getNumResults() != 1) {
    return failure();
  }

  OpOperand* dataOperand = dpsOp.getDpsInputOperand(0);
  OpOperand* filterOperand = dpsOp.getDpsInputOperand(1);
  unsigned dataOperandIdx = dataOperand->getOperandNumber();
  unsigned filterOperandIdx = filterOperand->getOperandNumber();
  Value data = dataOperand->get();
  Value filter = filterOperand->get();
  Location loc = convOp->getLoc();

  OpBuilder builder(convOp->getContext());

  // 1 & 2. Re-pack the operands into the layouts the kernel expects.
  ConvertLayoutOp dataConvert =
      createDetachedConvertLayout(builder, loc, data, dataFromLayout,
                                  layout.dataLayout, layout.dataDomainSchedule);
  ConvertLayoutOp filterConvert = createDetachedConvertLayout(
      builder, loc, filter, filterFromLayout, layout.filterLayout,
      layout.filterDomainSchedule);

  // 3. Clone the conv (a detached copy preserving op type, strides, dilations,
  // ...), point it at the re-packed operands, and annotate it with the result
  // layout and the selected kernel. The original `convOp` is untouched.
  Operation* newConv = convOp->clone();
  newConv->setOperand(dataOperandIdx, dataConvert.getResult());
  newConv->setOperand(filterOperandIdx, filterConvert.getResult());
  newConv->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
                   layout.resultLayout);
  newConv->setAttr(secret::SecretDialect::kKernelAttrName, layout.kernel);

  // 4. Convert the result back to the layout downstream consumers expect.
  ConvertLayoutOp resultConvert = createDetachedConvertLayout(
      builder, loc, newConv->getResult(0), layout.resultLayout, resultToLayout,
      /*domainSchedule=*/{});

  return ConvolutionLayoutOps{dataConvert, filterConvert, newConv,
                              resultConvert};
}

}  // namespace heir
}  // namespace mlir
