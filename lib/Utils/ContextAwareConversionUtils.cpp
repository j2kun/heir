#include "lib/Utils/ContextAwareConversionUtils.h"

#include <memory>

#include "mlir/include/mlir/IR/IRMapping.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::func::ReturnOp;

FailureOr<Operation *> convertAnyOperand(const TypeConverter *typeConverter,
                                         Operation *op,
                                         ArrayRef<Value> operands,
                                         ConversionPatternRewriter &rewriter) {
  const auto *contextAwareTypeConverter =
      dynamic_cast<const ContextAwareTypeConverter *>(typeConverter);

  if (contextAwareTypeConverter) {
    if (contextAwareTypeConverter->isLegal(op)) {
      return failure();
    }
  } else {
    if (typeConverter->isLegal(op)) {
      return failure();
    }
  }

  SmallVector<Type> newOperandTypes;
  SmallVector<Type> newResultTypes;
  if (contextAwareTypeConverter) {
    if (failed(contextAwareTypeConverter->convertTypes(
            op->getResultTypes(), op->getResults(), newResultTypes)))
      return failure();

    if (failed(contextAwareTypeConverter->convertTypes(
            op->getOperandTypes(), op->getOperands(), newOperandTypes)))
      return failure();

    if (newOperandTypes == op->getOperandTypes() &&
        newResultTypes == op->getResultTypes()) {
      return failure();
    }
  } else {
    auto result =
        typeConverter->convertTypes(op->getResultTypes(), newResultTypes);
    if (failed(result)) return failure();
    if (failed(typeConverter->convertTypes(op->getOperandTypes(),
                                           newOperandTypes)))
      return failure();
  }

  SmallVector<std::unique_ptr<Region>, 1> regions;
  IRMapping mapping;
  for (auto &r : op->getRegions()) {
    Region *newRegion = new Region(op);
    rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
    if (failed(rewriter.convertRegionTypes(newRegion, *typeConverter)))
      return failure();
    regions.emplace_back(newRegion);
  }

  Operation *newOp = rewriter.create(OperationState(
      op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
      op->getAttrs(), op->getSuccessors(), regions));

  rewriter.replaceOp(op, newOp);
  return newOp;
}

}  // namespace heir
}  // namespace mlir
