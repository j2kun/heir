#include "lib/Transforms/LayoutOptimization/InterfaceImpl.h"

#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/Kernel.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "llvm/include/llvm/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

using tensor_ext::ConvertLayoutOp;
using tensor_ext::NewLayoutAttr;
static auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;

Hoister createTrivialHoister(Operation* op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    Attribute outputLayout = convertLayoutOp.getToLayout();
    result.convertLayoutOp = convertLayoutOp;
    result.newInputLayouts =
        SmallVector<Attribute>(op->getNumOperands(), outputLayout);
    result.newKernel = KernelName::Trivial;
    result.newOutputLayout = outputLayout;
    return result;
  };
}

Hoister createPrecomposingMatvecHoister(linalg::MatvecOp op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    auto fromLayout = dyn_cast<NewLayoutAttr>(convertLayoutOp.getFromLayout());
    auto toLayout = dyn_cast<NewLayoutAttr>(convertLayoutOp.getToLayout());

    if (!fromLayout || !toLayout) return failure();

    // Operand order for Matvec op is:
    //
    // 0: matrix
    // 1: input vector
    // 2: output vector
    FailureOr<Attribute> oldMatrixLayoutRes =
        findAttributeAssociatedWith(op->getOperand(0), kLayoutAttrName);
    assert(succeeded(oldMatrixLayoutRes) && "failed to find matrix layout!");
    NewLayoutAttr oldMatrixLayout =
        dyn_cast<NewLayoutAttr>(oldMatrixLayoutRes.value());
    if (!oldMatrixLayout) return failure();

    result.convertLayoutOp = convertLayoutOp;
    // All the matvec kernels we have today should maintain the layout of the
    // vector before and after the op.
    result.newOutputLayout = toLayout;

    // The kernel is unchanged, so copy the existing kernel attr
    result.newKernel = op->getAttrOfType<secret::KernelAttr>(
                             secret::SecretDialect::kKernelAttrName)
                           .getName();

    presburger::IntegerRelation newMatrixLayoutRelation =
        hoistConversionThroughMatvec(oldMatrixLayout.getRelation(),
                                     fromLayout.getRelation(),
                                     toLayout.getRelation());
    // FIXME: construct new layout attr from relation
    Attribute newMatrixLayout = Attribute();
    result.newInputLayouts =
        SmallVector<Attribute>{newMatrixLayout, toLayout, toLayout};
    return result;
  };
}

}  // namespace heir
}  // namespace mlir
