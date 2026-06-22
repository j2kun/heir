#ifndef LIB_UTILS_LAYOUT_CONVOLUTIONLAYOUTBUILDER_H_
#define LIB_UTILS_LAYOUT_CONVOLUTIONLAYOUTBUILDER_H_

#include <cstdint>

#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

// A self-consistent packing choice for a convolution: the layouts its data
// operand, filter operand, and result must be in for the chosen `kernel`.
struct ConvolutionLayout {
  tensor_ext::LayoutAttr dataLayout;
  tensor_ext::LayoutAttr filterLayout;
  tensor_ext::LayoutAttr resultLayout;
  secret::KernelAttr kernel;

  // Optional domain-schedule hints forwarded to the operand convert_layout ops
  // to help ISL emit an efficient loop nest when lowering.
  SmallVector<int64_t> dataDomainSchedule;
  SmallVector<int64_t> filterDomainSchedule;
};

// Returns the Halevi-Shoup MatvecDiagonal packing for a multichannel
// convolution. `interchangeRows` toggles the pixel-shuffle row-interchange
// optimization. Fails if the diagonalized filter relation cannot be built.
FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv2DNchwFchwOp op, int64_t ciphertextSize, bool interchangeRows);
FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv1DNcwFcwOp op, int64_t ciphertextSize, bool interchangeRows);

// The detached ops that re-express a convolution in a chosen layout, wired to
// each other but not inserted into any block. `resultConvert.getResult()` is
// the value meant to replace the original convolution result.
struct ConvolutionLayoutOps {
  tensor_ext::ConvertLayoutOp dataConvert;
  tensor_ext::ConvertLayoutOp filterConvert;
  Operation* conv;
  tensor_ext::ConvertLayoutOp resultConvert;
};

// Builds, without inserting, the ops re-expressing `conv(data, filter)` in
// `layout`: a convert_layout for each operand (from its current layout to the
// kernel's), a clone of the conv carrying the result layout and kernel, and a
// convert_layout of the result back to `resultToLayout`. `convOp` is only read,
// so the existing IR is left untouched; the caller inserts the returned ops
// (data, filter, conv, result) and rewires uses. Fails if `convOp` is not a
// destination-style op with two inputs and one result.
FailureOr<ConvolutionLayoutOps> buildConvolutionWithLayout(
    Operation* convOp, tensor_ext::LayoutAttr dataFromLayout,
    tensor_ext::LayoutAttr filterFromLayout,
    tensor_ext::LayoutAttr resultToLayout, const ConvolutionLayout& layout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTIONLAYOUTBUILDER_H_
