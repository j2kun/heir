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

// A fully-specified, internally consistent packing choice for a single
// convolution op. The four fields must agree with each other: `kernel` is the
// kernel that will implement the conv, and `dataLayout` / `filterLayout` /
// `resultLayout` are the slot layouts that kernel expects its data operand,
// filter operand, and result to be in, respectively.
//
// Build instances with the factory functions below, which guarantee a valid
// combination, or assemble one by hand for a configuration not yet covered by a
// factory. The mechanical builder `convertConvolutionToLayout` does not check
// validity -- it simply materializes whatever layouts this struct names.
struct ConvolutionLayout {
  tensor_ext::LayoutAttr dataLayout;
  tensor_ext::LayoutAttr filterLayout;
  tensor_ext::LayoutAttr resultLayout;
  secret::KernelAttr kernel;

  // Optional domain-schedule hints forwarded to the operand convert_layout ops.
  // These do not change semantics; they only steer ISL toward an efficient loop
  // nest when the conversion is later lowered. Leave empty if unsure.
  SmallVector<int64_t> dataDomainSchedule;
  SmallVector<int64_t> filterDomainSchedule;
};

// Factory: the Halevi-Shoup "MatvecDiagonal" packing for a 2-D multichannel
// convolution (data laid out row-major, filter expanded to a diagonalized
// Toeplitz matrix, result in the matching row-major-with-gap layout).
//
// `interchangeRows` toggles the pixel-shuffle / depth-to-space row-interchange
// optimization on the filter and result relations. Both values produce a valid
// configuration for the same MatvecDiagonal kernel; they differ only in the
// resulting rotation cost.
//
// Returns failure if the diagonalized filter relation cannot be constructed
// (e.g. the data does not fit the ciphertext size).
FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv2DNchwFchwOp op, int64_t ciphertextSize, bool interchangeRows);

// Factory: the MatvecDiagonal packing for a 1-D multichannel convolution. See
// the 2-D overload for the meaning of `interchangeRows`.
FailureOr<ConvolutionLayout> getMatvecDiagonalConvolutionLayout(
    linalg::Conv1DNcwFcwOp op, int64_t ciphertextSize, bool interchangeRows);

// The freshly-created ops that together re-express a convolution in a chosen
// layout. They are produced detached (not inserted into any block) and wired to
// each other: `conv` consumes the results of `dataConvert` / `filterConvert`,
// and `resultConvert` consumes `conv`'s result.
//
// `resultConvert.getResult()` is the value intended to stand in for the
// original convolution's result.
struct ConvolutionLayoutOps {
  // data --(dataFromLayout -> layout.dataLayout)--> data_l
  tensor_ext::ConvertLayoutOp dataConvert;
  // filter --(filterFromLayout -> layout.filterLayout)--> filter_l
  tensor_ext::ConvertLayoutOp filterConvert;
  // conv(data_l, filter_l) carrying layout.resultLayout + layout.kernel.
  Operation* conv;
  // conv_result --(layout.resultLayout -> resultToLayout)--> new_result
  tensor_ext::ConvertLayoutOp resultConvert;
};

// Builds, but does NOT insert, the ops that re-express a layout-free
// convolution
//
//     result = conv(data, filter)
//
// in the layout named by `layout`:
//
//     data_l   = tensor_ext.convert_layout data   {from = dataFromLayout,
//                                                   to   = layout.dataLayout}
//     filter_l = tensor_ext.convert_layout filter {from = filterFromLayout,
//                                                   to   = layout.filterLayout}
//     result_l = conv(data_l, filter_l)
//                  {tensor_ext.layout = layout.resultLayout,
//                   secret.kernel     = layout.kernel}
//     new      = tensor_ext.convert_layout result_l {from =
//     layout.resultLayout,
//                                                     to   = resultToLayout}
//
// `convOp` is only read -- it is neither modified, replaced, nor erased, and no
// uses are rewired, so the existing IR is left exactly as it was. The returned
// ops are detached; it is the caller's responsibility to insert them (in the
// order dataConvert, filterConvert, conv, resultConvert so that defs precede
// uses) and to replace uses of the original result with
// `result.resultConvert.getResult()`. The caller also owns the returned ops
// until they are inserted (or destroyed).
//
// `convOp` must be a destination-style op with exactly two inputs (data,
// filter) and a single result -- i.e. any of the linalg conv ops. The new conv
// is a clone of `convOp`, so its op type and attributes (strides, dilations,
// ...) are preserved.
//
// `dataFromLayout` / `filterFromLayout` are the operands' current layouts (the
// `from` side of the operand conversions), and `resultToLayout` is the layout
// downstream consumers expect (the `to` side of the trailing conversion).
// These are explicit so a pass can supply them from whatever layout state it
// tracks rather than this helper having to rediscover them.
//
// Returns failure if `convOp` does not match the expected shape.
FailureOr<ConvolutionLayoutOps> buildConvolutionWithLayout(
    Operation* convOp, tensor_ext::LayoutAttr dataFromLayout,
    tensor_ext::LayoutAttr filterFromLayout,
    tensor_ext::LayoutAttr resultToLayout, const ConvolutionLayout& layout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTIONLAYOUTBUILDER_H_
