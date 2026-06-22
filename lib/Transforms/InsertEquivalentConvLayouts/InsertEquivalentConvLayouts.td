#ifndef LIB_TRANSFORMS_INSERTEQUIVALENTCONVLAYOUTS_INSERTEQUIVALENTCONVLAYOUTS_TD_
#define LIB_TRANSFORMS_INSERTEQUIVALENTCONVLAYOUTS_INSERTEQUIVALENTCONVLAYOUTS_TD_

include "mlir/Pass/PassBase.td"

def InsertEquivalentConvLayouts : Pass<"insert-equivalent-conv-layouts"> {
  let summary = "Materialize several layout assignments for each convolution.";
  let description = [{
    For every multichannel convolution, this pass generates multiple valid
    layout assignments (e.g. the Halevi-Shoup MatvecDiagonal packing with and
    without the pixel-shuffle row-interchange optimization). Each assignment is
    materialized as its own chain of `tensor_ext.convert_layout` ops feeding a
    copy of the convolution and a trailing conversion back to a common layout,
    inserted alongside the original op.

    The results of these trailing conversions are collected as the operands of
    an `equivalence.class` op, which records that they are interchangeable
    implementations of the same value. Users of the original convolution are
    rerouted to read the class result, so downstream e-graph extraction can
    select among the members.

    With `remove-original`, the original convolution is erased. Otherwise it is
    kept and added to the class as another equivalent member.
  }];
  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::equivalence::EquivalenceDialect",
  ];
  let options = [
    Option<"ciphertextSize", "ciphertext-size", "int", /*default=*/"1024",
           "Power of two length of the ciphertexts the data is packed in.">,
    Option<"removeOriginal", "remove-original", "bool", /*default=*/"false",
           "If true, replace the original convolution with the "
           "equivalence.class result and erase it.">,
  ];
}

#endif  // LIB_TRANSFORMS_INSERTEQUIVALENTCONVLAYOUTS_INSERTEQUIVALENTCONVLAYOUTS_TD_
