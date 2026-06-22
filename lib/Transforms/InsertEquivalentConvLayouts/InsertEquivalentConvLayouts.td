#ifndef LIB_TRANSFORMS_INSERTEQUIVALENTCONVLAYOUTS_INSERTEQUIVALENTCONVLAYOUTS_TD_
#define LIB_TRANSFORMS_INSERTEQUIVALENTCONVLAYOUTS_INSERTEQUIVALENTCONVLAYOUTS_TD_

include "mlir/Pass/PassBase.td"

def InsertEquivalentConvLayouts : Pass<"insert-equivalent-conv-layouts"> {
  let summary = "Materialize several layout assignments for each convolution.";
  let description = [{
    For every convolution (the multichannel `linalg.conv_1d_ncw_fcw` and
    `linalg.conv_2d_nchw_fchw`, as well as the single-channel `linalg.conv_1d`
    and `linalg.conv_2d`), this pass generates several valid layout assignments
    and records that they are interchangeable.

    The candidates are:

      * the Halevi-Shoup `MatvecDiagonal` packing (for multichannel convs, both
        with and without the pixel-shuffle row-interchange optimization; the two
        coincide for stride 1 and are then deduplicated), and
      * the naive `MatvecNaive` packing, in which the expanded Toeplitz filter
        matrix is laid out row-major instead of diagonalized.

    Each candidate is materialized as its own chain of `tensor_ext.convert_layout`
    ops feeding a copy of the convolution (carrying its `tensor_ext.layout` and
    `secret.kernel` attributes) and a trailing conversion back to a common
    layout, inserted alongside the original op. The results of these trailing
    conversions become the operands of an `equivalence.class` op, and users of
    the original convolution are rerouted to read the class result, so
    downstream e-graph extraction can select among the members.

    With `remove-original`, the original convolution is erased. Otherwise it is
    kept and added to the class as another equivalent member.

    Note: the `MatvecNaive` kernel does not yet have a lowering (TODO(#1589)).
    Its candidate is a sound layout assignment for exploration, but a program
    that selects it cannot currently be lowered end-to-end.
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
