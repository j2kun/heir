#ifndef LIB_TRANSFORMS_POPULATEPRIMITIVEROOTS_POPULATEPRIMITIVEROOTS_TD_
#define LIB_TRANSFORMS_POPULATEPRIMITIVEROOTS_POPULATEPRIMITIVEROOTS_TD_

include "mlir/Pass/PassBase.td"

def PopulatePrimitiveRoots : Pass<"populate-primitive-roots"> {
  let summary = "Populates primitive roots on NTT ops requiring a primitive root";
  let description = [{
  }];
  let dependentDialects = [
    "::mlir::heir::PolynomialDialect",
  ];
}

#endif  // LIB_TRANSFORMS_POPULATEPRIMITIVEROOTS_POPULATEPRIMITIVEROOTS_TD_
