#ifndef LIB_TRANSFORMS_POPULATEPRIMITIVEROOTS_POPULATEPRIMITIVEROOTS_TD_
#define LIB_TRANSFORMS_POPULATEPRIMITIVEROOTS_POPULATEPRIMITIVEROOTS_TD_

include "mlir/Pass/PassBase.td"

def PopulatePrimitiveRoots : Pass<"populate-primitive-roots"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = [

  ];
}

#endif  // LIB_TRANSFORMS_POPULATEPRIMITIVEROOTS_POPULATEPRIMITIVEROOTS_TD_
