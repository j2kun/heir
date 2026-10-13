#ifndef TOOLS_HEIROPTREGISTRATION_H_
#define TOOLS_HEIROPTREGISTRATION_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Registers all dialects, interfaces, passes, and pipelines exposed by
// heir-opt.
void registerHeirOptDialectsAndPasses(DialectRegistry& registry);

}  // namespace heir
}  // namespace mlir

#endif  // TOOLS_HEIROPTREGISTRATION_H_
