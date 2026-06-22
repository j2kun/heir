#include "EmatchDialect.h"
#include "EquivalenceDialect.h"
#include "TamagoyakiTiming.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Transforms/InsertEquivalentConvLayouts/InsertEquivalentConvLayouts.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"          // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"             // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project

int main(int argc, char **argv) {
  tamagoyaki::registerTimingCLOptions();

  mlir::equivalence::registerEquivalencePasses();
  mlir::ematch::registerEmatchPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::equivalence::EquivalenceDialect>();
  registry.insert<mlir::ematch::EmatchDialect>();
  registry.insert<mlir::heir::secret::SecretDialect>();
  registry.insert<mlir::heir::tensor_ext::TensorExtDialect>();

  mlir::heir::registerInsertEquivalentConvLayouts();

  int result = mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tamagoyaki Demo Driver", registry));

  tamagoyaki::printTimingReport();
  return result;
}
