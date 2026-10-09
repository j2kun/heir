#include "EmatchDialect.h"
#include "EquivalenceDialect.h"
#include "TamagoyakiTiming.h"
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

  int result = mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tamagoyaki Demo Driver", registry));

  tamagoyaki::printTimingReport();
  return result;
}
