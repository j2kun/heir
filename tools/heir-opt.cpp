#include "mlir/include/mlir/IR/DialectRegistry.h"          // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "tools/HeirOptRegistration.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::heir::registerHeirOptDialectsAndPasses(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HEIR Pass Driver", registry));
}
