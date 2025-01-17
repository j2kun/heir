#include "lib/Transforms/DropUnitDims/DropUnitDims.h"

#include "mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_DROPUNITDIMS
#include "lib/Transforms/DropUnitDims/DropUnitDims.h.inc"

struct DropUnitDims : impl::DropUnitDimsBase<DropUnitDims> {
  using DropUnitDimsBase::DropUnitDimsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateContractionOpRankReducingPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
