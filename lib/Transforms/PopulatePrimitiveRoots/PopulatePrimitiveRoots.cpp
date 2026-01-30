#include "lib/Transforms/PopulatePrimitiveRoots/PopulatePrimitiveRoots.h"

#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

using polynomial::NTTOp;

class PopulatePrimitiveRootNTT final : public OpRewritePattern<NTTOp> {
 public:
  using OpRewritePattern<NTTOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NTTOp op, PatternRewriter& rewriter) const {
    return success();
  }
};

#define GEN_PASS_DEF_POPULATEPRIMITIVEROOTS
#include "lib/Transforms/PopulatePrimitiveRoots/PopulatePrimitiveRoots.h.inc"

struct PopulatePrimitiveRoots
    : impl::PopulatePrimitiveRootsBase<PopulatePrimitiveRoots> {
  using PopulatePrimitiveRootsBase::PopulatePrimitiveRootsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<PopulatePrimitiveRootNTT>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
