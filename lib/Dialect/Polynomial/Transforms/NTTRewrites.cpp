#include "lib/Dialect/Polynomial/Transforms/NTTRewrites.h"

#include <utility>

#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DEF_POLYMULTONTT
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "lib/Dialect/Polynomial/Transforms/NTTRewrites.cpp.inc"
}  // namespace rewrites

struct PolyMulToNTT : impl::PolyMulToNTTBase<PolyMulToNTT> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // TODO(#1095): migrate to mod arith type
    // patterns.add<rewrites::NTTRewritePolyMul>(patterns.getContext());
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
