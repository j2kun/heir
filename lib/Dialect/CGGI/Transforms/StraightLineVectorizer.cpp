#include "include/Dialect/CGGI/Transforms/StraightLineVectorizer.h"

#include "include/Dialect/CGGI/IR/CGGIAttributes.h"
#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DEF_STRAIGHTLINEVECTORIZER
#include "include/Dialect/CGGI/Transforms/Passes.h.inc"

struct StraightLineVectorizer
    : impl::StraightLineVectorizerBase<StraightLineVectorizer> {
  using StraightLineVectorizerBase::StraightLineVectorizerBase;

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext &context = getContext();

    // FIXME: implement
  }
};

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
