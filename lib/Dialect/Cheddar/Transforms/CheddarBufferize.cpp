#include "lib/Dialect/Cheddar/Transforms/CheddarBufferize.h"

#include <memory>

#include "llvm/include/llvm/ADT/SCCIterator.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

namespace {

// CSE can share an empty tensor between independent insertions. Give each
// insertion its own undefined destination before empty-tensor elimination
// threads that destination back to its producer. Otherwise One-Shot must copy
// the produced payloads to preserve the separate tensor results.
struct SeparateEmptyDestinations
    : PassWrapper<SeparateEmptyDestinations, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SeparateEmptyDestinations)

  StringRef getArgument() const final {
    return "cheddar-separate-empty-destinations";
  }

  void runOnOperation() override {
    getOperation().walk([](tensor::InsertSliceOp insertion) {
      auto empty = insertion.getDest().getDefiningOp<tensor::EmptyOp>();
      if (!empty || empty->hasOneUse()) return;
      OpBuilder builder(insertion);
      auto separate = cast<tensor::EmptyOp>(builder.clone(*empty));
      insertion.getDestMutable().assign(separate.getResult());
    });
  }
};

// Convert callees before their callers, so allocations introduced at call sites
// are visible to upstream's return-allocation hoisting. Converting the whole
// module at once would first replace every return with a copy, then introduce
// the call-site allocations too late to hoist them into the caller's
// out-params.
struct PromoteBufferResults
    : PassWrapper<PromoteBufferResults, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteBufferResults)

  StringRef getArgument() const final {
    return "cheddar-promote-buffer-results";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const CallGraph callGraph(module);
    SmallVector<func::FuncOp> functions;
    // Collect the callee-first order before rewriting calls changes the graph.
    for (auto it = llvm::scc_begin(&callGraph); !it.isAtEnd(); ++it) {
      for (CallGraphNode* node : *it) {
        if (node->isExternal()) continue;
        auto function =
            dyn_cast<func::FuncOp>(node->getCallableRegion()->getParentOp());
        if (function && function->getParentOp() == module)
          functions.push_back(function);
      }
    }

    bufferization::BufferResultsToOutParamsOpts options;
    options.hoistStaticAllocs = true;
    options.addResultAttribute = true;
    options.modifyPublicFunctions = true;
    for (func::FuncOp function : functions) {
      options.filterFn = [function](func::FuncOp* candidate) {
        return *candidate == function;
      };
      if (failed(bufferization::promoteBufferResultsToOutParams(module,
                                                                options))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

void buildCheddarBufferizationPipeline(OpPassManager& pm) {
  pm.addPass(std::make_unique<SeparateEmptyDestinations>());
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  bufferization::OneShotBufferizePassOptions oneShot;
  oneShot.bufferizeFunctionBoundaries = true;
  oneShot.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShot));
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  bufferization::DropEquivalentBufferResultsPassOptions dropEquivalent;
  dropEquivalent.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createDropEquivalentBufferResultsPass(dropEquivalent));
  pm.addPass(std::make_unique<PromoteBufferResults>());
  pm.addPass(createCanonicalizerPass());
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
