#include "include/Dialect/Secret/Transforms/DistributeGeneric.h"

#include <algorithm>
#include <utility>

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret.distribute-generic"

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETDISTRIBUTEGENERIC
#include "include/Dialect/Secret/Transforms/Passes.h.inc"

std::optional<Value> ofrToValue(std::optional<OpFoldResult> ofr) {
  if (ofr.has_value()) {
    if (auto value = llvm::dyn_cast_if_present<Value>(*ofr)) {
      return value;
    }
  }
  return std::nullopt;
}

// Inline the inner block of a secret.generic that has no secret operands.
//
// E.g.,
//
//    %res = secret.generic ins(%value : i32) {
//     ^bb0(%clear_value: i32):
//       %c7 = arith.constant 7 : i32
//       %0 = arith.muli %clear_value, %c7 : i32
//       secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
// is transformed to
//
//    %0 = arith.constant 0 : i32
//    %res = arith.muli %value, %0 : i32
//
struct CollapseSecretlessGeneric : public OpRewritePattern<GenericOp> {
  CollapseSecretlessGeneric(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    for (Type ty : op.getOperandTypes()) {
      if (dyn_cast<SecretType>(ty)) {
        return failure();
      }
    }

    YieldOp yieldOp = dyn_cast<YieldOp>(op.getBody()->getOperations().back());
    rewriter.inlineBlockBefore(op.getBody(), op.getOperation(), op.getInputs());
    rewriter.replaceOp(op, yieldOp.getValues());
    rewriter.eraseOp(yieldOp);
    return success();
  }
};

// Remove unused args of a secret.generic op
//
// E.g.,
//
//    %res = secret.generic
//       ins(%value_sec, %unused_sec : !secret.secret<i32>, !secret.secret<i32>)
//       {
//     ^bb0(%used: i32, %unused: i32):
//       %0 = arith.muli %used, %used : i32
//       secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
// is transformed to
//
//    %res = secret.generic
//       ins(%value_sec : !secret.secret<i32>) {
//     ^bb0(%used: i32):
//       %0 = arith.muli %used, %used : i32
//       secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
struct RemoveUnusedGenericArgs : public OpRewritePattern<GenericOp> {
  RemoveUnusedGenericArgs(mlir::MLIRContext *context)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    bool hasUnusedOps = false;
    Block *body = op.getBody();
    for (int i = 0; i < body->getArguments().size(); ++i) {
      BlockArgument arg = body->getArguments()[i];
      if (arg.use_empty()) {
        hasUnusedOps = true;
        rewriter.updateRootInPlace(op, [&]() {
          body->eraseArgument(i);
          op.getOperation()->eraseOperand(i);
        });
        // Ensure the next iteration uses the right arg number
        --i;
      }
    }

    return hasUnusedOps ? success() : failure();
  }
};

// Split a secret.generic containing multiple ops into multiple secret.generics.
//
// E.g.,
//
//    %res = secret.generic ins(%value : !secret.secret<i32>) {
//    ^bb0(%clear_value: i32):
//      %c7 = arith.constant 7 : i32
//      %0 = arith.muli %clear_value, %c7 : i32
//      secret.yield %0 : i32
//    } -> (!secret.secret<i32>)
//
// is transformed to
//
//    %secret_7 = secret.generic {
//      %c7 = arith.constant 7 : i32
//      secret.yield %c7 : i32
//    } -> !secret.secret<i32>
//    %1 = secret.generic ins(
//       %arg0, %secret_7 : !secret.secret<i32>, !secret.secret<i32>) {
//    ^bb0(%clear_arg0: i32, %clear_7: i32):
//      %7 = arith.muli %clear_arg0, %clear_7 : i32
//      secret.yield %7 : i32
//    } -> !secret.secret<i32>
//
// When options are provided specifying which ops to distribute, the pattern
// will split at the first detected specified op, possibly creating three new
// secret.generics, and otherwise will split it at the first op from the entry
// block, and will always create two secret.generics.
struct SplitGeneric : public OpRewritePattern<GenericOp> {
  SplitGeneric(mlir::MLIRContext *context,
               llvm::ArrayRef<std::string> opsToDistribute)
      : OpRewritePattern<GenericOp>(context, /*benefit=*/1),
        opsToDistribute(opsToDistribute) {}

  void distributeThroughRegionHoldingOp(GenericOp genericOp,
                                        Operation &opToDistribute,
                                        PatternRewriter &rewriter) const {
    assert(opToDistribute.getNumRegions() > 0 &&
           "opToDistribute must have at least one region");
    assert(genericOp.getBody()->getOperations().size() == 2 &&
           "opToDistribute must have one non-yield op");

    // Supports ops with countable loop iterations, like affine.for and scf.for,
    // but not scf.while which has multiple associated regions.
    if (auto loop = dyn_cast<LoopLikeOpInterface>(opToDistribute)) {
      // FIXME: delete, not needed because these come from block arguments of
      // the generic which are never secret.
      //
      // auto iv = loop.getSingleInductionVar();
      // auto lowerBound = ofrToValue(loop.getSingleLowerBound());
      // auto upperBound = ofrToValue(loop.getSingleUpperBound());
      // auto step = ofrToValue(loop.getSingleStep());

      // if ((iv && isa<SecretType>(iv->getType())) ||
      //     (lowerBound && isa<SecretType>(lowerBound->getType())) ||
      //     (upperBound && isa<SecretType>(upperBound->getType())) ||
      //     (step && isa<SecretType>(step->getType()))) {
      //   opToDistribute.emitOpError()
      //       << "Found loop-like op with induction variable, step, or bounds
      //       of "
      //          "type secret.sercret.\n";
      // }

      // Example:
      //
      //   secret.generic ins(%value : !secret.secret<...>) {
      //   ^bb0(%clear_value: ...):
      //     %1 = scf.for ... iter_args(%iter_arg = %clear_value) -> ... {
      //       scf.yield ...
      //     }
      //     secret.yield %1 : ...
      //   } -> (!secret.secret<...>)
      //
      // This needs to be converted to:
      //
      //   %1 = scf.for ... iter_args(%iter_arg = %value) -> ... {
      //     %2 = secret.generic ins(%iter_arg : !secret.secret<...>) {
      //     ^bb0(%clear_iter_arg: ...):
      //       ...
      //       secret.yield %1 : ...
      //     }
      //     scf.yield %2 : ...
      //   }
      //
      // Terminators of the region are not part of the secret, since they just
      // handle control flow.

      // Before moving the loop out of the generic, connect the loop's operands
      // to the corresponding secret operands (via the block argument number).
      genericOp->getParentOp()->dump();
      rewriter.startRootUpdate(genericOp);

      // Set the op's operands
      MutableArrayRef<OpOperand> operands = opToDistribute.getOpOperands();
      for (OpOperand &operand : operands) {
        assert(
            isa<BlockArgument>(operand.get()) &&
            "loop init values must be block arguments of the secret.generic");
        BlockArgument initArg = cast<BlockArgument>(operand.get());
        operand.set(genericOp.getOperand(initArg.getArgNumber()));
      }

      // Set the op's region iter arg types, which need to match the possibly
      // new type of the operands modified above
      for (auto [arg, operand] :
           llvm::zip(loop.getRegionIterArgs(), loop.getInits())) {
        arg.setType(operand.getType());
      }

      opToDistribute.moveBefore(genericOp);

      // Now the loop is before the secret generic, but the generic still
      // yields the loop's result (the loop should yield the generic's result)
      // and the generic still needs to be moved inside the loop.

      genericOp->moveBefore(&loop.getLoopRegions()
                                 .front()
                                 ->getBlocks()
                                 .front()
                                 .getOperations()
                                 .front());

      auto yieldedValues = loop.getYieldedValues();
      auto *terminator = opToDistribute.getRegion(0).front().getTerminator();

      rewriter.finalizeRootUpdate(genericOp);
      genericOp->getParentOp()->dump();

      assert(false && "wat");

      return;
    }

    // Cast to loop-like interface to assert the induction variables and bounds
    // are not secret, but iter-args may be.

    // TODO: handle RegionBranchOpInterface (scf.while, scf.if) and potentially
    // BranchOpInterface (cf.br)
  }

  void splitGenericAroundOp(GenericOp op, Operation &opToDistribute,
                            PatternRewriter &rewriter) const {
    assert(false && "not implemented");
  }

  void splitGenericAfterOp(GenericOp op, Operation &opToDistribute,
                           PatternRewriter &rewriter) const {
    Block *body = op.getBody();
    // The inputs to the op are generic op's block arguments (cleartext
    // values), and they need to change to be the corresponding generic op's
    // normal operands (maybe secret values).
    SmallVector<Value> newInputs;
    // The indices of the new inputs in the original block argument list
    SmallVector<unsigned> newInputIndices;
    for (Value val : opToDistribute.getOperands()) {
      int index = std::find(body->getArguments().begin(),
                            body->getArguments().end(), val) -
                  body->getArguments().begin();
      newInputIndices.push_back(index);
      newInputs.push_back(op.getOperand(index));
    }

    // Result types are secret versions of the results of the block's only op
    SmallVector<Type> newResultTypes;
    for (Type ty : opToDistribute.getResultTypes()) {
      newResultTypes.push_back(SecretType::get(ty));
    }

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), newInputs, newResultTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
          auto *newOp = b.clone(opToDistribute);
          newOp->setOperands(blockArguments);
          b.create<YieldOp>(loc, newOp->getResults());
        });

    // Once the op is split off into a new generic op, we need to add new
    // operands to the old generic op, add new corresponding block arguments,
    // and replace all uses of the opToDistribute's results with the created
    // block arguments.
    SmallVector<Value> oldGenericNewBlockArgs;
    rewriter.updateRootInPlace(op, [&]() {
      op.getInputsMutable().append(newGeneric.getResults());
      for (auto ty : opToDistribute.getResultTypes()) {
        BlockArgument arg =
            op.getBody()->addArgument(ty, opToDistribute.getLoc());
        oldGenericNewBlockArgs.push_back(arg);
      }
    });
    rewriter.replaceOp(&opToDistribute, oldGenericNewBlockArgs);
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    Block *body = op.getBody();
    unsigned numOps = body->getOperations().size();
    assert(numOps > 0 &&
           "secret.generic must have nonempty body (the verifier should "
           "enforce this)");

    // Recursive base case: stop if there's only one op left, and it has no
    // regions, noting that we check for 2 ops because the last op is enforced
    // to be a yield op by the verifier.
    if (numOps == 2 && body->front().getRegions().empty()) {
      return failure();
    }

    Operation *opToDistribute;
    bool first = true;
    if (opsToDistribute.empty()) {
      opToDistribute = &body->front();
    } else {
      for (Operation &op : body->getOperations()) {
        // op.getName().getStringRef() is the qualified op name (e.g.,
        // affine.for)
        if (std::find(opsToDistribute.begin(), opsToDistribute.end(),
                      op.getName().getStringRef()) != opsToDistribute.end()) {
          opToDistribute = &op;
          break;
        }
        first = false;
      }
    }

    // Base case: if none of a generic op's member ops are in the list of ops
    // to process, stop.
    if (!opToDistribute) return failure();

    if (numOps == 2 && !opToDistribute->getRegions().empty()) {
      distributeThroughRegionHoldingOp(op, *opToDistribute, rewriter);
      return success();
    }

    if (first) {
      splitGenericAfterOp(op, *opToDistribute, rewriter);
    } else {
      splitGenericAroundOp(op, *opToDistribute, rewriter);
    }
    return success();
  }

 private:
  llvm::ArrayRef<std::string> opsToDistribute;
};

struct DistributeGeneric
    : impl::SecretDistributeGenericBase<DistributeGeneric> {
  using SecretDistributeGenericBase::SecretDistributeGenericBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    LLVM_DEBUG({
      llvm::dbgs() << "Running secret-distribute-generic ";
      if (opsToDistribute.empty()) {
        llvm::dbgs() << "on all ops\n";
      } else {
        llvm::dbgs() << "on ops: \n";
        for (const auto &op : opsToDistribute) {
          llvm::dbgs() << " - " << op << "\n";
        }
      }
    });

    patterns.add<SplitGeneric>(context, opsToDistribute);
    patterns.add<CollapseSecretlessGeneric, RemoveUnusedGenericArgs>(context);
    // TODO(https://github.com/google/heir/issues/170): add a pattern that
    // distributes generic through a single op containing one or more regions.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
