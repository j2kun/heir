#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"

#include <cassert>
#include <cstdint>

#include "lib/Analysis/ScaleAnalysis/ScaleModel.h"
#include "lib/Analysis/ScaleAnalysis/ScaleState.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"    // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"     // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

#define DEBUG_TYPE "scale-analysis"

namespace mlir {
namespace heir {

static void logForwardTransfer(StringRef opName, OperandScales operands,
                               const ScaleState& result) {
  LLVM_DEBUG({
    llvm::dbgs() << "transferForward: " << opName << "(";
    for (auto* operand : operands) {
      operand->getValue().print(llvm::dbgs());
      llvm::dbgs() << ", ";
    }
    llvm::dbgs() << ") = ";
    result.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
};

ScaleState ScaleAnalysis::transferForward(mgmt::ModReduceOp op,
                                          OperandScales operands) {
  SmallVector<ScaleState> scales = getOperandScales(op);
  // It is critical that we getLocalParam from the input, not the result,
  // because the result will have level 1 lower than the input, which impacts
  // the BGV scale when there is a level-specific scaling factor.
  auto result =
      model.evalModReduceScale(getLocalParam(op.getInput()), scales[0]);
  return ScaleState(result);
}

ScaleState ScaleAnalysis::transferForward(mgmt::BootstrapOp op,
                                          OperandScales operands) {
  // inputScale is either Delta or Delta^2 depending on the analysis
  // initialization.
  return ScaleState(inputScale);
}

ScaleState ScaleAnalysis::transferForward(mgmt::AdjustScaleOp op,
                                          OperandScales operands) {
  return ScaleState(scale::Free{});
}

ScaleState ScaleAnalysis::transferForward(mgmt::InitOp op,
                                          OperandScales operands) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(op.getResult());
  if (mgmtAttr && mgmtAttr.getScale() != 0) {
    return ScaleState(mgmtAttr.getScale());
  }
  return ScaleState(scale::Free{});
}

ScaleState ScaleAnalysis::transferForward(arith::MulIOp op,
                                          OperandScales operands) {
  SmallVector<ScaleState> scales = getOperandScales(op);
  auto result =
      model.evalMulScale(getLocalParam(op.getResult()), scales[0], scales[1]);
  return ScaleState(result);
}

ScaleState ScaleAnalysis::transferForward(arith::MulFOp op,
                                          OperandScales operands) {
  SmallVector<ScaleState> scales = getOperandScales(op);
  auto result =
      model.evalMulScale(getLocalParam(op.getResult()), scales[0], scales[1]);
  return ScaleState(result);
}

ScaleState ScaleAnalysis::deriveResultScale(Operation* op,
                                            OperandScales operands) {
  return llvm::TypeSwitch<Operation*, ScaleState>(op)
      .Case<mgmt::ModReduceOp, mgmt::AdjustScaleOp, mgmt::BootstrapOp,
            mgmt::InitOp, arith::MulIOp, arith::MulFOp>(
          [&](auto op) -> ScaleState {
            auto result = transferForward(op, operands);
            LLVM_DEBUG(logForwardTransfer(op->getName().getStringRef(),
                                          operands, result));
            return result;
          })
      .Default([&](auto* op) -> ScaleState {
        ScaleState result;
        for (auto* operandState : operands) {
          result = ScaleState::join(result, operandState->getValue());
        }
        LLVM_DEBUG(
            logForwardTransfer(op->getName().getStringRef(), operands, result));
        return result;
      });
}

LogicalResult ScaleAnalysis::visitOperation(Operation* op,
                                            OperandScales operands,
                                            MutableResultScales results) {
  auto propagate = [&](Value value, const ScaleState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  ScaleState resultScale = deriveResultScale(op, operands);
  SmallVector<OpResult> secretResults;
  this->getSecretResults(op, secretResults);
  for (auto result : secretResults) {
    propagate(result, resultScale);
  }

  return success();
}

void ScaleAnalysis::visitExternalCall(CallOpInterface call,
                                      OperandScales argumentLattices,
                                      MutableResultScales resultLattices) {
  auto callback = std::bind(&ScaleAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<scale::ScaleState, ScaleLattice>(
      call, argumentLattices, resultLattices, callback);
}

//===----------------------------------------------------------------------===//
// ScaleAnalysis (Backward)
//===----------------------------------------------------------------------===//

static void logBackwardTransfer(StringRef opName, ResultScales results,
                                const ScaleState& operand,
                                unsigned operandNum) {
  LLVM_DEBUG({
    llvm::dbgs() << "transferBackward: " << opName << " results(";
    for (auto* result : results) {
      result->getValue().print(llvm::dbgs());
      llvm::dbgs() << ", ";
    }
    llvm::dbgs() << ") -> operand " << operandNum << " = ";
    operand.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
};

LogicalResult ScaleAnalysisBackward::visitOperation(
    Operation* op, MutableOperandScales operands, ResultScales results) {
  auto propagate = [&](OpOperand& operand, const ScaleState& state) {
    auto* lattice = getLatticeElement(operand.get());
    ChangeResult changed = lattice->join(state);
    logBackwardTransfer(op->getName().getStringRef(), results, state,
                        operand.getOperandNumber());
    propagateIfChanged(lattice, changed);
  };

  SmallVector<unsigned> secretResultIndices;
  getSecretResultIndices(op, secretResultIndices);
  if (secretResultIndices.empty()) {
    LDBG() << "Not back propagating for " << op->getName()
           << " because no results are secret";
    return success();
  }

  SmallVector<OpOperand*> operandsWithoutScale;
  getOperandsWithoutScale(op, operandsWithoutScale);
  if (operandsWithoutScale.empty()) return success();

  LDBG() << "Back propagating scale for " << op->getName();
  llvm::TypeSwitch<Operation&, void>(*op)
      .Case<arith::MulIOp, arith::MulFOp>([&](auto mulOp) {
        SmallVector<OpOperand*> operandsWithScale;
        getOperandsWithScale(op, operandsWithScale);
        if (operandsWithScale.empty()) return;

        ScaleState resultScale = results[0]->getValue();
        ScaleLattice* presentScale =
            operands[operandsWithScale.front()->getOperandNumber()];
        auto scaleOther =
            model.evalMulScaleBackward(getLocalParam(mulOp.getResult()),
                                       resultScale, presentScale->getValue());
        propagate(*operandsWithoutScale[0], ScaleState(scaleOther));
      })
      .Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        ScaleState resultScale = results[0]->getValue();
        auto operandScale = model.evalModReduceScaleBackward(
            getLocalParam(modReduceOp.getResult()), resultScale);
        propagate(modReduceOp->getOpOperand(0), ScaleState(operandScale));
      })
      // All other ops propagate result scale to scale-less operands without
      // changing the scale value. This handles situations where, to back
      // propagate scale to an op like an adjust_scale (whose output scale is
      // free), you may have to propagate backward through ops that do not
      // impact scale.
      .Default([&](auto& op) {
        // Joining all results into one is not necessary... there should always
        // be a single op result, but let's do it anyway to try to be
        // future-proof.
        ScaleState resultScale;
        for (auto* resultLattice : results) {
          resultScale =
              ScaleState::join(resultScale, resultLattice->getValue());
        }
        for (auto* operand : operandsWithoutScale) {
          propagate(*operand, resultScale);
        }
      });
  return success();
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

int64_t getScale(Value value, DataFlowSolver* solver) {
  auto* lattice = solver->lookupState<ScaleLattice>(value);
  if (!lattice) {
    assert(false && "ScaleLattice not found");
    return 0;
  }
  if (!lattice->getValue().getInt()) {
    assert(false && "ScaleLattice not initialized");
    return 0;
  }
  return lattice->getValue().getInt();
}

int64_t getScaleFromMgmtAttr(Value value) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value);
  if (!mgmtAttr) {
    assert(false && "MgmtAttr not found");
    return 0;
  }
  return mgmtAttr.getScale();
}

void annotateScale(Operation* top, DataFlowSolver* solver) {
  auto getIntegerAttr = [&](int scale) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), scale);
  };

  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      setAttributeAssociatedWith(value, kArgScaleAttrName,
                                 getIntegerAttr(getScale(value, solver)));
    }
  });
}

}  // namespace heir
}  // namespace mlir
