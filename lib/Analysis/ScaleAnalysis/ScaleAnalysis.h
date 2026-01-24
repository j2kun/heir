#ifndef LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
#define LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_

#include <cassert>
#include <cstdint>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/ScaleAnalysis/ScaleModel.h"
#include "lib/Analysis/ScaleAnalysis/ScaleState.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Parameters/RLWEParams.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

using scale::ScaleState;

class ScaleLattice : public dataflow::Lattice<ScaleState> {
 public:
  using Lattice::Lattice;
};

using OperandScales = ArrayRef<const ScaleLattice*>;
using ResultScales = ArrayRef<const ScaleLattice*>;
using MutableOperandScales = ArrayRef<ScaleLattice*>;
using MutableResultScales = ArrayRef<ScaleLattice*>;

/// Forward Analyse the scale of each secret Value
///
/// This forward analysis roots from user input as `inputScale`,
/// and after each HE operation, the scale will be updated.
/// For ct-pt or cross-level operation, we will assume the scale of the
/// undetermined hand side to be the same as the determined one.
/// This forms the level-specific scaling factor constraint.
/// See also the "Ciphertext management" section in the document.
///
/// The analysis will stop propagation for AdjustScaleOp, as the scale
/// of it should be determined together by the forward pass (from input
/// to its operand) and the backward pass (from a determined ciphertext to
/// its result).
///
/// This analysis is expected to determine (almost) all the scales of
/// the secret Value, or ciphertext in the program.
/// The level of plaintext Value, or the opaque result of AdjustLevelOp
/// should be determined by the Backward Analysis below.
class ScaleAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>,
      public SecretnessAnalysisDependent<ScaleAnalysis> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysis>;

  ScaleAnalysis(DataFlowSolver& solver, const ScaleModel& model,
                const RLWESchemeParam& schemeParam, int64_t inputScale)
      : dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>(solver),
        model(model),
        schemeParam(schemeParam),
        inputScale(inputScale) {}

  void setToEntryState(ScaleLattice* lattice) override {
    if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
      propagateIfChanged(lattice, lattice->join(ScaleState(inputScale)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(ScaleState()));
  }

  SmallVector<ScaleState> getOperandScales(Operation* op) {
    SmallVector<ScaleState> scales;
    for (Value operand : op->getOperands()) {
      auto operandState = getLatticeElement(operand)->getValue();
      scales.push_back(operandState);
    }
    return scales;
  }

  RLWELocalParam getLocalParam(Value value) {
    auto level = getLevelFromMgmtAttr(value).getInt();
    auto dimension = getDimensionFromMgmtAttr(value);
    return RLWELocalParam(&schemeParam, level, dimension);
  };

  // Transfer functions with non-default forward propagation rules
  ScaleState transferForward(mgmt::ModReduceOp op, OperandScales operands);
  ScaleState transferForward(mgmt::BootstrapOp op, OperandScales operands);
  ScaleState transferForward(mgmt::AdjustScaleOp op, OperandScales operands);
  ScaleState transferForward(mgmt::InitOp op, OperandScales operands);
  ScaleState transferForward(arith::MulIOp op, OperandScales operands);
  ScaleState transferForward(arith::MulFOp op, OperandScales operands);

  ScaleState deriveResultScale(Operation* op, OperandScales operands);

  LogicalResult visitOperation(Operation* op, OperandScales operands,
                               MutableResultScales results) override;

  void visitExternalCall(CallOpInterface call, OperandScales argumentLattices,
                         MutableResultScales resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }

 private:
  const ScaleModel& model;
  const RLWESchemeParam& schemeParam;
  int64_t inputScale;
};

/// Backward Analyse the scale of plaintext Value / opaque result of
/// AdjustLevelOp
///
/// This analysis should be run after the (forward) ScaleAnalysis
/// where the scale of (almost) all the secret Value is determined.
///
/// A special example is ct2 = mul(ct0, rs(adjust_scale(ct1))), where the scale
/// of ct0, ct1, ct2 is determined by the forward pass, rs is rescaling. Then
/// the scale of adjust_scale(ct1) should be determined by the backward pass
/// via backpropagation from ct2 to rs then to adjust_scale.
class ScaleAnalysisBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<ScaleLattice>,
      public SecretnessAnalysisDependent<ScaleAnalysisBackward> {
 public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysisBackward>;

  ScaleAnalysisBackward(DataFlowSolver& solver,
                        SymbolTableCollection& symbolTable,
                        const ScaleModel& model,
                        const RLWESchemeParam& schemeParam)
      : dataflow::SparseBackwardDataFlowAnalysis<ScaleLattice>(solver,
                                                               symbolTable),
        model(model),
        schemeParam(schemeParam) {}

  void setToExitState(ScaleLattice* lattice) override {
    propagateIfChanged(lattice, lattice->join(ScaleState()));
  }

  RLWELocalParam getLocalParam(Value value) {
    auto level = getLevelFromMgmtAttr(value).getInt();
    auto dimension = getDimensionFromMgmtAttr(value);
    return RLWELocalParam(&schemeParam, level, dimension);
  };

  void getOperandsWithoutScale(
      Operation* op, SmallVectorImpl<OpOperand*>& operandsWithoutScale) {
    for (OpOperand& operand : op->getOpOperands()) {
      ScaleLattice* lattice = getLatticeElement(operand.get());
      ScaleState scale = lattice->getValue();
      if (!scale.isInt()) {
        operandsWithoutScale.push_back(&operand);
      }
    }
  }

  void getOperandsWithScale(Operation* op,
                            SmallVectorImpl<OpOperand*>& operandsWithoutScale) {
    for (OpOperand& operand : op->getOpOperands()) {
      ScaleLattice* lattice = getLatticeElement(operand.get());
      ScaleState scale = lattice->getValue();
      if (scale.isInt()) {
        operandsWithoutScale.push_back(&operand);
      }
    }
  }

  LogicalResult visitOperation(Operation* op, MutableOperandScales operands,
                               ResultScales results) override;

  // dummy impl
  void visitBranchOperand(OpOperand& operand) override {}
  void visitCallOperand(OpOperand& operand) override {}
  void visitNonControlFlowArguments(
      RegionSuccessor& successor, ArrayRef<BlockArgument> arguments) override {}

 private:
  const ScaleModel& model;
  const RLWESchemeParam& schemeParam;
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

int64_t getScale(Value value, DataFlowSolver* solver);

constexpr StringRef kArgScaleAttrName = "mgmt.scale";

void annotateScale(Operation* top, DataFlowSolver* solver);

int64_t getScaleFromMgmtAttr(Value value);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
