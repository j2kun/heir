#include "lib/Analysis/ScaleAnalysis/ScaleModel.h"

#include <cassert>

#include "lib/Analysis/ScaleAnalysis/ScaleState.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/Overloaded.h"
#include "llvm/include/llvm/Support/Debug.h"     // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

#define DEBUG_TYPE "scale-model"

namespace mlir {
namespace heir {

using scale::Free;
using scale::ScaleState;

ScaleState BGVScaleModel::evalMulScale(const RLWELocalParam& param,
                                       ScaleState lhs, ScaleState rhs) const {
  const auto& bgvParam = static_cast<const bgv::LocalParam&>(param);
  const auto* schemeParam = bgvParam.getSchemeParam();
  int64_t t = schemeParam->getPlaintextModulus();
  LDBG() << "BGVScaleModel::evalMulScale with t = " << t << ", rhs=" << rhs
         << ", lhs=" << lhs;
  return std::visit(
      Overloaded{
          [](Free, Free) -> ScaleState { return ScaleState(Free{}); },
          [t](Free, int rhs) -> ScaleState {
            return ScaleState(rhs * rhs % t);
          },
          [t](int lhs, Free) -> ScaleState {
            return ScaleState(lhs * lhs % t);
          },
          [t](int lhs, int rhs) -> ScaleState {
            return ScaleState(lhs * rhs % t);
          },
      },
      lhs.getScale(), rhs.getScale());
}

ScaleState BGVScaleModel::evalMulScaleBackward(const RLWELocalParam& param,
                                               ScaleState result,
                                               ScaleState lhs) const {
  const auto& bgvParam = static_cast<const bgv::LocalParam&>(param);
  const auto* schemeParam = bgvParam.getSchemeParam();
  int64_t t = schemeParam->getPlaintextModulus();
  return std::visit(
      Overloaded{
          [](Free, Free) -> ScaleState { return ScaleState(Free{}); },
          // invalid?
          [&](Free, int lhs) -> ScaleState { return ScaleState(Free{}); },
          [&](int result, Free) -> ScaleState { return ScaleState(result); },
          [&](int result, int lhs) -> ScaleState {
            auto lhsInv = multiplicativeInverse(APInt(64, lhs), APInt(64, t));
            return ScaleState(result * lhsInv.getSExtValue() % t);
          },
      },
      result.getScale(), lhs.getScale());
}

ScaleState BGVScaleModel::evalModReduceScale(const RLWELocalParam& inputParam,
                                             ScaleState scale) const {
  const auto& bgvParam = static_cast<const bgv::LocalParam&>(inputParam);
  const auto* schemeParam = bgvParam.getSchemeParam();
  int64_t t = schemeParam->getPlaintextModulus();
  std::vector<int64_t> qi = schemeParam->getQi();
  int64_t level = bgvParam.getCurrentLevel();
  APInt qInvT = multiplicativeInverse(APInt(64, qi[level] % t), APInt(64, t));
  return std::visit(Overloaded{
                        [](Free) -> ScaleState { return ScaleState(Free{}); },
                        [&](int val) -> ScaleState {
                          return ScaleState(val * qInvT.getSExtValue() % t);
                        },
                    },
                    scale.getScale());
}

ScaleState BGVScaleModel::evalModReduceScaleBackward(
    const RLWELocalParam& inputParam, ScaleState resultScale) const {
  const auto& bgvParam = static_cast<const bgv::LocalParam&>(inputParam);
  const auto* schemeParam = bgvParam.getSchemeParam();
  auto t = schemeParam->getPlaintextModulus();
  auto qi = schemeParam->getQi();
  auto level = bgvParam.getCurrentLevel();
  return std::visit(Overloaded{
                        [](Free) -> ScaleState { return ScaleState(Free{}); },
                        [&](int val) -> ScaleState {
                          return ScaleState(val * (qi[level] % t) % t);
                        },
                    },
                    resultScale.getScale());
}

ScaleState CKKSScaleModel::evalMulScale(const RLWELocalParam& param,
                                        ScaleState lhs, ScaleState rhs) const {
  // TODO(#1640): support high-precision scale management
  return std::visit(
      Overloaded{
          [](Free, Free) -> ScaleState { return ScaleState(Free{}); },
          [](Free, int rhs) -> ScaleState { return ScaleState(rhs + rhs); },
          [](int lhs, Free) -> ScaleState { return ScaleState(lhs + lhs); },
          [](int lhs, int rhs) -> ScaleState { return ScaleState(lhs + rhs); },
      },
      lhs.getScale(), rhs.getScale());
}

ScaleState CKKSScaleModel::evalMulScaleBackward(const RLWELocalParam& param,
                                                ScaleState result,
                                                ScaleState lhs) const {
  // TODO(#1640): support high-precision scale management
  return std::visit(
      Overloaded{
          [](Free, Free) -> ScaleState { return ScaleState(Free{}); },
          // Free result but non-free operand: invalid?
          [](Free, int lhs) -> ScaleState { return ScaleState(Free{}); },
          [](int result, Free) -> ScaleState { return ScaleState(result); },
          [](int result, int lhs) -> ScaleState {
            return ScaleState(result - lhs);
          },
      },
      result.getScale(), lhs.getScale());
}

ScaleState CKKSScaleModel::evalModReduceScale(const RLWELocalParam& inputParam,
                                              ScaleState scale) const {
  const auto& ckksParam = static_cast<const ckks::LocalParam&>(inputParam);
  const auto* schemeParam = ckksParam.getSchemeParam();
  // TODO(#1640): rescale using logqi instead of logDefaultScale
  // auto logqi = schemeParam->getLogqi();
  // auto level = inputParam.getCurrentLevel();
  auto logDefaultScale = schemeParam->getLogDefaultScale();
  return std::visit(Overloaded{
                        [](Free) -> ScaleState { return ScaleState(Free{}); },
                        [&](int val) -> ScaleState {
                          return ScaleState(val - logDefaultScale);
                        },
                    },
                    scale.getScale());
}

ScaleState CKKSScaleModel::evalModReduceScaleBackward(
    const RLWELocalParam& inputParam, ScaleState resultScale) const {
  const auto& ckksParam = static_cast<const ckks::LocalParam&>(inputParam);
  const auto* schemeParam = ckksParam.getSchemeParam();
  // TODO(#1640): rescale using logqi instead of logDefaultScale
  // auto logqi = schemeParam->getLogqi();
  // auto level = inputParam.getCurrentLevel();
  auto logDefaultScale = schemeParam->getLogDefaultScale();
  return std::visit(Overloaded{
                        [](Free) -> ScaleState { return ScaleState(Free{}); },
                        [&](int val) -> ScaleState {
                          return ScaleState(val + logDefaultScale);
                        },
                    },
                    resultScale.getScale());
}

}  // namespace heir
}  // namespace mlir
