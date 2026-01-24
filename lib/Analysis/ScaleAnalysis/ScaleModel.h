#ifndef LIB_ANALYSIS_SCALEANALYSIS_SCALEMODEL_H_
#define LIB_ANALYSIS_SCALEANALYSIS_SCALEMODEL_H_

#include <cassert>

#include "lib/Analysis/ScaleAnalysis/ScaleState.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"

namespace mlir {
namespace heir {

using scale::ScaleState;

class ScaleModel {
 public:
  virtual ~ScaleModel() = default;

  virtual ScaleState evalMulScale(const RLWELocalParam& param, ScaleState lhs,
                                  ScaleState rhs) const = 0;
  virtual ScaleState evalMulScaleBackward(const RLWELocalParam& param,
                                          ScaleState result,
                                          ScaleState lhs) const = 0;
  virtual ScaleState evalModReduceScale(const RLWELocalParam& inputParam,
                                        ScaleState scale) const = 0;
  virtual ScaleState evalModReduceScaleBackward(
      const RLWELocalParam& inputParam, ScaleState resultScale) const = 0;
};

struct BGVScaleModel : public ScaleModel {
  using SchemeParam = bgv::SchemeParam;
  using LocalParam = bgv::LocalParam;

  ScaleState evalMulScale(const RLWELocalParam& param, ScaleState lhs,
                          ScaleState rhs) const override;
  ScaleState evalMulScaleBackward(const RLWELocalParam& param,
                                  ScaleState result,
                                  ScaleState lhs) const override;
  ScaleState evalModReduceScale(const RLWELocalParam& inputParam,
                                ScaleState scale) const override;
  ScaleState evalModReduceScaleBackward(const RLWELocalParam& inputParam,
                                        ScaleState resultScale) const override;
};

struct CKKSScaleModel : public ScaleModel {
  using SchemeParam = ckks::SchemeParam;
  using LocalParam = ckks::LocalParam;

  ScaleState evalMulScale(const RLWELocalParam& param, ScaleState lhs,
                          ScaleState rhs) const override;
  ScaleState evalMulScaleBackward(const RLWELocalParam& param,
                                  ScaleState result,
                                  ScaleState lhs) const override;
  ScaleState evalModReduceScale(const RLWELocalParam& inputParam,
                                ScaleState scale) const override;
  ScaleState evalModReduceScaleBackward(const RLWELocalParam& inputParam,
                                        ScaleState resultScale) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCALEANALYSIS_SCALEMODEL_H_
