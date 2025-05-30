#ifndef LIB_PARAMETERS_PLAINTEXTPARAMS_H_
#define LIB_PARAMETERS_PLAINTEXTPARAMS_H_

#include <cstdint>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

// Parameter for Plaintext backend ModuleOp level. This should only be used for
// the plaintext pipeline, which is for debugging programs.
class PlaintextSchemeParam {
 public:
  PlaintextSchemeParam(int logDefaultScale)
      : logDefaultScale(logDefaultScale) {}

 private:
  // log of the default scale used to scale the message
  int64_t logDefaultScale;

 public:
  int64_t getLogDefaultScale() const { return logDefaultScale; }
  void print(llvm::raw_ostream &os) const;
};

// Parameter for each plaintext SSA value.
class LocalParam {
 public:
  LocalParam(int64_t currentLogScale) : currentLogScale(currentLogScale) {}

 private:
  int64_t currentLogScale;

 public:
  int64_t getCurrentLogScale() const { return currentLogScale; }
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_PLAINTEXTPARAMS_H_
