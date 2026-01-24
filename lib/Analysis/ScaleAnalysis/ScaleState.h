#ifndef LIB_ANALYSIS_SCALEANALYSIS_SCALESTATE_H_
#define LIB_ANALYSIS_SCALEANALYSIS_SCALESTATE_H_

#include <cassert>
#include <cstdint>
#include <variant>

#include "lib/Utils/Overloaded.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project

namespace mlir {
namespace heir {
namespace scale {

// A sentinel for an undetermined scale value. This is used to signify
// a plaintext operand (or the output of an adjust scale op) for which
// the scale may be chosen by the user.
struct Free {
  bool operator==(const Free&) const = default;
};

class ScaleState {
 public:
  // Currently we use logScale for CKKS.
  // TODO(#1640): support high-precision scale management
  using ScaleType = std::variant<Free, int64_t>;

  ScaleState() : value(Free{}) {}
  explicit ScaleState(ScaleType scale) : value(scale) {}
  ScaleState(int64_t scale) : value(scale) {}
  ~ScaleState() = default;

  ScaleType getScale() const { return value; }
  void setScale(ScaleType val) { value = val; }
  ScaleType get() const { return getScale(); }

  bool operator==(const ScaleState& rhs) const = default;

  bool isFree() const { return std::holds_alternative<Free>(value); }
  bool isInt() const { return std::holds_alternative<int64_t>(value); }
  int64_t getInt() const { return std::get<int64_t>(value); }

  static ScaleState join(const ScaleState& lhs, const ScaleState& rhs) {
    return std::visit(
        Overloaded{
            [](Free, auto other) -> ScaleState { return ScaleState(other); },
            [](int64_t lhs, Free) -> ScaleState { return ScaleState(lhs); },
            [](int64_t lhs, int64_t rhs) -> ScaleState {
              return ScaleState(std::max(lhs, rhs));
            },
        },
        lhs.value, rhs.value);
  }

  void print(llvm::raw_ostream& os) const {
    std::visit(Overloaded{[&](Free) { os << "Scale(Free)"; },
                          [&](int64_t val) { os << "Scale(" << val << ")"; }},
               value);
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const ScaleState& state) {
    state.print(os);
    return os;
  }

 private:
  ScaleType value;
};

}  // namespace scale
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCALEANALYSIS_SCALESTATE_H_
