#ifndef LIB_DIALECT_ROTOM_UTILS_ITERATIONALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_ITERATIONALIGNMENT_H_

// Design sketch: operator alignment over the operator's iteration space.
//
// This is an alternative formulation of LayoutAlignment.h. It reasons about
// what a layout *means* instead of about the syntax of its piece list:
//
//   1. An operator is described by an einsum-style OpSpec: each lhs, rhs, and
//      result axis names an iteration index. Reduced indices, result axis
//      numbering, and broadcast are all derived from that one description.
//   2. Each operand layout is lifted into a fixed-granularity normal form
//      (IterLayout): every piece is split into radix-2 digits, and every data
//      digit names an (iteration index, bit place) pair. Equivalent spellings
//      of one packing (split vs merged, regrouped replication) lift to the same
//      digit list, so comparisons are positionwise equality.
//   3. Rolls name their arguments by digit id, never by list position, so no
//      transform has to re-index them. Positions are computed exactly once,
//      when a result is encoded back into a LayoutAttr.
//   4. Alignment is checked once, by align(), which returns an AlignedPair
//      that carries the product's digits and rolls. outputLayout consumes that
//      pair, so it cannot be called on an unaligned or mismatched input.
//
// Scope of the sketch: the alignment check and the output layout. The
// candidate generators of LayoutAlignment.h (mirrorDims, replicateForAlignment,
// applySumRoll, rollToAlign, matchPublicLayout) would become transforms on
// IterLayout followed by encodeLayout(); they are not implemented here.
// Ciphertext pieces must have power-of-two sizes (TODO: prime-radix digits).

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

// An operator as an einsum over iteration indices 0..numIters-1. For example
// matmul C[i,j] = sum_k A[i,k] * B[k,j] is lhs {i,k}, rhs {k,j}, result {i,j};
// k is reduced because the result does not name it.
struct OpSpec {
  int64_t numIters = 0;
  SmallVector<int64_t> lhsAxisToIter;
  SmallVector<int64_t> rhsAxisToIter;
  SmallVector<int64_t> resultAxisToIter;

  static OpSpec elementwise(int64_t rank);
  // [..batch, m, k] x [..batch, k, n]; a rank-1 operand is [k]. Batch axes
  // align from the right; a batch axis only one side has is broadcast on the
  // other.
  static OpSpec matmul(int64_t lhsRank, int64_t rhsRank);

  // Every entry is in range, no operand names an iteration index twice, every
  // index is used by an operand, and the result names only operand indices.
  LogicalResult verify() const;
  bool isReduced(int64_t iter) const;
};

// One radix-2 digit of a lifted layout.
struct Digit {
  enum class Kind : uint8_t { Data, Replicate, Gap };
  // Stable identity within its IterLayout; roll arguments refer to it.
  int64_t id = -1;
  Kind kind = Kind::Gap;
  bool inCt = false;
  // Data only: the iteration index this digit enumerates, and which bit of
  // that index (0 = least significant). Stride order is part of equality.
  int64_t iter = -1;
  int64_t place = 0;
};

// A roll whose arguments are digit runs, most significant digit first.
struct DigitRoll {
  SmallVector<int64_t> from;
  SmallVector<int64_t> by;
};

// A layout in digit normal form: ciphertext digits precede slot digits, and
// within each region digits are listed outermost first.
struct IterLayout {
  int64_t n = 0;
  SmallVector<Digit> digits;
  SmallVector<DigitRoll> rolls;
};

// Lifts `layout` into iteration space, renaming operand axis a to iteration
// index axisToIter[a]. Fails if the layout names an axis outside axisToIter or
// has a piece whose size is not a power of two.
FailureOr<IterLayout> liftLayout(LayoutAttr layout,
                                 ArrayRef<int64_t> axisToIter);

// A pair of operand layouts proven aligned for an OpSpec. Only align() can
// construct one; it carries what the product of the two operands holds at
// every position, so consumers never re-derive the correspondence.
class AlignedPair {
 public:
  LayoutAttr lhs() const { return lhs_; }
  LayoutAttr rhs() const { return rhs_; }
  // What the elementwise product holds at each position. Digit ids equal
  // positions.
  ArrayRef<Digit> productDigits() const { return productDigits_; }
  // Every roll the product carries, over productDigits ids.
  ArrayRef<DigitRoll> productRolls() const { return productRolls_; }

 private:
  friend FailureOr<AlignedPair> align(const OpSpec& spec, LayoutAttr lhs,
                                      LayoutAttr rhs);
  AlignedPair() = default;
  LayoutAttr lhs_, rhs_;
  SmallVector<Digit> productDigits_;
  SmallVector<DigitRoll> productRolls_;
};

// Succeeds iff the two layouts place every element pair the operator combines
// at the same (ciphertext, slot). Position by position, after lifting:
//   - gap faces gap;
//   - a data digit faces the same (iteration index, place) on the other side,
//     or replication on a side whose operand lacks that index;
//   - replication faces replication.
// A roll whose FROM enumerates only indices the other operand lacks is exempt
// (the other side is replicated there); every other roll must appear on both
// sides with the same meaning.
FailureOr<AlignedPair> align(const OpSpec& spec, LayoutAttr lhs,
                             LayoutAttr rhs);

inline bool isAligned(const OpSpec& spec, LayoutAttr lhs, LayoutAttr rhs) {
  return succeeded(align(spec, lhs, rhs));
}

// The layout of the operator's result: the product's digits with reduced
// ciphertext digits removed (the reduction adds those ciphertexts), reduced
// slot digits turned into gaps, rolls on reduced indices dropped, and
// iteration indices renamed to result axes. Fails if a surviving roll shifts
// by a reduced index, which no layout can express.
FailureOr<LayoutAttr> outputLayout(const OpSpec& spec, const AlignedPair& pair);

// Encodes digits back into a LayoutAttr. Data digits name tensor axes here.
// Adjacent digits merge into one piece unless a roll argument needs a piece
// boundary between them.
FailureOr<LayoutAttr> encodeLayout(MLIRContext* ctx, const IterLayout& layout);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_ITERATIONALIGNMENT_H_
