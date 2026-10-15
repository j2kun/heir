#include "lib/Dialect/Rotom/Utils/IterationAlignment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"     // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

//===----------------------------------------------------------------------===//
// OpSpec
//===----------------------------------------------------------------------===//

OpSpec OpSpec::elementwise(int64_t rank) {
  OpSpec spec;
  spec.numIters = rank;
  for (int64_t d = 0; d < rank; ++d) {
    spec.lhsAxisToIter.push_back(d);
    spec.rhsAxisToIter.push_back(d);
    spec.resultAxisToIter.push_back(d);
  }
  return spec;
}

OpSpec OpSpec::matmul(int64_t lhsRank, int64_t rhsRank) {
  // Iteration indices: batch..., then i (lhs rank >= 2), j (rhs rank >= 2),
  // then the contraction index k.
  const int64_t lhsBatch = std::max<int64_t>(lhsRank - 2, 0);
  const int64_t rhsBatch = std::max<int64_t>(rhsRank - 2, 0);
  const int64_t batch = std::max(lhsBatch, rhsBatch);
  const bool hasI = lhsRank >= 2, hasJ = rhsRank >= 2;
  const int64_t i = batch, j = batch + (hasI ? 1 : 0);
  const int64_t k = batch + (hasI ? 1 : 0) + (hasJ ? 1 : 0);

  OpSpec spec;
  spec.numIters = k + 1;
  for (int64_t b = 0; b < lhsBatch; ++b) {
    spec.lhsAxisToIter.push_back(batch - lhsBatch + b);
  }
  if (hasI) spec.lhsAxisToIter.push_back(i);
  spec.lhsAxisToIter.push_back(k);

  for (int64_t b = 0; b < rhsBatch; ++b) {
    spec.rhsAxisToIter.push_back(batch - rhsBatch + b);
  }
  spec.rhsAxisToIter.push_back(k);
  if (hasJ) spec.rhsAxisToIter.push_back(j);

  for (int64_t b = 0; b < batch; ++b) spec.resultAxisToIter.push_back(b);
  if (hasI) spec.resultAxisToIter.push_back(i);
  if (hasJ) spec.resultAxisToIter.push_back(j);
  return spec;
}

LogicalResult OpSpec::verify() const {
  SmallVector<bool> used(numIters, false);
  auto checkOperand = [&](ArrayRef<int64_t> axisToIter,
                          bool markUsed) -> LogicalResult {
    SmallVector<bool> seen(numIters, false);
    for (int64_t iter : axisToIter) {
      if (iter < 0 || iter >= numIters || seen[iter]) return failure();
      seen[iter] = true;
      if (markUsed) used[iter] = true;
    }
    return success();
  };
  if (failed(checkOperand(lhsAxisToIter, /*markUsed=*/true)) ||
      failed(checkOperand(rhsAxisToIter, /*markUsed=*/true)) ||
      failed(checkOperand(resultAxisToIter, /*markUsed=*/false))) {
    return failure();
  }
  if (!llvm::all_of(used, [](bool u) { return u; })) return failure();
  return success();
}

bool OpSpec::isReduced(int64_t iter) const {
  return !llvm::is_contained(resultAxisToIter, iter);
}

//===----------------------------------------------------------------------===//
// Lifting
//===----------------------------------------------------------------------===//

FailureOr<IterLayout> liftLayout(LayoutAttr layout,
                                 ArrayRef<int64_t> axisToIter) {
  if (!layout) return failure();
  SmallVector<DimAttr> dims;
  for (Attribute attr : layout.getDims()) dims.push_back(cast<DimAttr>(attr));
  const size_t ctLen = inferCtPrefixLen(dims, layout.getN());

  IterLayout out;
  out.n = layout.getN();
  // The digit ids each piece expands to, most significant first.
  SmallVector<SmallVector<int64_t>> pieceDigits(dims.size());
  for (auto [p, piece] : llvm::enumerate(dims)) {
    const int64_t size = piece.getSize();
    if (!llvm::isPowerOf2_64(size)) return failure();  // TODO: prime radix
    const int64_t numDigits = llvm::Log2_64(size);

    Digit::Kind kind = Digit::Kind::Data;
    int64_t iter = -1, lowPlace = 0;
    if (piece.isReplicate()) {
      kind = Digit::Kind::Replicate;
    } else if (piece.isGap()) {
      kind = Digit::Kind::Gap;
    } else {
      const int64_t axis = piece.getDim();
      if (axis >= static_cast<int64_t>(axisToIter.size())) return failure();
      iter = axisToIter[axis];
      // A split piece reads (index / stride) % size, so its low digit is
      // bit log2(stride) of the axis index. A lone piece reads the whole index.
      if (axisPieces(ArrayRef<DimAttr>(dims), axis).isSplit()) {
        if (!llvm::isPowerOf2_64(piece.getStride())) return failure();
        lowPlace = llvm::Log2_64(piece.getStride());
      }
    }
    for (int64_t d = numDigits - 1; d >= 0; --d) {
      Digit digit;
      digit.id = static_cast<int64_t>(out.digits.size());
      digit.kind = kind;
      digit.inCt = p < ctLen;
      digit.iter = iter;
      digit.place = kind == Digit::Kind::Data ? lowPlace + d : d;
      pieceDigits[p].push_back(digit.id);
      out.digits.push_back(digit);
    }
  }

  // Roll arguments become digit runs; from here on nothing refers to piece
  // positions.
  auto argDigits =
      [&](const RollArg& arg) -> std::optional<SmallVector<int64_t>> {
    if (!arg.isAxis) {
      if (arg.index < 0 || arg.index >= static_cast<int64_t>(dims.size())) {
        return std::nullopt;
      }
      return pieceDigits[arg.index];
    }
    if (arg.index >= static_cast<int64_t>(axisToIter.size())) {
      return std::nullopt;
    }
    const int64_t iter = axisToIter[arg.index];
    SmallVector<int64_t> run;
    for (const Digit& d : out.digits) {
      if (d.kind == Digit::Kind::Data && d.iter == iter) run.push_back(d.id);
    }
    llvm::stable_sort(run, [&](int64_t a, int64_t b) {
      return out.digits[a].place > out.digits[b].place;
    });
    return run;
  };
  for (const RollSpec& roll : getRollSpecs(layout)) {
    auto from = argDigits(roll.from);
    auto by = argDigits(roll.by);
    if (!from || !by) return failure();
    out.rolls.push_back({std::move(*from), std::move(*by)});
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Alignment
//===----------------------------------------------------------------------===//

namespace {

// What a digit means once both operands are in view. A replication digit
// facing data enumerates that data's iteration index (it is the broadcast
// copy), so it gets the data's meaning. Unbound replication and gaps have no
// index; they are identified by position, which is shared by both sides.
using Meaning = std::tuple<Digit::Kind, int64_t, int64_t>;  // kind, iter, place

struct SideView {
  const IterLayout* layout;
  ArrayRef<int64_t> operandIters;
  // For replication digits facing a data digit: that digit's meaning.
  llvm::DenseMap<int64_t, Meaning> bound;

  Meaning meaningOf(int64_t id) const {
    const Digit& d = layout->digits[id];
    if (d.kind == Digit::Kind::Data) return {d.kind, d.iter, d.place};
    if (auto it = bound.find(id); it != bound.end()) return it->second;
    return {d.kind, -1, id};  // id == position in a lifted layout
  }

  SmallVector<Meaning> meaningOf(ArrayRef<int64_t> ids) const {
    SmallVector<Meaning> out;
    for (int64_t id : ids) out.push_back(meaningOf(id));
    return out;
  }
};

using RollMeaning = std::pair<SmallVector<Meaning>, SmallVector<Meaning>>;

}  // namespace

FailureOr<AlignedPair> align(const OpSpec& spec, LayoutAttr lhs,
                             LayoutAttr rhs) {
  if (!lhs || !rhs || lhs.getN() != rhs.getN()) return failure();
  if (failed(spec.verify())) return failure();
  FailureOr<IterLayout> l = liftLayout(lhs, spec.lhsAxisToIter);
  FailureOr<IterLayout> r = liftLayout(rhs, spec.rhsAxisToIter);
  if (failed(l) || failed(r)) return failure();
  if (l->digits.size() != r->digits.size()) return failure();

  SideView lv{&*l, spec.lhsAxisToIter, llvm::DenseMap<int64_t, Meaning>()};
  SideView rv{&*r, spec.rhsAxisToIter, llvm::DenseMap<int64_t, Meaning>()};

  AlignedPair pair;
  pair.lhs_ = lhs;
  pair.rhs_ = rhs;
  for (size_t t = 0; t < l->digits.size(); ++t) {
    const Digit& a = l->digits[t];
    const Digit& b = r->digits[t];
    if (a.inCt != b.inCt) return failure();
    Digit product;
    product.id = static_cast<int64_t>(t);
    product.inCt = a.inCt;
    using K = Digit::Kind;
    if (a.kind == K::Gap || b.kind == K::Gap) {
      // A gap never faces data or replication.
      if (a.kind != b.kind) return failure();
      product.kind = K::Gap;
    } else if (a.kind == K::Data && b.kind == K::Data) {
      if (a.iter != b.iter || a.place != b.place) return failure();
      product = a;
      product.id = static_cast<int64_t>(t);
    } else if (a.kind == K::Data || b.kind == K::Data) {
      // Data faces replication: legal only if the replicated operand does not
      // have that index at all, i.e. the replication is its broadcast.
      const bool lhsHasData = a.kind == K::Data;
      const Digit& data = lhsHasData ? a : b;
      SideView& replSide = lhsHasData ? rv : lv;
      if (llvm::is_contained(replSide.operandIters, data.iter)) {
        return failure();
      }
      replSide.bound[static_cast<int64_t>(t)] = {K::Data, data.iter,
                                                 data.place};
      product = data;
      product.id = static_cast<int64_t>(t);
    } else {
      product.kind = K::Replicate;
    }
    pair.productDigits_.push_back(product);
  }

  // Rolls. A roll whose FROM enumerates only indices the other operand lacks
  // is exempt: the other side holds replication there, so every rotation still
  // meets the same value. All other rolls must match in meaning.
  auto isExempt = [](const SideView& self, const SideView& other,
                     const DigitRoll& roll) {
    return llvm::all_of(roll.from, [&](int64_t id) {
      auto [kind, iter, place] = self.meaningOf(id);
      return kind == Digit::Kind::Data &&
             !llvm::is_contained(other.operandIters, iter);
    });
  };
  SmallVector<RollMeaning> lhsRequired, rhsRequired, carried;
  for (const DigitRoll& roll : l->rolls) {
    RollMeaning m{lv.meaningOf(roll.from), lv.meaningOf(roll.by)};
    (isExempt(lv, rv, roll) ? carried : lhsRequired).push_back(m);
  }
  for (const DigitRoll& roll : r->rolls) {
    RollMeaning m{rv.meaningOf(roll.from), rv.meaningOf(roll.by)};
    (isExempt(rv, lv, roll) ? carried : rhsRequired).push_back(m);
  }
  llvm::sort(lhsRequired);
  llvm::sort(rhsRequired);
  if (lhsRequired != rhsRequired) return failure();
  llvm::append_range(carried, lhsRequired);

  // Restate every carried roll over product digit ids (== positions).
  auto productId = [&](const Meaning& m) -> std::optional<int64_t> {
    auto [kind, iter, place] = m;
    if (kind != Digit::Kind::Data) return place;  // position, see meaningOf
    for (const Digit& d : pair.productDigits_) {
      if (d.kind == Digit::Kind::Data && d.iter == iter && d.place == place) {
        return d.id;
      }
    }
    return std::nullopt;
  };
  for (const RollMeaning& m : carried) {
    DigitRoll roll;
    for (const Meaning& x : m.first) {
      std::optional<int64_t> id = productId(x);
      if (!id) return failure();
      roll.from.push_back(*id);
    }
    for (const Meaning& x : m.second) {
      std::optional<int64_t> id = productId(x);
      if (!id) return failure();
      roll.by.push_back(*id);
    }
    pair.productRolls_.push_back(std::move(roll));
  }
  return pair;
}

//===----------------------------------------------------------------------===//
// Output layout
//===----------------------------------------------------------------------===//

FailureOr<LayoutAttr> outputLayout(const OpSpec& spec,
                                   const AlignedPair& pair) {
  llvm::DenseMap<int64_t, int64_t> iterToResultAxis;
  for (auto [axis, iter] : llvm::enumerate(spec.resultAxisToIter)) {
    iterToResultAxis[iter] = static_cast<int64_t>(axis);
  }

  IterLayout out;
  out.n = pair.lhs().getN();
  // productDigits id -> out digit id, or -1 when the digit is removed.
  SmallVector<int64_t> outIdOf(pair.productDigits().size(), -1);
  auto isReducedData = [&](const Digit& d) {
    return d.kind == Digit::Kind::Data && spec.isReduced(d.iter);
  };
  for (const Digit& d : pair.productDigits()) {
    Digit o = d;
    if (isReducedData(d)) {
      // The reduction adds the ciphertexts a reduced ct digit enumerates, so
      // the digit disappears; in the slots only one offset keeps the sum.
      if (d.inCt) continue;
      o.kind = Digit::Kind::Gap;
      o.iter = -1;
      o.place = 0;
    } else if (d.kind == Digit::Kind::Data) {
      o.iter = iterToResultAxis.lookup(d.iter);  // now a result axis
    }
    o.id = static_cast<int64_t>(out.digits.size());
    outIdOf[d.id] = o.id;
    out.digits.push_back(o);
  }

  for (const DigitRoll& roll : pair.productRolls()) {
    auto touchesReduced = [&](ArrayRef<int64_t> ids) {
      return llvm::any_of(ids, [&](int64_t id) {
        return isReducedData(pair.productDigits()[id]);
      });
    };
    // Summing over a rolled index sums the same terms in another order.
    if (touchesReduced(roll.from)) continue;
    // A shift by an index that no longer exists has no layout.
    if (touchesReduced(roll.by)) return failure();
    DigitRoll o;
    for (int64_t id : roll.from) o.from.push_back(outIdOf[id]);
    for (int64_t id : roll.by) o.by.push_back(outIdOf[id]);
    out.rolls.push_back(std::move(o));
  }
  return encodeLayout(pair.lhs().getContext(), out);
}

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

FailureOr<LayoutAttr> encodeLayout(MLIRContext* ctx, const IterLayout& layout) {
  const ArrayRef<Digit> digits = layout.digits;
  // Positions where a new piece must start: region changes and the edges of
  // every roll argument.
  SmallVector<bool> forcedStart(digits.size() + 1, false);
  llvm::DenseMap<int64_t, int64_t> posOf;
  for (auto [pos, d] : llvm::enumerate(digits)) posOf[d.id] = pos;
  auto markRun = [&](ArrayRef<int64_t> ids) -> LogicalResult {
    if (ids.empty()) return failure();
    for (int64_t id : ids) {
      if (!posOf.count(id)) return failure();
    }
    forcedStart[posOf[ids.front()]] = true;
    forcedStart[posOf[ids.back()] + 1] = true;
    return success();
  };
  for (const DigitRoll& roll : layout.rolls) {
    if (failed(markRun(roll.from)) || failed(markRun(roll.by))) {
      return failure();
    }
  }

  struct Piece {
    Digit::Kind kind;
    int64_t axis;
    int64_t start;
    int64_t numDigits;
    int64_t lowPlace;
  };
  SmallVector<Piece> pieces;
  for (auto [pos, d] : llvm::enumerate(digits)) {
    const bool extends = !pieces.empty() && !forcedStart[pos] &&
                         digits[pos - 1].inCt == d.inCt &&
                         pieces.back().kind == d.kind &&
                         (d.kind != Digit::Kind::Data ||
                          (pieces.back().axis == d.iter &&
                           pieces.back().lowPlace == d.place + 1));
    if (extends) {
      ++pieces.back().numDigits;
      pieces.back().lowPlace = d.place;
      continue;
    }
    pieces.push_back({d.kind, d.iter, static_cast<int64_t>(pos), 1, d.place});
  }

  llvm::DenseMap<int64_t, int64_t> piecesPerAxis;
  for (const Piece& p : pieces) {
    if (p.kind == Digit::Kind::Data) ++piecesPerAxis[p.axis];
  }
  SmallVector<DimAttr> dims;
  for (const Piece& p : pieces) {
    const int64_t size = int64_t{1} << p.numDigits;
    switch (p.kind) {
      case Digit::Kind::Replicate:
        dims.push_back(DimAttr::get(ctx, /*dim=*/-1, size, /*stride=*/1));
        break;
      case Digit::Kind::Gap:
        dims.push_back(DimAttr::get(ctx, /*dim=*/-2, size, /*stride=*/1));
        break;
      case Digit::Kind::Data: {
        const bool split = piecesPerAxis[p.axis] > 1;
        if (!split && p.lowPlace != 0) return failure();
        const int64_t stride = split ? int64_t{1} << p.lowPlace : 1;
        dims.push_back(DimAttr::get(ctx, p.axis, size, stride));
        break;
      }
    }
  }

  // Roll arguments: exactly one piece, or every digit of a split axis.
  auto encodeArg = [&](ArrayRef<int64_t> ids) -> std::optional<int64_t> {
    const int64_t first = posOf[ids.front()];
    bool contiguous = true;
    for (auto [k, id] : llvm::enumerate(ids)) {
      contiguous &= posOf[id] == first + static_cast<int64_t>(k);
    }
    for (auto [pi, p] : llvm::enumerate(pieces)) {
      if (contiguous && p.start == first &&
          p.numDigits == static_cast<int64_t>(ids.size())) {
        return static_cast<int64_t>(pi);
      }
    }
    const Digit& d = digits[first];
    if (d.kind != Digit::Kind::Data || piecesPerAxis[d.iter] < 2) {
      return std::nullopt;
    }
    int64_t axisDigits = 0;
    for (const Digit& x : digits) {
      if (x.kind == Digit::Kind::Data && x.iter == d.iter) ++axisDigits;
    }
    if (axisDigits != static_cast<int64_t>(ids.size())) return std::nullopt;
    return encodeRollArg({/*isAxis=*/true, d.iter});
  };
  SmallVector<int64_t> rolls;
  for (const DigitRoll& roll : layout.rolls) {
    std::optional<int64_t> from = encodeArg(roll.from);
    std::optional<int64_t> by = encodeArg(roll.by);
    if (!from || !by) return failure();
    rolls.push_back(*from);
    rolls.push_back(*by);
  }
  // TODO: verify the result once verification without diagnostics exists.
  return LayoutAttr::getCanonical(ctx, dims, layout.n, rolls);
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
