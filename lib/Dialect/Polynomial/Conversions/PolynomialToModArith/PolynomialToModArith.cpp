#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/Polynomial.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMAttrs.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"             // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

using namespace mlir::heir::polynomial;
using namespace mlir::heir::mod_arith;

#define DEBUG_TYPE "polynomial-to-mod-arith"

#define GEN_PASS_DEF_POLYNOMIALTOMODARITH
#include "lib/Dialect/Polynomial/Conversions/PolynomialToModArith/PolynomialToModArith.h.inc"

// Callback type for getting pre-generated FuncOp implementing
// helper functions for various lowerings.
using GetFuncCallbackTy = function_ref<func::FuncOp(FunctionType, RingAttr)>;

RankedTensorType convertPolynomialType(PolynomialType type) {
  RingAttr attr = type.getRing();
  // We must remove the ring attribute on the tensor, since the
  // unrealized_conversion_casts cannot carry the poly.ring attribute
  // through.
  auto degree = attr.getPolynomialModulus().getPolynomial().getDegree();
  return RankedTensorType::get({degree}, attr.getCoefficientType());
}

struct CommonConversionInfo {
  PolynomialType polynomialType;
  RingAttr ringAttr;

  Type coefficientType;
  Type coefficientStorageType;

  APInt coefficientModulusValue;
  unsigned coefficientModulusWidth;

  // Poly -> tensor converted type
  RankedTensorType tensorType;
};

FailureOr<CommonConversionInfo> getCommonConversionInfo(
    Operation *op, const TypeConverter *typeConverter,
    std::optional<Type> polyType = std::nullopt) {
  // Most ops have a single result type that is a polynomial
  PolynomialType polyTy;
  if (polyType.has_value()) {
    polyTy = dyn_cast<PolynomialType>(polyType.value());
  } else {
    polyTy = dyn_cast<PolynomialType>(op->getResult(0).getType());
  }

  if (!polyTy) {
    op->emitError(
        "Can't directly lower for a tensor of polynomials. "
        "First run --convert-elementwise-to-affine.");
    return failure();
  }

  CommonConversionInfo info;
  info.polynomialType = polyTy;
  info.ringAttr = info.polynomialType.getRing();
  info.coefficientType = info.ringAttr.getCoefficientType();
  info.tensorType = cast<RankedTensorType>(typeConverter->convertType(polyTy));

  FailureOr<Type> res =
      llvm::TypeSwitch<Type, FailureOr<Type>>(info.coefficientType)
          .Case<IntegerType>([&](auto intTy) { return intTy; })
          .Case<ModArithType>(
              [&](ModArithType intTy) { return intTy.getModulus().getType(); })
          .Default([&](Type ty) { return failure(); });
  if (failed(res)) {
    assert(false && "unsupported coefficient type");
  }
  info.coefficientStorageType = res.value();
  return std::move(info);
}

Value getConstantCoefficient(Type type, int64_t value,
                             ImplicitLocOpBuilder &builder) {
  return llvm::TypeSwitch<Type, Value>(type)
      .Case<IntegerType>([&](auto intTy) {
        return builder.create<arith::ConstantOp>(
            builder.getIntegerAttr(intTy, value));
      })
      .Case<ModArithType>([&](ModArithType modTy) {
        return builder.create<mod_arith::ConstantOp>(
            ModArithAttr::get(modTy, value));
      })
      .Default([&](Type ty) {
        assert(false && "unsupported coefficient type");
        return Value();
      });
}

std::pair<APInt, APInt> extendWidthsToLargest(const APInt &a, const APInt &b) {
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth());
  return {a.zextOrTrunc(width), b.zextOrTrunc(width)};
}

/// Cloned after upstream removal in
/// https://github.com/llvm/llvm-project/pull/87644
///
/// Computes the multiplicative inverse of this APInt for a given modulo. The
/// iterative extended Euclidean algorithm is used to solve for this value,
/// however we simplify it to speed up calculating only the inverse, and take
/// advantage of div+rem calculations. We also use some tricks to avoid copying
/// (potentially large) APInts around.
/// WARNING: a value of '0' may be returned,
///          signifying that no multiplicative inverse exists!
APInt multiplicativeInverse(const APInt &x, const APInt &modulo) {
  assert(x.ult(modulo) && "This APInt must be smaller than the modulo");
  // Using the properties listed at the following web page (accessed 06/21/08):
  //   http://www.numbertheory.org/php/euclid.html
  // (especially the properties numbered 3, 4 and 9) it can be proved that
  // BitWidth bits suffice for all the computations in the algorithm implemented
  // below. More precisely, this number of bits suffice if the multiplicative
  // inverse exists, but may not suffice for the general extended Euclidean
  // algorithm.

  auto BitWidth = x.getBitWidth();
  APInt r[2] = {modulo, x};
  APInt t[2] = {APInt(BitWidth, 0), APInt(BitWidth, 1)};
  APInt q(BitWidth, 0);

  unsigned i;
  for (i = 0; r[i ^ 1] != 0; i ^= 1) {
    // An overview of the math without the confusing bit-flipping:
    // q = r[i-2] / r[i-1]
    // r[i] = r[i-2] % r[i-1]
    // t[i] = t[i-2] - t[i-1] * q
    x.udivrem(r[i], r[i ^ 1], q, r[i]);
    t[i] -= t[i ^ 1] * q;
  }

  // If this APInt and the modulo are not coprime, there is no multiplicative
  // inverse, so return 0. We check this by looking at the next-to-last
  // remainder, which is the gcd(*this,modulo) as calculated by the Euclidean
  // algorithm.
  if (r[i] != 1) return APInt(BitWidth, 0);

  // The next-to-last t is the multiplicative inverse.  However, we are
  // interested in a positive inverse. Calculate a positive one from a negative
  // one if necessary. A simple addition of the modulo suffices because
  // abs(t[i]) is known to be less than *this/2 (see the link above).
  if (t[i].isNegative()) t[i] += modulo;

  return std::move(t[i]);
}

class PolynomialToModArithTypeConverter : public TypeConverter {
 public:
  PolynomialToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PolynomialType type) -> Type {
      return convertPolynomialType(type);
    });

    // We don't include any custom materialization ops because this lowering is
    // all done in a single pass. The dialect conversion framework works by
    // resolving intermediate (mid-pass) type conflicts by inserting
    // unrealized_conversion_cast ops, and only converting those to custom
    // materializations if they persist at the end of the pass. In our case,
    // we'd only need to use custom materializations if we split this lowering
    // across multiple passes.
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto resultShape = typeInfo.tensorType.getShape()[0];
    auto resultEltTy = typeInfo.tensorType.getElementType();
    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getInput();

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));

      auto padValue = getConstantCoefficient(resultEltTy, 0, b);
      coeffValue = b.create<tensor::PadOp>(typeInfo.tensorType, coeffValue, low,
                                           high, padValue,
                                           /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  ConvertToTensor(mlir::MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto attr = dyn_cast<TypedIntPolynomialAttr>(op.getValue());
    if (!attr) return failure();
    SmallVector<Attribute> coeffs;
    Type eltStorageType = typeInfo.coefficientStorageType;

    // Create all the attributes as arith types since mod_arith.constant
    // doesn't support tensor attribute inputs. Instead we
    // mod_arith.encapsulate them.
    //
    // This is inefficient for large-degree polys, but as of this writing we
    // don't have a lowering that uses a sparse representation.
    unsigned numTerms = typeInfo.tensorType.getShape()[0];
    coeffs.reserve(numTerms);
    for (size_t i = 0; i < numTerms; ++i) {
      coeffs.push_back(IntegerAttr::get(eltStorageType, 0));
    }

    // WARNING: if you don't store the IntPolynomial as an intermediate value
    // before iterating over the terms, you will get a use-after-free bug. See
    // the "Temporary range expression" section in
    // https://en.cppreference.com/w/cpp/language/range-for
    const IntPolynomial &poly = attr.getValue().getPolynomial();
    for (const auto &term : poly.getTerms()) {
      int64_t idx = term.getExponent().getSExtValue();
      APInt coeff = term.getCoefficient();
      // If the coefficient type is a mod_arith type, then we need to ensure
      // the input is normalized properly to the modulus. I.e., if the
      // polynomial coefficient is a literal -1 and the mod_arith modulus is 7
      // : i32, then -1 equiv 6 mod 7, but -1 as an i32 is 2147483647 equiv 1
      // mod 7.
      if (auto modArithType =
              dyn_cast<ModArithType>(typeInfo.coefficientType)) {
        APInt modulus = modArithType.getModulus().getValue();
        // APInt srem gives remainder with sign matching the sign of the
        // context argument (here, it's the sign of coeff)
        coeff = coeff.sextOrTrunc(modulus.getBitWidth()).srem(modulus);
        if (coeff.isNegative()) {
          // We need to add the modulus to get the positive remainder.
          coeff += modulus;
        }
        assert(coeff.sge(0));
      }
      coeffs[idx] = IntegerAttr::get(eltStorageType, coeff.getSExtValue());
    }

    return llvm::TypeSwitch<Type, LogicalResult>(typeInfo.coefficientType)
        .Case<IntegerType>([&](auto intTy) {
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(
              op, DenseElementsAttr::get(typeInfo.tensorType, coeffs));
          return success();
        })
        .Case<ModArithType>([&](ModArithType intTy) {
          auto intTensorType = RankedTensorType::get(
              typeInfo.tensorType.getShape(), intTy.getModulus().getType());
          auto constOp = b.create<arith::ConstantOp>(
              DenseElementsAttr::get(intTensorType, coeffs));
          rewriter.replaceOpWithNewOp<mod_arith::EncapsulateOp>(
              op, typeInfo.tensorType, constOp.getResult());
          return success();
        })
        .Default([&](Type ty) {
          op.emitError("unsupported coefficient type: ") << ty;
          return failure();
        });
    return success();
  }
};

struct ConvertMonomial : public OpConversionPattern<MonomialOp> {
  ConvertMonomial(mlir::MLIRContext *context)
      : OpConversionPattern<MonomialOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MonomialOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto storageTensorType = RankedTensorType::get(
        typeInfo.tensorType.getShape(), typeInfo.coefficientStorageType);
    auto tensor = b.create<arith::ConstantOp>(DenseElementsAttr::get(
        storageTensorType,
        b.getIntegerAttr(typeInfo.coefficientStorageType, 0)));

    Value result = tensor.getResult();
    if (isa<ModArithType>(
            typeInfo.polynomialType.getRing().getCoefficientType())) {
      result = b.create<mod_arith::EncapsulateOp>(typeInfo.tensorType, tensor)
                   .getResult();
    }
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getCoefficient(),
                                                  result, adaptor.getDegree());
    return success();
  }
};

struct ConvertMulScalar : public OpConversionPattern<MulScalarOp> {
  ConvertMulScalar(mlir::MLIRContext *context)
      : OpConversionPattern<MulScalarOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulScalarOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto coeffType = dyn_cast<ModArithType>(typeInfo.coefficientType);
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // SplatOp only accepts integer/float inputs, so we can't splat a mod_arith
    // directly.
    auto storageTensorType = RankedTensorType::get(
        typeInfo.tensorType.getShape(), coeffType.getModulus().getType());
    auto tensor = b.create<tensor::SplatOp>(
        b.create<mod_arith::ExtractOp>(storageTensorType.getElementType(),
                                       adaptor.getScalar()),
        storageTensorType);
    auto modArithTensor =
        b.create<mod_arith::EncapsulateOp>(typeInfo.tensorType, tensor);
    auto mulOp =
        b.create<mod_arith::MulOp>(adaptor.getPolynomial(), modArithTensor);
    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

/// Returns true if and only if the polynomial modulus of the ring for this op
/// is of the form x^n - 1 for some n. This is "cyclic" in the sense that
/// multiplication by a monomial corresponds to a cyclic shift of the
/// coefficients.
bool hasCyclicModulus(MonicMonomialMulOp op) {
  auto ring = cast<PolynomialType>(op.getInput().getType()).getRing();
  IntPolynomial ideal = ring.getPolynomialModulus().getPolynomial();
  auto idealTerms = ideal.getTerms();
  APInt constantCoeff = idealTerms[0].getCoefficient();
  return idealTerms.size() == 2 &&
         (constantCoeff == APInt(constantCoeff.getBitWidth(), -1) &&
          idealTerms[0].getExponent().isZero()) &&
         (idealTerms[1].getCoefficient().isOne());
}

// Implement rotation via tensor.insert_slice
struct ConvertMonicMonomialMul
    : public OpConversionPattern<MonicMonomialMulOp> {
  ConvertMonicMonomialMul(mlir::MLIRContext *context)
      : OpConversionPattern<MonicMonomialMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MonicMonomialMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!hasCyclicModulus(op)) {
      // When upstreaming, this should be expanded to support all inputs.
      InFlightDiagnostic diag =
          op.emitError()
          << "unsupported ideal for monic monomial multiplication:"
          << cast<PolynomialType>(op.getInput().getType())
                 .getRing()
                 .getPolynomialModulus();
      diag.attachNote() << "expected x**n - 1 for some n";
      return diag;
    }

    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // In general, a rotation would correspond to multiplication by x^n,
    // which requires a modular reduction step. But because the verifier
    // requires the ring to have a specific structure (x^n - 1), this op
    // can be implemented as a cyclic shift with wraparound.
    auto outputTensorContainer = b.create<tensor::EmptyOp>(
        typeInfo.tensorType.getShape(), typeInfo.tensorType.getElementType());

    RankedTensorType dynamicSliceType = RankedTensorType::get(
        ShapedType::kDynamic, typeInfo.tensorType.getElementType());

    // split the tensor into two pieces at index N - rotation_amount
    // e.g., if rotation_amount is 2,
    //
    //   [0, 1, 2, 3, 4 | 5, 6]
    //
    // the resulting output is
    //
    //   [5, 6 | 0, 1, 2, 3, 4]
    auto constTensorDim = b.create<arith::ConstantOp>(
        b.getIndexType(), b.getIndexAttr(typeInfo.tensorType.getShape()[0]));
    auto splitPoint =
        b.create<arith::SubIOp>(constTensorDim, adaptor.getMonomialDegree());

    SmallVector<OpFoldResult> firstHalfExtractOffsets{b.getIndexAttr(0)};
    SmallVector<OpFoldResult> firstHalfExtractSizes{splitPoint.getResult()};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1)};
    auto firstHalfExtractOp = b.create<tensor::ExtractSliceOp>(
        /*resultType=*/dynamicSliceType,
        /*source=*/adaptor.getInput(), firstHalfExtractOffsets,
        firstHalfExtractSizes, strides);

    SmallVector<OpFoldResult> secondHalfExtractOffsets{splitPoint.getResult()};
    SmallVector<OpFoldResult> secondHalfExtractSizes{
        adaptor.getMonomialDegree()};
    auto secondHalfExtractOp = b.create<tensor::ExtractSliceOp>(
        /*resultType=*/dynamicSliceType,
        /*source=*/adaptor.getInput(), secondHalfExtractOffsets,
        secondHalfExtractSizes, strides);

    SmallVector<OpFoldResult> firstHalfInsertOffsets{
        adaptor.getMonomialDegree()};
    SmallVector<OpFoldResult> firstHalfInsertSizes{splitPoint.getResult()};
    auto firstHalfInsertOp = b.create<tensor::InsertSliceOp>(
        /*source=*/firstHalfExtractOp.getResult(),
        /*dest=*/outputTensorContainer.getResult(), firstHalfInsertOffsets,
        firstHalfInsertSizes, strides);

    SmallVector<OpFoldResult> secondHalfInsertOffsets{b.getIndexAttr(0)};
    SmallVector<OpFoldResult> secondHalfInsertSizes{
        adaptor.getMonomialDegree()};
    auto secondHalfInsertOp = b.create<tensor::InsertSliceOp>(
        /*source=*/secondHalfExtractOp.getResult(),
        /*dest=*/firstHalfInsertOp.getResult(), secondHalfInsertOffsets,
        secondHalfInsertSizes, strides);

    rewriter.replaceOp(op, secondHalfInsertOp.getResult());
    return success();
  }
};

struct ConvertLeadingTerm : public OpConversionPattern<LeadingTermOp> {
  ConvertLeadingTerm(mlir::MLIRContext *context)
      : OpConversionPattern<LeadingTermOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LeadingTermOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffs = adaptor.getInput();
    auto tensorType = cast<RankedTensorType>(coeffs.getType());

    auto res =
        getCommonConversionInfo(op, typeConverter, op.getInput().getType());
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto c0 = b.create<arith::ConstantOp>(
        b.getIntegerAttr(typeInfo.coefficientStorageType, 0));
    auto c1 = b.create<arith::ConstantOp>(b.getIndexAttr(1));
    auto initIndex = b.create<arith::ConstantOp>(
        b.getIndexAttr(tensorType.getShape()[0] - 1));

    auto degreeOp = b.create<scf::WhileOp>(
        /*resultTypes=*/
        TypeRange{b.getIndexType()},
        /*operands=*/ValueRange{initIndex.getResult()},
        /*beforeBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value index = args[0];
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          auto coeff = b.create<tensor::ExtractOp>(coeffs, ValueRange{index});
          auto normalizedCoeff = b.create<mod_arith::ReduceOp>(coeff);
          auto extractedCoeff = b.create<mod_arith::ExtractOp>(
              typeInfo.coefficientStorageType, normalizedCoeff);
          auto cmpOp = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                               extractedCoeff, c0);
          b.create<scf::ConditionOp>(cmpOp.getResult(), index);
        },
        /*afterBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          Value currentIndex = args[0];
          auto nextIndex =
              b.create<arith::SubIOp>(currentIndex, c1.getResult());
          b.create<scf::YieldOp>(nextIndex.getResult());
        });
    auto degree = degreeOp.getResult(0);
    auto leadingCoefficient =
        b.create<tensor::ExtractOp>(coeffs, ValueRange{degree});
    rewriter.replaceOp(op, ValueRange{degree, leadingCoefficient.getResult()});
    return success();
  }
};

template <typename SourceOp, typename TargetArithOp, typename TargetModArithOp>
struct ConvertPolyBinop : public OpConversionPattern<SourceOp> {
  ConvertPolyBinop(mlir::MLIRContext *context)
      : OpConversionPattern<SourceOp>(context) {}

  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, this->typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    return llvm::TypeSwitch<Type, LogicalResult>(typeInfo.coefficientType)
        .template Case<IntegerType>([&](auto intTy) {
          auto result =
              b.create<TargetArithOp>(adaptor.getLhs(), adaptor.getRhs());
          rewriter.replaceOp(op, result);
          return success();
        })
        .template Case<ModArithType>([&](ModArithType intTy) {
          auto result =
              b.create<TargetModArithOp>(adaptor.getLhs(), adaptor.getRhs());
          rewriter.replaceOp(op, result);
          return success();
        })
        .Default([&](Type ty) {
          op.emitError("unsupported coefficient type: ") << ty;
          return failure();
        });
  }
};

RankedTensorType polymulOutputTensorType(PolynomialType type) {
  auto convDegree =
      2 * type.getRing().getPolynomialModulus().getPolynomial().getDegree() - 1;
  return RankedTensorType::get({convDegree},
                               type.getRing().getCoefficientType());
}

// Lower polynomial multiplication to a 1D convolution, followed by with a
// modulus reduction in the ring.
struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(const TypeConverter &typeConverter, mlir::MLIRContext *context,
             GetFuncCallbackTy cb)
      : OpConversionPattern<MulOp>(typeConverter, context),
        getFuncOpCallback(cb) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, this->typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();
    auto coeffType = dyn_cast<ModArithType>(typeInfo.coefficientType);
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // Implementing a naive polymul operation which is a loop
    //
    // for i = 0, ..., N-1
    //   for j = 0, ..., N-1
    //     c[i+j] += a[i] * b[j]
    //
    RankedTensorType polymulTensorType =
        polymulOutputTensorType(typeInfo.polynomialType);

    SmallVector<utils::IteratorType> iteratorTypes(
        2, utils::IteratorType::parallel);
    AffineExpr d0, d1;
    bindDims(getContext(), d0, d1);
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(2, 0, {d0}),      // i
        AffineMap::get(2, 0, {d1}),      // j
        AffineMap::get(2, 0, {d0 + d1})  // i+j
    };

    auto intStorageType = coeffType.getModulus().getType();
    auto storageTensorType =
        RankedTensorType::get(polymulTensorType.getShape(), intStorageType);
    auto tensor = b.create<arith::ConstantOp>(DenseElementsAttr::get(
        storageTensorType, b.getIntegerAttr(intStorageType, 0)));
    // The tensor of zeros in which to store the naive polymul output from the
    // linalg.generic op below.
    auto polymulOutput =
        b.create<mod_arith::EncapsulateOp>(polymulTensorType, tensor);

    auto polyMul = b.create<linalg::GenericOp>(
        /*resultTypes=*/polymulTensorType,
        /*inputs=*/adaptor.getOperands(),
        /*outputs=*/polymulOutput.getResult(),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          auto lhs = args[0];
          auto rhs = args[1];
          auto accum = args[2];
          auto mulOp = b.create<mod_arith::MulOp>(lhs, rhs);
          auto addOp = b.create<mod_arith::AddOp>(mulOp, accum);
          b.create<linalg::YieldOp>(addOp.getResult());
        });

    auto postReductionType = convertPolynomialType(typeInfo.polynomialType);
    FunctionType funcType = FunctionType::get(
        op.getContext(), {polymulTensorType}, {postReductionType});

    // 2N - 1 sized result tensor -> reduce modulo ideal to get a N sized tensor
    func::FuncOp divMod = getFuncOpCallback(funcType, typeInfo.ringAttr);
    if (!divMod) {
      return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
        diag << "Missing software implementation for polynomial mod op of type"
             << funcType << " and for ring " << typeInfo.ringAttr;
      });
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, divMod, polyMul.getResult(0));
    return success();
  }

 private:
  GetFuncCallbackTy getFuncOpCallback;
};

struct PolynomialToModArith
    : impl::PolynomialToModArithBase<PolynomialToModArith> {
  using PolynomialToModArithBase::PolynomialToModArithBase;

  void runOnOperation() override;

 private:
  // Generate implementations for operations
  void generateOpImplementations();

  func::FuncOp buildPolynomialModFunc(FunctionType funcType, RingAttr ringAttr);

  // A map containing modular reduction function implementations, generated once
  // at the beginning of this pass based on the ops to be converted, intended to
  // be retrieved by ConvertMul to construct CallOps so that later optimization
  // passes can determine when to inline the implementation.
  DenseMap<std::pair<Type, RingAttr>, func::FuncOp> modImpls;
};

void PolynomialToModArith::generateOpImplementations() {
  ModuleOp module = getOperation();
  module.walk([&](MulOp op) {
    auto polyTy = dyn_cast<PolynomialType>(op.getResult().getType());
    if (!polyTy) {
      op.emitError()
          << "Encountered elementwise polynomial.mul op. The caller must use "
             "convert-elementwise-to-affine pass before lowering polynomial.";
      return WalkResult::interrupt();
    }
    auto convType = polymulOutputTensorType(polyTy);
    auto postReductionType = convertPolynomialType(polyTy);
    FunctionType funcType =
        FunctionType::get(op.getContext(), {convType}, {postReductionType});

    // Generate the software implementation of modular reduction if it has not
    // been generated yet.
    auto key = std::pair(funcType, polyTy.getRing());
    if (!modImpls.count(key)) {
      func::FuncOp modOp = buildPolynomialModFunc(funcType, polyTy.getRing());
      modImpls.insert(std::pair(key, modOp));
    }
    return WalkResult::advance();
  });
}

// Create a software implementation that reduces a polynomial
// modulo a statically known divisor polynomial.
func::FuncOp PolynomialToModArith::buildPolynomialModFunc(FunctionType funcType,
                                                          RingAttr ring) {
  ModuleOp module = getOperation();
  Location loc = module->getLoc();
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  // These tensor types are used in the implementation
  //
  //  - The input tensor<2047x!coeff_ty>, e.g., the output of a naive polymul
  //    of two tensor<1024x!coeff_ty>
  //  - The result tensor<1024x!coeff_ty>, e.g., the result after modular
  //    reduction is complete and represents the remainder being accumulated
  RankedTensorType inputType =
      llvm::cast<RankedTensorType>(funcType.getInput(0));
  RankedTensorType resultType =
      llvm::cast<RankedTensorType>(funcType.getResult(0));

  // TODO(#202): this function name probably also needs the input tensor type in
  // the name, or it could conflict with other implementations that have the
  // same cmod+ideal.
  auto coeffTy = ring.getCoefficientType();
  std::string coeffTyId;
  if (auto intTy = dyn_cast<IntegerType>(coeffTy)) {
    coeffTyId = llvm::formatv("i{0}", intTy.getWidth());
  } else if (auto modTy = dyn_cast<ModArithType>(coeffTy)) {
    IntegerType intTy = cast<IntegerType>(modTy.getModulus().getType());
    SmallString<10> modulusStr;
    modTy.getModulus().getValue().toStringUnsigned(modulusStr);
    coeffTyId =
        llvm::formatv("{0}_i{1}", modulusStr, intTy.getIntOrFloatBitWidth());
  }
  std::string funcName =
      llvm::formatv("__heir_poly_mod_{0}_{1}", coeffTyId,
                    ring.getPolynomialModulus().getPolynomial().toIdentifier());

  auto funcOp = builder.create<func::FuncOp>(funcName, funcType);
  LLVM::linkage::Linkage inlineLinkage = LLVM::linkage::Linkage::LinkonceODR;
  Attribute linkage =
      LLVM::LinkageAttr::get(builder.getContext(), inlineLinkage);
  funcOp->setAttr("llvm.linkage", linkage);
  funcOp.setPrivate();

  Block *funcBody = funcOp.addEntryBlock();
  Value coeffsArg = funcOp.getArgument(0);

  builder.setInsertionPointToStart(funcBody);

  // Implementing the textbook division algorithm
  //
  // def divmod(poly, divisor):
  //   divisorLC = divisor.leadingCoefficient()
  //   divisorDeg = divisor.degree()
  //   remainder = poly
  //   quotient = Zero()
  //
  //   while remainder.degree() >= divisorDeg:
  //      monomialExponent = remainder.degree() - divisorDeg
  //      monomialDivisor = monomial(
  //        monomialExponent,
  //        remainder.leadingCoefficient() / divisorLC
  //      )
  //      quotient += monomialDivisor
  //      remainder -= monomialDivisor * divisor
  //
  //   return quotient, remainder
  //

  // Implementing the algorithm using poly ops, but the function signature
  // input is in terms of the lowered tensor types, so we need a from_tensor.
  // We also need to pick an appropriate ring, which in our case will be the
  // ring of polynomials mod (x^n - 1).
  std::vector<IntMonomial> monomials;
  // If the input has size N as a tensor, then as a polynomial its max degree is
  // N-1, and we want the ring to be mod (x^N - 1).
  unsigned remRingDegree = inputType.getShape()[0];
  monomials.emplace_back(1, remRingDegree);
  monomials.emplace_back(-1, 0);
  IntPolynomial xnMinusOne = IntPolynomial::fromMonomials(monomials).value();
  IntPolynomialAttr xnMinusOneAttr =
      IntPolynomialAttr::get(&getContext(), xnMinusOne);
  auto remRing = RingAttr::get(coeffTy, xnMinusOneAttr);
  auto remRingPolynomialType = PolynomialType::get(&getContext(), remRing);

  // Start by converting the input tensor back to a poly.
  auto fromTensorOp =
      builder.create<FromTensorOp>(remRingPolynomialType, coeffsArg);

  // If the leading coefficient of the divisor has no inverse, we can't do
  // division. The lowering must fail:
  auto divisor = ring.getPolynomialModulus().getPolynomial();
  APInt rawCoeffMod =
      cast<ModArithType>(ring.getCoefficientType()).getModulus().getValue();
  auto [leadingCoef, coeffMod] = extendWidthsToLargest(
      divisor.getTerms().back().getCoefficient(), rawCoeffMod);
  auto leadingCoefInverse = multiplicativeInverse(leadingCoef, coeffMod);
  // returning zero means no inverse was found
  if (leadingCoefInverse.isZero()) {
    signalPassFailure();
  }
  auto divisorLcInverse = getConstantCoefficient(
      coeffTy, leadingCoefInverse.getSExtValue(), builder);
  auto divisorDeg = builder.create<arith::ConstantOp>(
      builder.getIndexType(), builder.getIndexAttr(divisor.getDegree()));

  // while remainder.degree() >= divisorDeg:
  auto remainder = fromTensorOp.getResult();
  auto whileOp = builder.create<scf::WhileOp>(
      /*resultTypes=*/
      remainder.getType(),
      /*operands=*/remainder,
      /*beforeBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value remainder = args[0];
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        // remainder.degree() >= divisorDeg
        auto remainderLt = b.create<LeadingTermOp>(
            b.getIndexType(), inputType.getElementType(), remainder);
        auto cmpOp = b.create<arith::CmpIOp>(
            arith::CmpIPredicate::sge, remainderLt.getDegree(), divisorDeg);
        b.create<scf::ConditionOp>(cmpOp, remainder);
      },
      /*afterBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value remainder = args[0];
        // TODO(#97): move this out of the loop when it has ConstantLike trait
        auto divisorOp = builder.create<ConstantOp>(
            remRingPolynomialType,
            TypedIntPolynomialAttr::get(remRingPolynomialType, divisor));

        // monomialExponent = remainder.degree() - divisorDeg
        auto ltOp = b.create<LeadingTermOp>(
            b.getIndexType(), inputType.getElementType(), remainder);
        auto monomialExponentOp =
            b.create<arith::SubIOp>(ltOp.getDegree(), divisorDeg);

        // monomialDivisor = monomial(
        //   monomialExponent, remainder.leadingCoefficient() / divisorLC)
        auto monomialLc =
            b.create<mod_arith::MulOp>(ltOp.getCoefficient(), divisorLcInverse);

        // remainder -= monomialDivisor * divisor
        auto scaledDivisor = b.create<MulScalarOp>(divisorOp, monomialLc);
        auto remainderIncrement = b.create<MonicMonomialMulOp>(
            scaledDivisor, monomialExponentOp.getResult());
        auto nextRemainder = b.create<SubOp>(remainder, remainderIncrement);

        b.create<scf::YieldOp>(nextRemainder.getResult());
      });

  // The result remainder is still in the larger ring, so we need to convert to
  // the smaller ring.
  auto toTensorOp = builder.create<ToTensorOp>(inputType, whileOp.getResult(0));

  // Smaller ring has a coefficient modulus that needs to be accounted for.
  // Probably a better way to define a splatted dense elements attr, but either
  // way this should be folded/canonicalized into a single op.
  SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes{
      builder.getIndexAttr(resultType.getShape()[0])};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
  auto extractedTensor = builder.create<tensor::ExtractSliceOp>(
      resultType, toTensorOp.getResult(), offsets, sizes, strides);

  builder.create<func::ReturnOp>(extractedTensor.getResult());
  return funcOp;
}

// Multiply two integers x, y modulo cmod.
static APInt mulMod(const APInt &_x, const APInt &_y, const APInt &_cmod) {
  assert(_x.getBitWidth() == _y.getBitWidth() &&
         "expected same bitwidth of operands");
  auto intermediateBitwidth = _cmod.getBitWidth() * 2;
  APInt x = _x.zext(intermediateBitwidth);
  APInt y = _y.zext(intermediateBitwidth);
  APInt cmod = _cmod.zext(intermediateBitwidth);
  APInt res = (x * y).urem(cmod);
  return res.trunc(_x.getBitWidth());
}

// Compute the first degree powers of root modulo cmod.
static SmallVector<APInt> precomputeRoots(APInt root, const APInt &cmod,
                                          unsigned degree) {
  APInt baseRoot = root;
  root = 1;
  SmallVector<APInt> vals(degree);
  for (unsigned i = 0; i < degree; i++) {
    vals[i] = root;
    root = mulMod(root, baseRoot, cmod);
  }
  return vals;
}

static Value computeReverseBitOrder(ImplicitLocOpBuilder &b,
                                    RankedTensorType tensorType, Type modType,
                                    Value tensor) {
  unsigned degree = tensorType.getShape()[0];
  double degreeLog = std::log2((double)degree);
  assert(std::floor(degreeLog) == degreeLog &&
         "expected the degree to be a power of 2");

  unsigned indexBitWidth = (unsigned)degreeLog;
  auto indicesType = RankedTensorType::get(tensorType.getShape(),
                                           IndexType::get(b.getContext()));

  SmallVector<APInt> _indices(degree);
  for (unsigned index = 0; index < degree; index++) {
    _indices[index] = APInt(indexBitWidth, index).reverseBits();
  }
  auto indices = b.create<arith::ConstantOp>(
      indicesType, DenseElementsAttr::get(indicesType, _indices));

  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  AffineExpr d0;
  bindDims(b.getContext(), d0);
  SmallVector<AffineMap> indexingMaps = {AffineMap::get(1, 0, {d0}),
                                         AffineMap::get(1, 0, {d0})};
  auto out = b.create<arith::ConstantOp>(tensorType,
                                         DenseElementsAttr::get(tensorType, 0));
  auto modOut = b.create<mod_arith::EncapsulateOp>(modType, out);
  auto shuffleOp = b.create<linalg::GenericOp>(
      /*resultTypes=*/TypeRange{modType},
      /*inputs=*/ValueRange{indices.getResult()},
      /*outputs=*/ValueRange{modOut.getResult()},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        auto idx = args[0];
        auto elem = b.create<tensor::ExtractOp>(tensor, ValueRange{idx});
        b.create<linalg::YieldOp>(elem.getResult());
      });
  return shuffleOp.getResult(0);
}

static std::pair<Value, Value> bflyCT(ImplicitLocOpBuilder &b, Value A, Value B,
                                      Value root) {
  auto rootB = b.create<mod_arith::MulOp>(B, root);
  auto ctPlus = b.create<mod_arith::AddOp>(A, rootB);
  auto ctMinus = b.create<mod_arith::SubOp>(A, rootB);
  return {ctPlus, ctMinus};
}

static std::pair<Value, Value> bflyGS(ImplicitLocOpBuilder &b, Value A, Value B,
                                      Value root) {
  auto gsPlus = b.create<mod_arith::AddOp>(A, B);
  auto gsMinus = b.create<mod_arith::SubOp>(A, B);
  auto gsMinusRoot = b.create<mod_arith::MulOp>(gsMinus, root);
  return {gsPlus, gsMinusRoot};
}

template <bool inverse>
static Value fastNTT(ImplicitLocOpBuilder &b, RingAttr ring,
                     PrimitiveRootAttr rootAttr, RankedTensorType tensorType,
                     Type modType, Value input) {
  // Compute the number of stages required to compute the NTT
  auto degree = tensorType.getShape()[0];
  unsigned stages = (unsigned)std::log2((double)degree);

  // Precompute the roots
  auto modArithType = cast<ModArithType>(ring.getCoefficientType());
  APInt cmod = modArithType.getModulus().getValue();
  APInt root = rootAttr.getValue().getValue();
  root = !inverse ? root
                  : multiplicativeInverse(root.zext(cmod.getBitWidth()), cmod)
                        .trunc(root.getBitWidth());
  // Initialize the mod_arith roots constant
  auto rootsType = tensorType.clone({degree});
  Value roots = b.create<arith::ConstantOp>(
      rootsType,
      DenseElementsAttr::get(rootsType, precomputeRoots(root, cmod, degree)));
  roots = b.create<mod_arith::EncapsulateOp>(modType, roots);

  // Here is a slightly modified implementation of the standard iterative NTT
  // computation using Cooley-Turkey/Gentleman-Sande butterfly. For reader
  // reference: https://doi.org/10.1007/978-3-031-46077-7_22, and,
  // https://doi.org/10.1109/ACCESS.2023.3294446
  //
  // We modify the standard implementation by pre-computing the root
  // exponential values during compilation instead of doing so at runtime.
  //
  // Let roots be a tensor of <n x ix> where roots[i] = \psi^i, n be the
  // degree of the polynomial and inverse denote the direction. Then we
  // implement the following:
  //
  // def fastNTT(coeffs, n, cmod, roots, inverse):
  //  m = inverse ? n : 2             # m denotes the batchSize or stride
  //  r = inverse ? 1 : degree / 2    # r denotes the exponent of the root
  //  for (s = 0; s < log2(n); s++):
  //    for (k = 0; k < n / m; k++):
  //      for (j = 0; j < m / 2; j++):
  //        A = coeffs[k * m + j]
  //        B = coeffs[k * m + j + m / 2]
  //        root = roots[(2 * j + 1) * rootExp]
  //        coeffs[k * m + j], coeffs[k * m + j + m / 2]
  //          = bflyOp(A, B, root, cmod)
  //      end
  //    end
  //    m = inverse ? m / 2 : m * 2
  //    r = inverse ? r * 2 : m / 2
  //  end
  //
  //  where bflyOp is one of:
  //    bflyCT(A, B, root, cmod):
  //      (A + root * B % cmod, A - root * B % cmod)
  //
  //    bflyGS(A, B, root, cmod):
  //      (A + B % cmod, (A - B) * root % cmod)

  // Initialize the variables
  Value initialValue = b.create<mod_arith::ReduceOp>(input);
  Value initialBatchSize =
      b.create<arith::ConstantIndexOp>(inverse ? degree : 2);
  Value initialRootExp =
      b.create<arith::ConstantIndexOp>(inverse ? 1 : degree / 2);
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value two = b.create<arith::ConstantIndexOp>(2);
  Value n = b.create<arith::ConstantIndexOp>(degree);

  // Define index affine mappings
  AffineExpr x, y;
  bindDims(b.getContext(), x, y);

  auto stagesLoop = b.create<affine::AffineForOp>(
      /*lowerBound=*/0, /* upperBound=*/stages, /*step=*/1,
      /*iterArgs=*/ValueRange{initialValue, initialBatchSize, initialRootExp},
      /*bodyBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value index,
          ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value batchSize = args[1];
        Value rootExp = args[2];

        auto innerLoop = b.create<affine::AffineForOp>(
            /*lbOperands=*/zero, /*lbMap=*/AffineMap::get(1, 0, x),
            /*ubOperands=*/ValueRange{n, batchSize},
            /*ubMap=*/AffineMap::get(2, 0, x.floorDiv(y)),
            /*step=*/1, /*iterArgs=*/args[0],
            /*bodyBuilder=*/
            [&](OpBuilder &nestedBuilder, Location nestedLoc, Value index,
                ValueRange args) {
              ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
              Value indexK = b.create<affine::AffineApplyOp>(
                  x * y, ValueRange{batchSize, index});

              auto arithLoop = b.create<affine::AffineForOp>(
                  /*lbOperands=*/zero, /*lbMap=*/AffineMap::get(1, 0, x),
                  /*ubOperands=*/batchSize,
                  /*ubMap=*/AffineMap::get(1, 0, x.floorDiv(2)),
                  /*step=*/1, /*iterArgs=*/args[0],
                  /*bodyBuilder=*/
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      Value indexJ, ValueRange args) {
                    ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);

                    Value target = args[0];

                    // Get A
                    Value indexA = b.create<affine::AffineApplyOp>(
                        x + y, ValueRange{indexJ, indexK});
                    Value A = b.create<tensor::ExtractOp>(target, indexA);

                    // Get B
                    Value indexB = b.create<affine::AffineApplyOp>(
                        x + y.floorDiv(2), ValueRange{indexA, batchSize});
                    Value B = b.create<tensor::ExtractOp>(target, indexB);

                    // Get root
                    Value rootIndex = b.create<affine::AffineApplyOp>(
                        (2 * x + 1) * y, ValueRange{indexJ, rootExp});
                    Value root = b.create<tensor::ExtractOp>(roots, rootIndex);

                    auto bflyResult =
                        inverse ? bflyGS(b, A, B, root) : bflyCT(b, A, B, root);

                    // Store updated values into accumulator
                    auto insertPlus = b.create<tensor::InsertOp>(
                        bflyResult.first, target, indexA);
                    auto insertMinus = b.create<tensor::InsertOp>(
                        bflyResult.second, insertPlus, indexB);

                    b.create<affine::AffineYieldOp>(insertMinus.getResult());
                  });

              b.create<affine::AffineYieldOp>(arithLoop.getResult(0));
            });

        batchSize = inverse
                        ? b.create<arith::DivUIOp>(batchSize, two).getResult()
                        : b.create<arith::MulIOp>(batchSize, two).getResult();

        rootExp = inverse ? b.create<arith::MulIOp>(rootExp, two).getResult()
                          : b.create<arith::DivUIOp>(rootExp, two).getResult();

        b.create<affine::AffineYieldOp>(
            ValueRange{innerLoop.getResult(0), batchSize, rootExp});
      });

  Value result = stagesLoop.getResult(0);
  if (inverse) {
    APInt degreeInv =
        multiplicativeInverse(APInt(cmod.getBitWidth(), degree), cmod)
            .trunc(root.getBitWidth());
    Value nInv = b.create<arith::ConstantOp>(
        rootsType, DenseElementsAttr::get(rootsType, degreeInv));
    nInv = b.create<mod_arith::EncapsulateOp>(modType, nInv);
    result = b.create<mod_arith::MulOp>(result, nInv);
  }

  return result;
}

struct ConvertNTT : public OpConversionPattern<NTTOp> {
  ConvertNTT(mlir::MLIRContext *context)
      : OpConversionPattern<NTTOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      NTTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto polyTy = dyn_cast<PolynomialType>(op.getInput().getType());
    if (!polyTy) {
      op.emitError(
          "Can't directly lower for a tensor of polynomials. "
          "First run --convert-elementwise-to-affine.");
      return failure();
    }

    if (!op.getRoot()) {
      op.emitError("missing root attribute");
      return failure();
    }

    RingAttr ring = polyTy.getRing();
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto coeffType =
        dyn_cast<ModArithType>(polyTy.getRing().getCoefficientType());
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }
    auto coeffStorageType = coeffType.getModulus().getType();
    auto intTensorType =
        RankedTensorType::get(inputType.getShape(), coeffStorageType);
    auto modType = adaptor.getInput().getType();

    // Compute the ntt and extract the values
    Value nttResult = fastNTT<false>(
        b, ring, op.getRoot().value(), intTensorType, modType,
        computeReverseBitOrder(b, intTensorType, modType, adaptor.getInput()));

    // Insert the ring encoding here to the input type
    auto outputType =
        RankedTensorType::get(inputType.getShape(), coeffType, ring);
    auto intResult = b.create<tensor::CastOp>(outputType, nttResult);
    rewriter.replaceOp(op, intResult);

    return success();
  }
};

struct ConvertINTT : public OpConversionPattern<INTTOp> {
  ConvertINTT(mlir::MLIRContext *context)
      : OpConversionPattern<INTTOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      INTTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    if (!op.getRoot()) {
      op.emitError("missing root attribute");
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto coeffType = dyn_cast<ModArithType>(typeInfo.coefficientType);
    if (!coeffType) {
      op.emitError("expected coefficient type to be mod_arith type");
      return failure();
    }
    auto coeffStorageType = coeffType.getModulus().getType();
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto intTensorType =
        RankedTensorType::get(inputType.getShape(), coeffStorageType);
    auto modType = typeConverter->convertType(op.getOutput().getType());

    // Remove the encoded ring from input tensor type and convert to mod_arith
    // type
    auto input = b.create<tensor::CastOp>(modType, adaptor.getInput());
    auto nttResult = fastNTT<true>(b, typeInfo.ringAttr, op.getRoot().value(),
                                   intTensorType, modType, input);

    auto reversedBitOrder =
        computeReverseBitOrder(b, intTensorType, modType, nttResult);
    rewriter.replaceOp(op, reversedBitOrder);

    return success();
  }
};

void PolynomialToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  // generateOpImplementations must be called before the conversion begins to
  // apply rewrite patterns, because adding function implementations makes
  // changes at the module level, while the conversion patterns are supposed to
  // be local to the op being converted. This design is borrowed from the MLIR
  // --math-to-funcs pass implementation.
  generateOpImplementations();
  auto getDivmodOp = [&](FunctionType funcType, RingAttr ring) -> func::FuncOp {
    auto it = modImpls.find(std::pair(funcType, ring));
    if (it == modImpls.end()) return nullptr;
    return it->second;
  };

  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  PolynomialToModArithTypeConverter typeConverter(context);

  target.addIllegalDialect<PolynomialDialect>();
  RewritePatternSet patterns(context);

  patterns.add<ConvertFromTensor, ConvertToTensor,
               ConvertPolyBinop<AddOp, arith::AddIOp, mod_arith::AddOp>,
               ConvertPolyBinop<SubOp, arith::SubIOp, mod_arith::SubOp>,
               ConvertLeadingTerm, ConvertMonomial, ConvertMonicMonomialMul,
               ConvertConstant, ConvertMulScalar, ConvertNTT, ConvertINTT>(
      typeConverter, context);
  patterns.add<ConvertMul>(typeConverter, patterns.getContext(), getDivmodOp);
  addStructuralConversionPatterns(typeConverter, patterns, target);
  addTensorOfTensorConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
