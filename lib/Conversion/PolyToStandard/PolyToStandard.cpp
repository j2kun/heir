#include "include/Conversion/PolyToStandard/PolyToStandard.h"

#include "include/Dialect/Poly/IR/PolyOps.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "lib/Conversion/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "include/Conversion/PolyToStandard/PolyToStandard.h.inc"

RankedTensorType convertPolyType(PolyType type) {
  RingAttr attr = type.getRing();
  uint32_t idealDegree = attr.ideal().getDegree();
  // We subtract one because the maximum value of a coefficient is one less
  // than the modulus. When the modulus is an exact power of 2 this matters.
  unsigned eltBitWidth = (attr.coefficientModulus() - 1).getActiveBits();
  IntegerType elementTy =
      IntegerType::get(type.getContext(), eltBitWidth,
                       IntegerType::SignednessSemantics::Signless);
  // We must remove the ring attribute on the tensor, since the
  // unrealized_conversion_casts cannot carry the poly.ring attribute
  // through.
  return RankedTensorType::get({idealDegree}, elementTy);
}

class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PolyType type) -> Type { return convertPolyType(type); });

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
    auto resultTy = typeConverter->convertType(op->getResultTypes()[0]);
    auto resultTensorTy = cast<RankedTensorType>(resultTy);
    auto resultShape = resultTensorTy.getShape()[0];
    auto resultEltTy = resultTensorTy.getElementType();

    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];
    auto inputEltTy = inputTensorTy.getElementType();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getOperands()[0];
    // Extend element type if needed.
    if (inputEltTy != resultEltTy) {
      // FromTensorOp verifies that the coefficient tensor's elements fit into
      // the polynomial.
      assert(inputEltTy.getIntOrFloatBitWidth() <
             resultEltTy.getIntOrFloatBitWidth());

      coeffValue = b.create<arith::ExtUIOp>(
          RankedTensorType::get(inputShape, resultEltTy), coeffValue);
    }

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
      coeffValue = b.create<tensor::PadOp>(
          resultTy, coeffValue, low, high,
          b.create<arith::ConstantOp>(rewriter.getIntegerAttr(resultEltTy, 0)),
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

struct ConvertMonomial : public OpConversionPattern<MonomialOp> {
  ConvertMonomial(mlir::MLIRContext *context)
      : OpConversionPattern<MonomialOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MonomialOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto polyType = cast<PolyType>(op.getResult().getType());
    auto tensorType =
        cast<RankedTensorType>(typeConverter->convertType(polyType));
    auto tensor = b.create<arith::ConstantOp>(DenseElementsAttr::get(
        tensorType, b.getIntegerAttr(tensorType.getElementType(), 0)));
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getCoefficient(),
                                                  tensor, adaptor.getDegree());
    return success();
  }
};

struct ConvertDegree : public OpConversionPattern<DegreeOp> {
  ConvertDegree(mlir::MLIRContext *context)
      : OpConversionPattern<DegreeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // FIXME: Is it worth implementing this as a function call as well?
  LogicalResult matchAndRewrite(
      DegreeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffs = adaptor.getInput();
    auto tensorType = cast<RankedTensorType>(coeffs.getType());
    auto c0 = b.create<arith::ConstantOp>(
        b.getIntegerAttr(tensorType.getElementType(), 0));
    auto c1 = b.create<arith::ConstantOp>(b.getIndexAttr(1));
    auto initIndex = b.create<arith::ConstantOp>(
        b.getIndexAttr(tensorType.getShape()[0] - 1));

    rewriter.replaceOpWithNewOp<scf::WhileOp>(
        op,
        /*resultTypes=*/TypeRange{b.getIndexType()},
        /*operands=*/ValueRange{initIndex.getResult()},
        /*beforeBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value index = args[0];
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          auto coeff = b.create<tensor::ExtractOp>(coeffs, ValueRange{index});
          auto cmpOp =
              b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, coeff, c0);
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
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert add lowers a poly.add operation to arith operations. A poly.add
  // operation is defined within the polynomial ring. Coefficients are added
  // element-wise as elements of the ring, so they are performed modulo the
  // coefficient modulus.
  //
  // This lowering detects when the ring's coefficient modulus is a power of 2,
  // and hence the natural overflow semantics can be relied upon to implement
  // modular arithmetic. In other cases, explicit modular arithmetic operations
  // are inserted, which requires representing the modulus as a constant, and
  // hence may require extending the intermediate arithmetic to higher bit
  // widths.
  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto type = cast<ShapedType>(adaptor.getLhs().getType());

    APInt mod =
        cast<PolyType>(op.getResult().getType()).getRing().coefficientModulus();
    bool needToExtend = !mod.isPowerOf2();

    if (!needToExtend) {
      auto result = b.create<arith::AddIOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }

    // The arithmetic may spill into higher bit width, so start by extending
    // all the types to the smallest bit width that can contain them all.
    unsigned nextHigherBitWidth = (mod - 1).getActiveBits() + 1;
    auto modIntType = rewriter.getIntegerType(nextHigherBitWidth);
    auto modIntTensorType = RankedTensorType::get(type.getShape(), modIntType);

    auto cmod = b.create<arith::ConstantOp>(DenseIntElementsAttr::get(
        modIntTensorType, {mod.trunc(nextHigherBitWidth)}));

    auto signExtensionLhs =
        b.create<arith::ExtUIOp>(modIntTensorType, adaptor.getLhs());
    auto signExtensionRhs =
        b.create<arith::ExtUIOp>(modIntTensorType, adaptor.getRhs());

    auto higherBitwidthAdd =
        b.create<arith::AddIOp>(signExtensionLhs, signExtensionRhs);
    auto remOp = b.create<arith::RemUIOp>(higherBitwidthAdd, cmod);
    auto truncOp = b.create<arith::TruncIOp>(type, remOp);

    rewriter.replaceOp(op, truncOp);

    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(mlir::MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(https://github.com/google/heir/issues/104): implement
    return success();
  }
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto module = getOperation();
    ConversionTarget target(*context);
    PolyToStandardTypeConverter typeConverter(context);

    target.addLegalDialect<arith::ArithDialect>();

    // target.addIllegalDialect<PolyDialect>();
    target
        .addIllegalOp<FromTensorOp, ToTensorOp, AddOp, DegreeOp, MonomialOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertFromTensor, ConvertToTensor, ConvertAdd, ConvertDegree,
                 ConvertMonomial>(typeConverter, context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    // TODO(https://github.com/google/heir/issues/143): Handle tensor of polys.
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace poly
}  // namespace heir
}  // namespace mlir