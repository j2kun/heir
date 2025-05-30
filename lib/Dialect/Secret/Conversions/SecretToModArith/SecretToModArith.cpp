#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETTOMODARITH
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h.inc"

bool isModArithOrContainerOfModArith(Type type) {
  return isa<mod_arith::ModArithType>(getElementTypeOrSelf(type));
}

class SecretToModArithTypeConverter : public TypeConverter {
 public:
  SecretToModArithTypeConverter(MLIRContext *ctx, int64_t ptm)
      : plaintextModulus(ptm) {
    addConversion([](Type type) { return type; });
    addConversion(
        [this](secret::SecretType type) { return convertSecretType(type); });
  }

  Type convertPlaintextType(Type type) {
    auto *ctx = type.getContext();
    return TypeSwitch<Type, Type>(type)
        .Case<IntegerType>([this, ctx](IntegerType intType) {
          Type newType;
          if (plaintextModulus == 0) {
            auto modulusBitSize = (int64_t)intType.getIntOrFloatBitWidth();
            plaintextModulus = (1L << (modulusBitSize - 1L));
            newType = mlir::IntegerType::get(intType.getContext(),
                                             modulusBitSize + 1);
          } else {
            newType = mlir::IntegerType::get(ctx, 64);
          }

          return mod_arith::ModArithType::get(
              ctx, mlir::IntegerAttr::get(newType, plaintextModulus));
        })
        .Case<ShapedType>([this](ShapedType shapedType) {
          if (auto arithType =
                  llvm::dyn_cast<IntegerType>(shapedType.getElementType())) {
            return shapedType.cloneWith(shapedType.getShape(),
                                        convertPlaintextType(arithType));
          }
          assert(false &&
                 "non-integer element type for tensor in plaintext type "
                 "conversion");
          return shapedType;
        })
        .Default([](Type t) { return t; });
  }

  Type convertSecretType(secret::SecretType type) {
    return convertPlaintextType(type.getValueType());
  }

 private:
  int64_t plaintextModulus;
};

template <typename T, typename Y = T>
class SecretGenericOpConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  SecretGenericOpConversion(const TypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<secret::GenericOp>(typeConverter, context,
                                               benefit) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (op.getBody()->getOperations().size() > 2) {
      // Each secret.generic should contain at most one instruction -
      // secret-distribute-generic can be used to distribute through the
      // arithmetic ops.
      return failure();
    }

    auto &innerOp = op.getBody()->getOperations().front();
    if (!isa<T>(innerOp)) {
      return failure();
    }

    // The inner op's arguments are either plaintext operands, in which case
    // they are already type-converted, or else they are ciphertext operands,
    // in which case we can get them in type-converted form from the adaptor.
    SmallVector<Value> inputs;
    for (Value operand : innerOp.getOperands()) {
      if (auto *secretArg = op.getOpOperandForBlockArgument(operand)) {
        inputs.push_back(adaptor.getInputs()[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand);
      }
    }

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    FailureOr<Operation *> newOpResult =
        matchAndRewriteInner(op, resultTypes, inputs, rewriter);
    if (failed(newOpResult)) return failure();
    return success();
  }

  // Default method for replacing the secret.generic with the target
  // operation.
  virtual FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter &rewriter) const {
    return rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs)
        .getOperation();
  }
};

// This is similar to ConversionUtils::convertAnyOperand, but it requires the
// cloning to occur on the op inside the secret generic, while using
// type-converted operands and results of the outer generic op.
class ConvertAnyNestedGeneric : public OpConversionPattern<secret::GenericOp> {
 public:
  ConvertAnyNestedGeneric(const TypeConverter &typeConverter,
                          MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<secret::GenericOp>(typeConverter, context,
                                               benefit) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp outerOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (outerOp.getBody()->getOperations().size() > 2) {
      return failure();
    }
    Operation *innerOp = &outerOp.getBody()->getOperations().front();

    SmallVector<Value> inputs;
    for (Value operand : innerOp->getOperands()) {
      if (auto *secretArg = outerOp.getOpOperandForBlockArgument(operand)) {
        inputs.push_back(adaptor.getInputs()[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand);
      }
    }

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(outerOp.getResultTypes(),
                                                resultTypes)))
      return failure();

    SmallVector<std::unique_ptr<Region>, 1> regions;
    IRMapping mapping;
    for (auto &r : innerOp->getRegions()) {
      Region *newRegion = new Region(innerOp);
      rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
      if (failed(rewriter.convertRegionTypes(newRegion, *typeConverter)))
        return failure();
      regions.emplace_back(newRegion);
    }

    Operation *newOp = rewriter.create(OperationState(
        outerOp.getLoc(), innerOp->getName().getStringRef(), inputs,
        resultTypes, innerOp->getAttrs(), innerOp->getSuccessors(), regions));
    rewriter.replaceOp(outerOp, newOp);
    return success();
  }
};

// "encode" a cleartext to mod_arith by sign extending and encapsulating it.
Value encodeCleartext(Value cleartext, Type resultType,
                      ImplicitLocOpBuilder &b) {
  // We start with something like an i16 (or tensor<Nxi16>) and the result
  // should be a (tensor of) !mod_arith.int<17 : i64> so we need to first
  // sign extend the input to the mod_arith storage type, then encapsulate it
  // into the mod_arith type.
  mod_arith::ModArithType resultEltTy =
      cast<mod_arith::ModArithType>(getElementTypeOrSelf(resultType));
  IntegerType modulusType =
      cast<IntegerType>(resultEltTy.getModulus().getType());
  Type extendedType = modulusType;

  if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
    extendedType = shapedType.cloneWith(shapedType.getShape(), modulusType);
  }

  auto extOp = b.create<arith::ExtSIOp>(extendedType, cleartext);
  auto encapsulateOp =
      b.create<mod_arith::EncapsulateOp>(resultType, extOp.getResult());
  return encapsulateOp.getResult();
}

struct ConvertConceal : public OpConversionPattern<secret::ConcealOp> {
  ConvertConceal(mlir::MLIRContext *context)
      : OpConversionPattern<secret::ConcealOp>(context) {}

  using OpConversionPattern<secret::ConcealOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, secret::ConcealOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // We start with something like an i16 (or tensor<Nxi16>) and the result
    // should be a (tensor of) !mod_arith.int<17 : i64> so we need to first
    // sign extend the input to the mod_arith storage type, then encapsulate it
    // into the mod_arith type.
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value replacementValue =
        encodeCleartext(adaptor.getCleartext(), resultType, b);
    rewriter.replaceOp(op, replacementValue);
    return success();
  }
};

struct ConvertReveal : public OpConversionPattern<secret::RevealOp> {
  ConvertReveal(mlir::MLIRContext *context)
      : OpConversionPattern<secret::RevealOp>(context) {}

  using OpConversionPattern<secret::RevealOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::RevealOp op, secret::RevealOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // We start with something like a secret<i16> (or secret<tensor<Nxi16>>)
    // and type conversion gives us a (tensor of) !mod_arith.int<17 : i64> so
    // we need to first mod_arith extract to get the result in terms of i64,
    // then truncate to the original i16 type.
    Type modArithTypeOrTensor = adaptor.getInput().getType();
    auto eltTy = cast<mod_arith::ModArithType>(
        getElementTypeOrSelf(modArithTypeOrTensor));
    IntegerType modulusType = cast<IntegerType>(eltTy.getModulus().getType());
    Type beforeTrunc = modulusType;
    if (auto shapedType = dyn_cast<ShapedType>(modArithTypeOrTensor)) {
      beforeTrunc = shapedType.cloneWith(shapedType.getShape(), modulusType);
    }
    Type truncatedType = op.getResult().getType();

    auto extractOp =
        b.create<mod_arith::ExtractOp>(beforeTrunc, adaptor.getInput());
    auto truncOp =
        b.create<arith::TruncIOp>(truncatedType, extractOp.getResult());
    rewriter.replaceOp(op, truncOp);
    return success();
  }
};

// This is like
// ContextAwareConversionUtils::SecretGenericOpCipherPlainConversion, except
// that secret types are type converted to mod_arith, while plaintext types
// stay as regular tensor types, and need to be "encoded" (encapsulated) into
// mod_arith tensors, whereas for normal secret-to-scheme, there is a dedicated
// ciphertext-plaintext op.
template <typename T, typename Y>
class SecretGenericOpCipherPlainConversion
    : public SecretGenericOpConversion<T, Y> {
 public:
  using SecretGenericOpConversion<T, Y>::SecretGenericOpConversion;

  // Ciphertext-plaintext ops should take precedence over ciphertext-ciphertext
  // ops because the ops being converted (e.g., addi) don't have a plaintext
  // variant.
  SecretGenericOpCipherPlainConversion(const TypeConverter &typeConverter,
                                       MLIRContext *context)
      : SecretGenericOpConversion<T, Y>(typeConverter, context, /*benefit=*/3) {
  }

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter &rewriter) const override {
    // Verify that exactly one of the two inputs is a ciphertext.
    if (inputs.size() != 2 ||
        llvm::count_if(inputs, [&](Value input) {
          return isModArithOrContainerOfModArith(input.getType());
        }) != 1) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    if (isModArithOrContainerOfModArith(input0.getType())) {
      auto encoded = encodeCleartext(input1, input0.getType(), b);
      auto newOp = rewriter.replaceOpWithNewOp<Y>(op, input0, encoded);
      return newOp.getOperation();
    }

    auto encoded = encodeCleartext(input0, input1.getType(), b);
    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, encoded, input1);
    return newOp.getOperation();
  }
};

struct SecretToModArith : public impl::SecretToModArithBase<SecretToModArith> {
  using SecretToModArithBase::SecretToModArithBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    SecretToModArithTypeConverter typeConverter(context, plaintextModulus);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<mod_arith::ModArithDialect>();
    target.addIllegalDialect<secret::SecretDialect>();

    // These patterns have higher benefit to take precedence over the default
    // pattern, which simply converts operand/result types and inlines the
    // operation inside the generic.
    patterns.add<
        SecretGenericOpCipherPlainConversion<arith::AddIOp, mod_arith::AddOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, mod_arith::SubOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, mod_arith::MulOp> >(
        typeConverter, context,
        /*benefit=*/3);

    patterns.add<SecretGenericOpConversion<arith::AddIOp, mod_arith::AddOp>,
                 SecretGenericOpConversion<arith::SubIOp, mod_arith::SubOp>,
                 SecretGenericOpConversion<arith::MulIOp, mod_arith::MulOp>,
                 ConvertReveal, ConvertConceal>(typeConverter, context,
                                                /*benefit=*/2);

    patterns.add<ConvertAnyNestedGeneric>(typeConverter, context,
                                          /*benefit=*/1);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clear any tensor_ext attributes from the func
    getOperation()->walk([&](FunctionOpInterface funcOp) {
      for (int i = 0; i < funcOp.getNumArguments(); ++i) {
        for (auto attr : funcOp.getArgAttrs(i)) {
          // the attr name is tensor_ext.foo, so just check for the prefix
          if (attr.getName().getValue().starts_with("tensor_ext")) {
            funcOp.removeArgAttr(i, attr.getName());
          }
        }
      }

      for (int i = 0; i < funcOp.getNumResults(); ++i) {
        for (auto attr : funcOp.getResultAttrs(i)) {
          if (attr.getName().getValue().starts_with("tensor_ext")) {
            funcOp.removeResultAttr(i, attr.getName());
          }
        }
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
