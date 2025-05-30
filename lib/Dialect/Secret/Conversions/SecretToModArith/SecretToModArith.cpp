#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Secret/Conversions/Patterns.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOMODARITH
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h.inc"

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
  using OpConversionPattern<secret::GenericOp>::OpConversionPattern;

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
    target.addIllegalDialect<mgmt::MgmtDialect>();

    patterns.add<SecretGenericOpConversion<arith::AddIOp, mod_arith::AddOp>,
                 SecretGenericOpConversion<arith::SubIOp, mod_arith::SubOp>,
                 SecretGenericOpConversion<arith::MulIOp, mod_arith::MulOp>>(
        typeConverter, context);

    // patterns.add<ConvertClientConceal>(typeConverter, context, usePublicKey,
    //                                    rlweRing.value());
    // patterns.add<ConvertClientReveal>(typeConverter, context,
    // rlweRing.value());

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
