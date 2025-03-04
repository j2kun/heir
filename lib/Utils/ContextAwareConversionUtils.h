#ifndef LIB_UTILS_CONTEXTAWARECOVNVERSIONUTILS_H_
#define LIB_UTILS_CONTEXTAWARECOVNVERSIONUTILS_H_

#include <numeric>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

FailureOr<Operation *> convertAnyOperand(
    const ContextAwareTypeConverter *typeConverter, Operation *op,
    ArrayRef<Value> operands, ContextAwareConversionPatternRewriter &rewriter);

template <typename T = void>
struct ConvertAny : public ContextAwareConversionPattern {
  ConvertAny(const ContextAwareTypeConverter &anyContextAwareTypeConverter,
             MLIRContext *context)
      : ContextAwareConversionPattern(anyContextAwareTypeConverter,
                                      RewritePattern::MatchAnyOpTypeTag(),
                                      /*benefit=*/1, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    if (!isa<T>(op)) {
      return failure();
    }

    return convertAnyOperand(getTypeConverter(), op, operands, rewriter);
  }
};

template <>
struct ConvertAny<void> : public ContextAwareConversionPattern {
  ConvertAny<void>(
      const ContextAwareTypeConverter &anyContextAwareTypeConverter,
      MLIRContext *context)
      : ContextAwareConversionPattern(anyContextAwareTypeConverter,
                                      RewritePattern::MatchAnyOpTypeTag(),
                                      /*benefit=*/1, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    return convertAnyOperand(getTypeConverter(), op, operands, rewriter);
  }
};

template <typename SourceArithOp, typename TargetModArithOp>
struct ConvertBinOp : public ContextAwareOpConversionPattern<SourceArithOp> {
  ConvertBinOp(mlir::MLIRContext *context)
      : ContextAwareOpConversionPattern<SourceArithOp>(context) {}

  using ContextAwareOpConversionPattern<
      SourceArithOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceArithOp op, typename SourceArithOp::Adaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<TargetModArithOp>(
        adaptor.getLhs().getType(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename T, typename Y = T>
class SecretGenericOpConversion
    : public ContextAwareOpConversionPattern<secret::GenericOp> {
 public:
  using ContextAwareOpConversionPattern<
      secret::GenericOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
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

    // Assemble the arguments for the Scheme operation.
    SmallVector<Value> inputs;
    SmallVector<Type> oldInputTypes;
    for (OpOperand &operand : innerOp.getOpOperands()) {
      if (auto *secretArg = op.getOpOperandForBlockArgument(operand.get())) {
        inputs.push_back(
            adaptor.getODSOperands(0)[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand.get());
      }
      oldInputTypes.push_back(inputs.back().getType());
    }

    if (contextAwareContextAwareTypeConverter) {
      // manually do the OpAdaptor's work
      SmallVector<Type> newInputTypes;
      if (failed(contextAwareContextAwareTypeConverter->convertTypes(
              oldInputTypes, inputs, newInputTypes)))
        return failure();
      rewriter.modifyOpInPlace(op, [&] {
        for (const auto &[input, newType] : llvm::zip(inputs, newInputTypes)) {
          input.setType(newType);
        }
      });
    }
    // else OpAdaptor will do it for us

    // convert the result types
    SmallVector<Type> resultTypes;
    if (contextAwareContextAwareTypeConverter) {
      if (failed(contextAwareContextAwareTypeConverter->convertTypes(
              op.getResultTypes(), op.getResults(), resultTypes)))
        return failure();
    } else {
      if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                  resultTypes)))
        return failure();
    }

    // only preserve dialect attrs
    // we do not want op attrs like overflowFlags from arith.add
    SmallVector<NamedAttribute> dialectAttrs;
    for (auto &namedAttr : innerOp.getDialectAttrs()) {
      dialectAttrs.push_back(namedAttr);
    }

    // special treatment for memref.alloc, which has alignment attr
    // that should be preserved
    // TODO: secret-to-cggi should handle this instead of here
    for (auto &namedAttr : innerOp.getAttrs()) {
      if (namedAttr.getName().getValue() == "alignment") {
        dialectAttrs.push_back(namedAttr);
      }
    }

    return matchAndRewriteInner(op, resultTypes, inputs, dialectAttrs,
                                rewriter);
  }

  // Default method for replacing the secret.generic with the target
  // operation.
  virtual LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs, attributes);
    return success();
  }
};

// A converter for a secret.generic that doesn't touch the inner ops
class SecretGenericConversion
    : public ContextAwareOpConversionPattern<secret::GenericOp> {
 public:
  using ContextAwareOpConversionPattern<
      secret::GenericOp>::ContextAwareOpConversionPattern;

  virtual LogicalResult finalizeOpModification(
      secret::GenericOp op,
      ContextAwareConversionPatternRewriter &rewriter) const {
    return success();
  };

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    // convert the result types
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                op.getResults(), resultTypes)))
      return failure();

    // Replace the generic op with a new generic op that has the new result
    // types
    // FIXME: replaceOpWithNewOp using adaptor converted types
    secret::GenericOp newGeneric =
        op.cloneWithNewResultTypes(resultTypes, rewriter);
    rewriter.replaceOp(op, newGeneric);

    if (failed(finalizeOpModification(newGeneric, rewriter))) return failure();

    return success();
  }
};

template <typename T, typename Y>
class SecretGenericOpCipherConversion : public SecretGenericOpConversion<T, Y> {
 public:
  using SecretGenericOpConversion<T, Y>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs);
    newOp->setDialectAttrs(attributes);

    // The context-aware type conversions may look for attributes attached
    // to the ops that generate operands given to a later op. This new op
    // may be the op that produces those operands, so we need to preserve
    // the attributes from the original (secret.generic) op on the newly
    // created op.
    for (auto attribute : op->getAttrs())
      newOp->setAttr(attribute.getName(), attribute.getValue());
    return success();
  }
};

template <typename T, typename Y>
class SecretGenericOpCipherPlainConversion
    : public SecretGenericOpConversion<T, Y> {
 public:
  using SecretGenericOpConversion<T, Y>::SecretGenericOpConversion;

  // Ciphertext-plaintext ops should take precedence over ciphertext-ciphertext
  // ops because the ops being converted (e.g., addi) don't have a plaintext
  // variant.
  SecretGenericOpCipherPlainConversion(
      const ContextAwareTypeConverter &typeConverter, MLIRContext *context)
      : SecretGenericOpConversion<T, Y>(typeConverter, context, /*benefit=*/2) {
  }

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    auto ciphertextValues =
        llvm::to_vector(llvm::make_filter_range(inputs, [&](Value input) {
          return isa<lwe::NewLWECiphertextType>(input.getType());
        }));
    if (ciphertextValues.size() != 1) {
      return failure();
    }
    auto ciphertext =
        dyn_cast<TypedValue<lwe::NewLWECiphertextType>>(ciphertextValues[0]);
    if (!ciphertext) {
      return failure();
    }

    auto cleartextValues =
        llvm::to_vector(llvm::make_filter_range(inputs, [&](Value input) {
          // The cleartext value could be a tensor of values or a scalar.
          return !isa<lwe::NewLWECiphertextType>(input.getType());
        }));
    if (cleartextValues.size() != 1) {
      return failure();
    }
    Value cleartext = cleartextValues[0];
    lwe::NewLWECiphertextType ciphertextTy = ciphertext.getType();
    auto plaintextTy = lwe::NewLWEPlaintextType::get(
        op.getContext(), ciphertextTy.getApplicationData(),
        ciphertextTy.getPlaintextSpace());
    auto plaintext = rewriter.create<lwe::RLWEEncodeOp>(
        op.getLoc(), plaintextTy, cleartext,
        ciphertextTy.getPlaintextSpace().getEncoding(),
        ciphertextTy.getPlaintextSpace().getRing());

    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, ciphertext, plaintext);
    newOp->setDialectAttrs(attributes);
    for (auto attribute : op->getAttrs())
      newOp->setAttr(attribute.getName(), attribute.getValue());
    return success();
  }
};

template <typename T>
class SecretGenericOpRelinearizeConversion
    : public SecretGenericOpConversion<mgmt::RelinearizeOp, T> {
 public:
  using SecretGenericOpConversion<mgmt::RelinearizeOp,
                                  T>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    auto inputDimension = cast<lwe::NewLWECiphertextType>(inputs[0].getType())
                              .getCiphertextSpace()
                              .getSize();
    SmallVector<int32_t> fromBasis;
    for (int i = 0; i < inputDimension; ++i) {
      fromBasis.push_back(i);
    }
    SmallVector<int32_t> toBasis = {0, 1};

    auto newOp = rewriter.replaceOpWithNewOp<T>(
        op, inputs[0], rewriter.getDenseI32ArrayAttr(fromBasis),
        rewriter.getDenseI32ArrayAttr(toBasis));
    newOp->setDialectAttrs(attributes);
    for (auto attribute : op->getAttrs())
      newOp->setAttr(attribute.getName(), attribute.getValue());
    return success();
  }
};

template <typename M, typename T, typename Y>
class SecretGenericOpMulConversion : public SecretGenericOpConversion<M, T> {
 public:
  using SecretGenericOpConversion<M, T>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    auto plaintextValues =
        llvm::to_vector(llvm::make_filter_range(inputs, [&](Value input) {
          return !isa<lwe::NewLWECiphertextType>(input.getType());
        }));
    if (!plaintextValues.empty()) {
      return failure();
    }

    // only left for CKKS, should be removed later
    auto newOp = rewriter.replaceOpWithNewOp<Y>(
        op, rewriter.create<T>(op.getLoc(), inputs),
        rewriter.getDenseI32ArrayAttr({0, 1, 2}),
        rewriter.getDenseI32ArrayAttr({0, 1}));
    newOp->setDialectAttrs(attributes);
    for (auto attribute : op->getAttrs())
      newOp->setAttr(attribute.getName(), attribute.getValue());
    return success();
  }
};

template <typename T>
class SecretGenericOpRotateConversion
    : public SecretGenericOpConversion<tensor_ext::RotateOp, T> {
 public:
  using SecretGenericOpConversion<tensor_ext::RotateOp,
                                  T>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    // Check that the offset is a constant.
    auto offset = inputs[1];
    auto constantOffset =
        dyn_cast<mlir::arith::ConstantOp>(offset.getDefiningOp());
    if (!constantOffset) {
      op.emitError("expected constant offset for rotate");
    }
    auto offsetAttr = llvm::dyn_cast<IntegerAttr>(constantOffset.getValue());
    auto newOp =
        rewriter.replaceOpWithNewOp<T>(op, outputTypes, inputs[0], offsetAttr);
    newOp->setDialectAttrs(attributes);
    for (auto attribute : op->getAttrs())
      newOp->setAttr(attribute.getName(), attribute.getValue());
    return success();
  }
};

template <typename Y>
class SecretGenericOpModulusSwitchConversion
    : public SecretGenericOpConversion<mgmt::ModReduceOp, Y> {
 public:
  using SecretGenericOpConversion<mgmt::ModReduceOp,
                                  Y>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    auto outputType = outputTypes[0];
    auto outputElementType = getElementTypeOrSelf(outputType);
    auto outputRing = cast<lwe::NewLWECiphertextType>(outputElementType)
                          .getCiphertextSpace()
                          .getRing();

    // secret-to-ckks allow tensor of ciphertext
    if (auto outputTensorType = dyn_cast<RankedTensorType>(outputType)) {
      // need tensor::extract/tensor::insert
      // manually fully unroll it
      auto shape = outputTensorType.getShape();
      auto totalSize = std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<int64_t>());
      auto emptyOp = rewriter.create<tensor::EmptyOp>(op.getLoc(), shape,
                                                      outputElementType);
      Operation *resultOp = emptyOp;
      for (int i = 0; i < totalSize; ++i) {
        SmallVector<int64_t> indices;
        auto iCopy = i;
        for (int64_t j : shape) {
          indices.push_back(iCopy % j);
          iCopy /= j;
        }
        SmallVector<Value> constants;
        for (int64_t index : indices) {
          constants.push_back(rewriter.create<mlir::arith::ConstantOp>(
              op.getLoc(), rewriter.getIndexAttr(index)));
        }
        auto extract = rewriter.create<tensor::ExtractOp>(op.getLoc(),
                                                          inputs[0], constants);
        auto modulusSwitchOp = rewriter.create<Y>(
            op.getLoc(), outputElementType, extract.getResult(), outputRing);
        modulusSwitchOp->setDialectAttrs(attributes);
        auto insert = rewriter.create<tensor::InsertOp>(
            op.getLoc(), modulusSwitchOp.getResult(), resultOp->getResult(0),
            constants);
        resultOp = insert;
      }
      rewriter.replaceOp(op, resultOp);
      for (auto attribute : op->getAttrs())
        resultOp->setAttr(attribute.getName(), attribute.getValue());
      return success();
    }

    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, outputTypes[0], inputs[0],
                                                outputRing);
    newOp->setDialectAttrs(attributes);
    for (auto attribute : op->getAttrs())
      newOp->setAttr(attribute.getName(), attribute.getValue());
    return success();
  }
};

}  // namespace heir
#endif  // LIB_UTILS_CONTEXTAWARECOVNVERSIONUTILS_H_
