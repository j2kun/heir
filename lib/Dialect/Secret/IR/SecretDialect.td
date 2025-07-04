#ifndef LIB_DIALECT_SECRET_IR_SECRETDIALECT_TD_
#define LIB_DIALECT_SECRET_IR_SECRETDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Secret_Dialect : Dialect {
  let name = "secret";
  let description = [{
    Secret is a dialect for computations that operate on encrypted data.

    Secret is intended to serve as a scheme-agnostic front-end for the HEIR
    ecosystem of dialects. It is supposed to be fully interoperable with the
    rest of MLIR via secret.generic, while lower-level HEIR dialects would have
    custom types for arithmetic on secret integers of various bit widths.
  }];

  let extraClassDeclaration = [{
    /// Name of the attribute indicate whether an argument of a function is a
    //secret.
    constexpr const static ::llvm::StringLiteral
        kArgSecretAttrName = "secret.secret";
    constexpr const static ::llvm::StringLiteral
        kArgPublicAttrName = "secret.public";
    constexpr const static ::llvm::StringLiteral
        kArgMissingAttrName = "secret.missing";
    constexpr const static ::llvm::StringLiteral
        kArgUnknownAttrName = "secret.unknown";
    constexpr const static ::llvm::StringLiteral
        kSecretInitsAttrName = "secret.secret_inits";
    constexpr const static ::llvm::StringLiteral
        kKernelAttrName = "secret.kernel";

    // Name of the attribute for plaintext backend execution result.
    constexpr const static ::llvm::StringLiteral
        kArgExecutionResultAttrName = "secret.execution_result";
  }];

  let cppNamespace = "::mlir::heir::secret";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_SECRET_IR_SECRETDIALECT_TD_
