#ifndef LIB_DIALECT_CKKS_IR_CKKSENUMS_TD_
#define LIB_DIALECT_CKKS_IR_CKKSENUMS_TD_

include "mlir/IR/EnumAttr.td"

def CKKS_EncryptionTypeEnum : I32EnumAttr<"CKKSEncryptionType", "An enum attribute representing an encryption method", [
  I32EnumAttrCase<"pk", 0>,
  I32EnumAttrCase<"sk", 1>
]> {
    let cppNamespace = "::mlir::heir::ckks";
}

def CKKS_EncryptionTechniqueEnum : I32EnumAttr<"CKKSEncryptionTechnique", "An enum attribute representing an encryption technique", [
  I32EnumAttrCase<"standard", 0>,
  // The extended encryption technique encrypts at Qp then mod reduces to Q.
  // This has the benefit of smaller encryption noise.
  // See https://ia.cr/2022/915
  I32EnumAttrCase<"extended", 1>
]> {
    let cppNamespace = "::mlir::heir::ckks";
}

#endif  // LIB_DIALECT_CKKS_IR_CKKSENUMS_TD_
