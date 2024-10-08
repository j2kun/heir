#ifndef LIB_DIALECT_LWE_IR_NEWLWEATTRIBUTES_TD_
#define LIB_DIALECT_LWE_IR_NEWLWEATTRIBUTES_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Below defines new LWE attributes following
// [#785](https://github.com/google/heir/issues/785).

class LWE_OverflowAttr<string attrName, string attrMnemonic>
  : AttrDef<LWE_Dialect, attrName> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def LWE_NoOverflowAttr : LWE_OverflowAttr<"NoOverflow", "no_overflow"> {
  let summary = "An attribute informing that application data never overflows.";
  let description = [{
    This attribute informs lowerings that a program is written so that the message data
    will never overflow beyond the message type.

    // FIXME: Have a separate WraparoundOverflow, which lowers the same as NoOverflow?
  }];
}

def LWE_PreserveOverflowAttr : LWE_OverflowAttr<"PreserveOverflow", "preserve_overflow"> {
  let summary = "An attribute informing that application data overflows in the message type.";
  let description = [{
    This attribute informs lowerings that a program is written so that the message data
    may overflow beyond the message type.
  }];
}

def LWE_ApplicationDataAttr : AttrDef<LWE_Dialect, "ApplicationData"> {
  let mnemonic = "application_data";
  let description = [{
    An attribute describing the semantics of the underlying application data.

    The `messageType` parameter is used to describe the type and bits of the
    original application data, e.g. i1, i32, f32. This type is later mapped
    into the plaintext space of an FHE scheme by embedding, scaling, or other
    techniques.

    This attribute also contains information about the overflow semantics of the
    data in the application. By default, we assume that the application program
    was written so that the overflow is not expected and the overflow attribute
    can can be `no_overflow`. For LWE-based CGGI ciphertexts, the overflow
    attribute will usually be `preserve_overflow`, since messages will overflow
    into padding bits.
  }];

  let parameters = (ins
    "mlir::Type":$message_type,
    DefaultValuedParameter<"Attribute", "NoOverflowAttr::get($_ctxt)">:$overflow
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$message_type, "Attribute":$overflow), [{
      return $_get(message_type.getContext(), message_type, overflow);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";

  // Verify that the overflow attribute is one of preserve or no overflow.
  let genVerifyDecl = 1;
}

class LWE_EncodingAttrForLWE<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LWE_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

class LWE_EncodingAttrWithScalingParam<string attrName, string attrMnemonic, list<Trait> traits = []>
    : LWE_EncodingAttrForLWE<attrName, attrMnemonic, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    "unsigned":$scaling_factor
  );
}

def LWE_ConstantCoefficientEncoding
  : LWE_EncodingAttrWithScalingParam<"ConstantCoefficientEncoding", "constant_coefficient_encoding"> {
  let summary = "An encoding of a scalar in the constant coefficient";
  let description = [{
    An encoding of a single scalar into the constant coefficient of the plaintext.

    All other coefficients of the plaintext are set to be zero. This encoding is
    used to encode scalar LWE ciphertexts where the plaintext space is viewed
    as a polynomial ring modulo `x`.

    The scalar is first multiplied by the `scaling_factor` and then rounded to
    the nearest integer before encoding into the plaintext coefficient.

    Example:

    ```
    #coeff_encoding = #lwe.constant_coefficient_encoding<scaling_factor=10000>
    ```
  }];
}

def LWE_CoefficientEncoding
  : LWE_EncodingAttrWithScalingParam<"CoefficientEncoding", "coefficient_encoding"> {
  let summary = "An encoding of cleartexts directly as coefficients.";
  let description = [{
    A coefficient encoding of a list of integers asserts that the coefficients
    of the polynomials contain the integers, with the same semantics as
    `constant_coefficient_encoding` for per-coefficient encodings.

    A `scaling_factor` is optionally applied on the scalar when converting from
    a rounded floating point to an integer.

    Example:

    ```
    #coeff_encoding = #lwe.coefficient_encoding<scaling_factor=10000>
    ```
  }];
}

def LWE_InverseCanonicalEmbeddingEncoding
  : LWE_EncodingAttrWithScalingParam<"InverseCanonicalEncoding", "inverse_canonical_encoding"> {
  let summary = "An encoding of cleartexts via the inverse canonical embedding.";
  let description = [{
    Let $n$ be the degree of the polynomials in the plaintext space. An
    "inverse_canonical_encoding" of a list of real or complex values
    $v_1, \dots, v_{n/2}$ is (almost) the inverse of the following decoding
    map.

    Define a map $\tau_N$ that maps a polynomial $p \in \mathbb{Z}[x] / (x^N + 1)
    \to \mathbb{C}^{N/2}$ by evaluating it at the following $N/2$ points,
    where $\omega = e^{2 \pi i / 2N}$ is the primitive $2N$th root of unity:

    \[
      \omega, \omega^3, \omega^5, \dots, \omega^{N-1}
    \]

    Then the complete decoding operation is $\textup{Decode}(p) =
    (1/\Delta)\tau_N(p)$, where $\Delta$ is a scaling parameter and $\tau_N$ is
    the truncated canonical embedding above. The encoding operation is the
    inverse of the decoding operation, with some caveats explained below.

    The map $\tau_N$ is derived from the so-called _canonical embedding_
    $\tau$, though in the standard canonical embedding, we evaluate at all odd
    powers of the root of unity, $\omega, \omega^3, \dots, \omega^{2N-1}$. For
    polynomials in the slightly larger space $\mathbb{R}[x] / (x^N + 1)$, the
    image of the canonical embedding is the subspace $H \subset \mathbb{C}^N$
    defined by tuples $(z_1, \dots, z_N)$ such that $\overline{z_i} =
    \overline{z_{N-i+1}}$. Note that this property holds because polynomial
    evaluation commutes with complex conjugates, and the second half of the
    roots of unity evaluate are complex conjugates of the first half. The
    converse, that any such tuple with complex conjugate symmetry has an
    inverse under $\tau$ with all real coefficients, makes $\tau$ is a
    bijection onto $H$. $\tau$ and its inverse are explicitly computable as
    discrete Fourier Transforms.

    Because of the symmetry in canonical embedding for real polynomials, inputs
    to this encoding can be represented as a list of $N/2$ complex points, with
    the extra symmetric structure left implicit. $\tau_N$ and its inverse can
    also be explicitly computed without need to expand the vectors to length
    $N$.

    The rounding step is required to invert the decoding because, while
    cleartexts must be (implicitly) in the subspace $H$, they need not be the
    output of $\tau_N$ for an _integer_ polynomial. The rounding step ensures
    we can use integer polynomial plaintexts for the FHE operations. There are
    multiple rounding mechanisms, and this attribute does not specify which is
    used, because in theory two ciphertexts that have used different roundings
    are still compatible, though they may have different noise growth patterns.

    The scaling parameter $\Delta$ is specified by the `scaling_factor`, which
    are applied coefficient-wise using the same semantics as the
    `constant_coefficient_encoding`.

    A typical flow for the CKKS scheme using this encoding would be to apply an
    inverse FFT operation to invert the canonical embedding to be a polynomial
    with real coefficients, then encrypt scale the resulting polynomial's
    coefficients according to the scaling parameters, then round to get integer
    coefficients.

    Example:

    ```
    #canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor=10000>
    ```
  }];
}

def LWE_FullCRTPackingEncoding
  : LWE_EncodingAttrWithScalingParam<"FullCRTPackingEncoding", "full_crt_packing_encoding"> {
  let summary = "An encoding of cleartexts via CRT slots.";
  let description = [{
    This encoding maps a list of integers via the Chinese Remainder Theorem (CRT) into the plaintext space.

    Given a ring with irreducible ideal polynomial `f(x)` and coefficient
    modulus `q`, `f(x)` can be decomposed modulo `q` into a direct product of
    lower-degree polynomials. This allows full SIMD-style homomorphic operations
    across the slots formed from each factor.

    This attribute can only be used in the context of on full CRT packing, where
    the polynomial `f(x)` splits completely (into linear factors) and the number
    of slots equals the degree of `f(x)`. This happens when `q` is prime and `q
    = 1 mod n`.

    A `scaling_factor` is optionally applied on the scalar when converting from
    a rounded floating point to an integer.

    Example:

    ```
    #coeff_encoding = #lwe.full_crt_packing_encoding<scaling_factor=10000>
    ```
  }];
}

def LWE_AnyPlaintextEncodingAttr : LWE_EncodingAttrForLWE<"PlaintextEncoding", "plaintext_encoding"> {
  let returnType = "Attribute";
  let convertFromStorage = "$_self";
  string cppType = "Attribute";
  let predicate = Or<[
      LWE_ConstantCoefficientEncoding.predicate,
      LWE_CoefficientEncoding.predicate,
      LWE_InverseCanonicalEmbeddingEncoding.predicate,
      LWE_FullCRTPackingEncoding.predicate
    ]>;
}

def LWE_PlaintextSpaceAttr : AttrDef<LWE_Dialect, "PlaintextSpace"> {
  let mnemonic = "plaintext_space";
  let description = [{
    An attribute describing the plaintext space and the transformation from
    application data to plaintext space of an FHE scheme.

    The plaintext space information is the ring structure, which contains the
    plaintext modulus $t$, which may be a power of two in the case of CGGI
    ciphertexts, or a prime power for RLWE. LWE ciphertexts use the
    ideal polynomial of degree 1 $x$. The plaintext modulus used in LWE-based
    CGGI plaintexts describes the full message space $\mathbb{Z}_p$ including
    the padding bits. The application data info attribute describes the space
    $\mathbb{Z}_p'$ where $p' < p$ that the underlying message belongs to.

    For RLWE schemes, this will include the type of encoding of application data
    integers to a plaintext space `Z_p[X]/X^N + 1`. This may be a constant
    coefficient encoding, CRT-based packing for SIMD semantics, or other slot
    packing. When using full CRT packing, the ring must split into linear
    factors. The CKKS scheme will also include attributes describing the complex
    encoding, including the scaling factor, which will change after
    multiplication and rescaling.
  }];

  let parameters = (ins
    "::mlir::polynomial::RingAttr":$ring,
    LWE_AnyPlaintextEncodingAttr:$encoding
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

def LWE_KeyAttr : AttrDef<LWE_Dialect, "Key"> {
  let mnemonic = "key";
  let description = [{
    An attribute describing the key used for encrypting the ciphertext.

    This attribute includes a key identifier for the original key used to
    encrypt the secret key.

    The `key_size` parameter is used to describe the number of polynomials of
    the secret key. This is typically $1$ for RLWE ciphertexts and greater than
    $1$ for LWE instances. A ciphertext encrypted with a `key_size` of $k$ will
    have size $k+1$.

    The key basis describes the inner product used in the phase calculation in
    decryption. This attribute is only supported for RLWE ciphertexts whose
    `key_size` is $1$. An RLWE ciphertext is canonically encrypted against key
    basis `(1, s)`. After a multiplication, its size will increase and the basis
    will be `(1, s, s^2)`. The array that represents the key basis is
    constructed by listing the powers of `s` at each position of the array. For
    example, `(1, s, s^2)` corresponds to `[0, 1, 2]`, while `(1, s^2)`
    corresponds to `[0, 2]`.
  }];

  let parameters = (ins
    "::mlir::StringAttr":$id,
    DefaultValuedParameter<"unsigned", "1">:$size,
    OptionalArrayRefParameter<"unsigned int">:$basis
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LWE_EncryptionTypeEnum : I32EnumAttr<"LweEncryptionType", "An enum attribute representing an encryption method", [
  I32EnumAttrCase<"msb", 0>,
  I32EnumAttrCase<"lsb", 1>,
  I32EnumAttrCase<"mix", 2>
]> {
    let cppNamespace = "::mlir::heir::lwe";
}

def LWE_CiphertextSpaceAttr : AttrDef<LWE_Dialect, "CiphertextSpace"> {
  let mnemonic = "ciphertext_space";
  let description = [{
    An attribute describing the ciphertext space and the transformation from
    plaintext space to ciphertext space of an FHE scheme.

    The ciphertext space information includes the ring structure, which contains
    the ciphertext modulus $q$. Ciphertexts using an RNS representation for $q$
    will represent their ciphertext components in the ring attribute. Scalar LWE
    ciphertexts (as opposed to RLWE) use an ideal polynomial of degree 1, $x$.
    CGGI ciphertexts will typically use a power of two modulus.

    The ciphertext encoding info is used to describe the way the plaintext data
    is encoded into the ciphertext (in the MSB, LSB, or mixed).
  }];

  let parameters = (ins
    "::mlir::polynomial::RingAttr":$ring,
    "::mlir::heir::lwe::LweEncryptionType":$encryption_type
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LWE_ModulusChainAttr : AttrDef<LWE_Dialect, "ModulusChain"> {
  let mnemonic = "modulus_chain";
  let description = [{
    An attribute describing the elements of the modulus chain of an RLWE scheme.
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::IntegerAttr">:$elements,
    "int":$current
  );

  let assemblyFormat = "`<` `elements` `=` `<` $elements `>``,` `current` `=` $current `>`";

  // let genVerifyDecl = 1; // Verify index into list
}

#endif  // LIB_DIALECT_LWE_IR_NEWLWEATTRIBUTES_TD_
