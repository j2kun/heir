package binops

import (
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

func TestBinops(t *testing.T) {
	var err error
	var params bgv.Parameters

	// 128-bit secure parameters enabling depth-7 circuits.
	// LogN:14, LogQP: 431.
	if params, err = bgv.NewParametersFromLiteral(
		bgv.ParametersLiteral{
			LogN:             14,                                    // log2(ring degree)
			LogQ:             []int{55, 45, 45, 45, 45, 45, 45, 45}, // log2(primes Q) (ciphertext modulus)
			LogP:             []int{61},                             // log2(primes P) (auxiliary modulus)
			PlaintextModulus: 0x10001,                               // log2(scale)
		}); err != nil {
		panic(err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	ecd := bgv.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)
	dec := rlwe.NewDecryptor(params, sk)
	relinKeys := kgen.GenRelinearizationKeyNew(sk)
	galKey := kgen.GenGaloisKeyNew(5, sk)
	evalKeys := rlwe.NewMemEvaluationKeySet(relinKeys, galKey)
	evaluator := bgv.NewEvaluator(params, evalKeys /*scaleInvariant=*/, false)

	// Vector of plaintext values
	// 0, 1, 2, 3
	arg0 := []int16{0, 1, 2, 3}
	// 1, 2, 3, 4
	arg1 := []int16{1, 2, 3, 4}

	expected := []int16{6, 15, 28, 1}

	ct0, ct1 := add__encrypt(evaluator, params, ecd, enc, arg0, arg1)

	resultCt := add(evaluator, params, ecd, ct0, ct1)

	result := add__decrypt(evaluator, params, ecd, dec, resultCt)

	for i := range 4 {
		if result[i] != expected[i] {
			t.Errorf("Decryption error at index %d: %d != %d", i, result[i], expected[i])
		}
	}
}
