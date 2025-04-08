// RUN: heir-opt --lwe-to-openfhe %s

!Z34359754753_i64 = !mod_arith.int<34359754753 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
#alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>
#full_crt_packing_encoding1 = #lwe.full_crt_packing_encoding<scaling_factor = 14980>
#key = #lwe.key<>
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <67239937 : i64, 34359754753 : i64>, current = 0>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <67239937 : i64, 34359754753 : i64>, current = 1>
!rns_L0 = !rns.rns<!Z67239937_i64>
!rns_L1 = !rns.rns<!Z67239937_i64, !Z34359754753_i64>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 32), alignment = #alignment>
#ring_Z65537_i64_1_x32 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32>>
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
#ring_rns_L0_1_x32 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**32>>
!pkey_L1 = !lwe.new_lwe_public_key<key = #key, ring = #ring_rns_L1_1_x32>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>>
!pt1 = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding1>>
!skey_L0 = !lwe.new_lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x32>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32, encryption_type = lsb>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32, encryption_type = lsb>
!ct_L0 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding1>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>
!ct_L1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [67239937, 34359754753], P = [34359771137], plaintextModulus = 65537>, scheme.bgv} {
  func.func @simple_sum__encrypt__arg0(%arg0: tensor<32xi16>, %pk: !pkey_L1) -> !ct_L1 {
    %c31 = arith.constant 31 : index
    %c30 = arith.constant 30 : index
    %c29 = arith.constant 29 : index
    %c28 = arith.constant 28 : index
    %c27 = arith.constant 27 : index
    %c26 = arith.constant 26 : index
    %c25 = arith.constant 25 : index
    %c24 = arith.constant 24 : index
    %c23 = arith.constant 23 : index
    %c22 = arith.constant 22 : index
    %c21 = arith.constant 21 : index
    %c20 = arith.constant 20 : index
    %c19 = arith.constant 19 : index
    %c18 = arith.constant 18 : index
    %c17 = arith.constant 17 : index
    %c16 = arith.constant 16 : index
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0> : tensor<4096xi16>
    %extracted = tensor.extract %arg0[%c0] : tensor<32xi16>
    %inserted = tensor.insert %extracted into %cst[%c0] : tensor<4096xi16>
    %extracted_0 = tensor.extract %arg0[%c1] : tensor<32xi16>
    %inserted_1 = tensor.insert %extracted_0 into %inserted[%c1] : tensor<4096xi16>
    %extracted_2 = tensor.extract %arg0[%c2] : tensor<32xi16>
    %inserted_3 = tensor.insert %extracted_2 into %inserted_1[%c2] : tensor<4096xi16>
    %extracted_4 = tensor.extract %arg0[%c3] : tensor<32xi16>
    %inserted_5 = tensor.insert %extracted_4 into %inserted_3[%c3] : tensor<4096xi16>
    %extracted_6 = tensor.extract %arg0[%c4] : tensor<32xi16>
    %inserted_7 = tensor.insert %extracted_6 into %inserted_5[%c4] : tensor<4096xi16>
    %extracted_8 = tensor.extract %arg0[%c5] : tensor<32xi16>
    %inserted_9 = tensor.insert %extracted_8 into %inserted_7[%c5] : tensor<4096xi16>
    %extracted_10 = tensor.extract %arg0[%c6] : tensor<32xi16>
    %inserted_11 = tensor.insert %extracted_10 into %inserted_9[%c6] : tensor<4096xi16>
    %extracted_12 = tensor.extract %arg0[%c7] : tensor<32xi16>
    %inserted_13 = tensor.insert %extracted_12 into %inserted_11[%c7] : tensor<4096xi16>
    %extracted_14 = tensor.extract %arg0[%c8] : tensor<32xi16>
    %inserted_15 = tensor.insert %extracted_14 into %inserted_13[%c8] : tensor<4096xi16>
    %extracted_16 = tensor.extract %arg0[%c9] : tensor<32xi16>
    %inserted_17 = tensor.insert %extracted_16 into %inserted_15[%c9] : tensor<4096xi16>
    %extracted_18 = tensor.extract %arg0[%c10] : tensor<32xi16>
    %inserted_19 = tensor.insert %extracted_18 into %inserted_17[%c10] : tensor<4096xi16>
    %extracted_20 = tensor.extract %arg0[%c11] : tensor<32xi16>
    %inserted_21 = tensor.insert %extracted_20 into %inserted_19[%c11] : tensor<4096xi16>
    %extracted_22 = tensor.extract %arg0[%c12] : tensor<32xi16>
    %inserted_23 = tensor.insert %extracted_22 into %inserted_21[%c12] : tensor<4096xi16>
    %extracted_24 = tensor.extract %arg0[%c13] : tensor<32xi16>
    %inserted_25 = tensor.insert %extracted_24 into %inserted_23[%c13] : tensor<4096xi16>
    %extracted_26 = tensor.extract %arg0[%c14] : tensor<32xi16>
    %inserted_27 = tensor.insert %extracted_26 into %inserted_25[%c14] : tensor<4096xi16>
    %extracted_28 = tensor.extract %arg0[%c15] : tensor<32xi16>
    %inserted_29 = tensor.insert %extracted_28 into %inserted_27[%c15] : tensor<4096xi16>
    %extracted_30 = tensor.extract %arg0[%c16] : tensor<32xi16>
    %inserted_31 = tensor.insert %extracted_30 into %inserted_29[%c16] : tensor<4096xi16>
    %extracted_32 = tensor.extract %arg0[%c17] : tensor<32xi16>
    %inserted_33 = tensor.insert %extracted_32 into %inserted_31[%c17] : tensor<4096xi16>
    %extracted_34 = tensor.extract %arg0[%c18] : tensor<32xi16>
    %inserted_35 = tensor.insert %extracted_34 into %inserted_33[%c18] : tensor<4096xi16>
    %extracted_36 = tensor.extract %arg0[%c19] : tensor<32xi16>
    %inserted_37 = tensor.insert %extracted_36 into %inserted_35[%c19] : tensor<4096xi16>
    %extracted_38 = tensor.extract %arg0[%c20] : tensor<32xi16>
    %inserted_39 = tensor.insert %extracted_38 into %inserted_37[%c20] : tensor<4096xi16>
    %extracted_40 = tensor.extract %arg0[%c21] : tensor<32xi16>
    %inserted_41 = tensor.insert %extracted_40 into %inserted_39[%c21] : tensor<4096xi16>
    %extracted_42 = tensor.extract %arg0[%c22] : tensor<32xi16>
    %inserted_43 = tensor.insert %extracted_42 into %inserted_41[%c22] : tensor<4096xi16>
    %extracted_44 = tensor.extract %arg0[%c23] : tensor<32xi16>
    %inserted_45 = tensor.insert %extracted_44 into %inserted_43[%c23] : tensor<4096xi16>
    %extracted_46 = tensor.extract %arg0[%c24] : tensor<32xi16>
    %inserted_47 = tensor.insert %extracted_46 into %inserted_45[%c24] : tensor<4096xi16>
    %extracted_48 = tensor.extract %arg0[%c25] : tensor<32xi16>
    %inserted_49 = tensor.insert %extracted_48 into %inserted_47[%c25] : tensor<4096xi16>
    %extracted_50 = tensor.extract %arg0[%c26] : tensor<32xi16>
    %inserted_51 = tensor.insert %extracted_50 into %inserted_49[%c26] : tensor<4096xi16>
    %extracted_52 = tensor.extract %arg0[%c27] : tensor<32xi16>
    %inserted_53 = tensor.insert %extracted_52 into %inserted_51[%c27] : tensor<4096xi16>
    %extracted_54 = tensor.extract %arg0[%c28] : tensor<32xi16>
    %inserted_55 = tensor.insert %extracted_54 into %inserted_53[%c28] : tensor<4096xi16>
    %extracted_56 = tensor.extract %arg0[%c29] : tensor<32xi16>
    %inserted_57 = tensor.insert %extracted_56 into %inserted_55[%c29] : tensor<4096xi16>
    %extracted_58 = tensor.extract %arg0[%c30] : tensor<32xi16>
    %inserted_59 = tensor.insert %extracted_58 into %inserted_57[%c30] : tensor<4096xi16>
    %extracted_60 = tensor.extract %arg0[%c31] : tensor<32xi16>
    %inserted_61 = tensor.insert %extracted_60 into %inserted_59[%c31] : tensor<4096xi16>
    %pt = lwe.rlwe_encode %inserted_61 {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32} : tensor<4096xi16> -> !pt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L1) -> !ct_L1
    return %ct : !ct_L1
  }
}
