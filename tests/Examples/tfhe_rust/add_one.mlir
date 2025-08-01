#key = #lwe.key<slot_index = 0>
#preserve_overflow = #lwe.preserve_overflow<>
#app_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!pt_ty = !lwe.lwe_plaintext<application_data = #app_data, plaintext_space = #pspace>
!ct_ty = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

module {
  func.func @fn_under_test(%arg0: memref<8x!ct_ty>) -> memref<8x!ct_ty> {
    %true = arith.constant true
    %false = arith.constant false
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<8xi1>
    memref.store %true, %alloc[%c0] : memref<8xi1>
    memref.store %false, %alloc[%c1] : memref<8xi1>
    memref.store %false, %alloc[%c2] : memref<8xi1>
    memref.store %false, %alloc[%c3] : memref<8xi1>
    memref.store %false, %alloc[%c4] : memref<8xi1>
    memref.store %false, %alloc[%c5] : memref<8xi1>
    memref.store %false, %alloc[%c6] : memref<8xi1>
    memref.store %false, %alloc[%c7] : memref<8xi1>
    %0 = memref.load %alloc[%c0] : memref<8xi1>
    %1 = memref.load %arg0[%c0] : memref<8x!ct_ty>
    %2 = lwe.encode %false { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %3 = lwe.trivial_encrypt %2  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %4 = lwe.encode %0 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %5 = lwe.trivial_encrypt %4  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %6 = cggi.lut3 %3, %5, %1 {lookup_table = 8 : ui8} : !ct_ty
    %7 = memref.load %alloc[%c1] : memref<8xi1>
    %8 = memref.load %arg0[%c1] : memref<8x!ct_ty>
    %9 = lwe.encode %7 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %10 = lwe.trivial_encrypt %9  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %11 = cggi.lut3 %10, %8, %6 {lookup_table = 150 : ui8} : !ct_ty
    %12 = cggi.lut3 %10, %8, %6 {lookup_table = 23 : ui8} : !ct_ty
    %13 = memref.load %alloc[%c2] : memref<8xi1>
    %14 = memref.load %arg0[%c2] : memref<8x!ct_ty>
    %15 = lwe.encode %13 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %16 = lwe.trivial_encrypt %15  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %17 = cggi.lut3 %16, %14, %12 {lookup_table = 43 : ui8} : !ct_ty
    %18 = memref.load %alloc[%c3] : memref<8xi1>
    %19 = memref.load %arg0[%c3] : memref<8x!ct_ty>
    %20 = lwe.encode %18 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %21 = lwe.trivial_encrypt %20  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %22 = cggi.lut3 %21, %19, %17 {lookup_table = 43 : ui8} : !ct_ty
    %23 = memref.load %alloc[%c4] : memref<8xi1>
    %24 = memref.load %arg0[%c4] : memref<8x!ct_ty>
    %25 = lwe.encode %23 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %26 = lwe.trivial_encrypt %25  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %27 = cggi.lut3 %26, %24, %22 {lookup_table = 43 : ui8} : !ct_ty
    %28 = memref.load %alloc[%c5] : memref<8xi1>
    %29 = memref.load %arg0[%c5] : memref<8x!ct_ty>
    %30 = lwe.encode %28 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %31 = lwe.trivial_encrypt %30  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %32 = cggi.lut3 %31, %29, %27 {lookup_table = 43 : ui8} : !ct_ty
    %33 = memref.load %alloc[%c6] : memref<8xi1>
    %34 = memref.load %arg0[%c6] : memref<8x!ct_ty>
    %35 = lwe.encode %33 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %36 = lwe.trivial_encrypt %35  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %37 = cggi.lut3 %36, %34, %32 {lookup_table = 105 : ui8} : !ct_ty
    %38 = cggi.lut3 %36, %34, %32 {lookup_table = 43 : ui8} : !ct_ty
    %39 = memref.load %alloc[%c7] : memref<8xi1>
    %40 = memref.load %arg0[%c7] : memref<8x!ct_ty>
    %41 = lwe.encode %39 { overflow = #preserve_overflow, plaintext_bits = 3 : index } : i1 to !pt_ty
    %42 = lwe.trivial_encrypt %41  { ciphertext_bits = 32 : index } : !pt_ty to !ct_ty
    %43 = cggi.lut3 %42, %40, %38 {lookup_table = 105 : ui8} : !ct_ty
    %44 = cggi.lut3 %3, %5, %1 {lookup_table = 6 : ui8} : !ct_ty
    %45 = cggi.lut3 %16, %14, %12 {lookup_table = 105 : ui8} : !ct_ty
    %46 = cggi.lut3 %21, %19, %17 {lookup_table = 105 : ui8} : !ct_ty
    %47 = cggi.lut3 %26, %24, %22 {lookup_table = 105 : ui8} : !ct_ty
    %48 = cggi.lut3 %31, %29, %27 {lookup_table = 105 : ui8} : !ct_ty
    %alloc_0 = memref.alloc() : memref<8x!ct_ty>
    memref.store %44, %alloc_0[%c0] : memref<8x!ct_ty>
    memref.store %11, %alloc_0[%c1] : memref<8x!ct_ty>
    memref.store %45, %alloc_0[%c2] : memref<8x!ct_ty>
    memref.store %46, %alloc_0[%c3] : memref<8x!ct_ty>
    memref.store %47, %alloc_0[%c4] : memref<8x!ct_ty>
    memref.store %48, %alloc_0[%c5] : memref<8x!ct_ty>
    memref.store %37, %alloc_0[%c6] : memref<8x!ct_ty>
    memref.store %43, %alloc_0[%c7] : memref<8x!ct_ty>
    return %alloc_0 : memref<8x!ct_ty>
  }
}
