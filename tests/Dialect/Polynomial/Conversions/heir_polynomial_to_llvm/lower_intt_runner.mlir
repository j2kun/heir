// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_NTT < %t

// This follows from example 3.10 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

#cycl = #polynomial.int_polynomial<1 + x**4>
!coeff_ty = !mod_arith.int<7681:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl>
#root = #polynomial.primitive_root<value=1925:i32, degree=8:i32>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func @test_poly_ntt() {
  %coeffsRaw = arith.constant dense<[1467,2807,3471,7621]> : tensor<4xi32>
  %coeffs = tensor.cast %coeffsRaw : tensor<4xi32> to tensor <4xi32, #ring>
  %coeffs_enc = mod_arith.encapsulate %coeffs : tensor<4xi32, #ring> -> tensor<4x!coeff_ty, #ring>
  %0 = polynomial.intt %coeffs_enc {root=#root} : tensor<4x!coeff_ty, #ring> -> !poly_ty

  %1 = polynomial.to_tensor %0 : !poly_ty -> tensor<4x!coeff_ty>
  %2 = mod_arith.extract %1 : tensor<4x!coeff_ty> -> tensor<4xi32>
  %3 = bufferization.to_buffer %2 : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %3 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NTT: [1, 2, 3, 4]
