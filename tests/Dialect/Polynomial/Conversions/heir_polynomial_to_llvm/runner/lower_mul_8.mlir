// WARNING: this file is autogenerated. Do not edit manually, instead see
// tests/polynomial/runner/generate_test_cases.py

//-------------------------------------------------------
// entry and check_prefix are re-set per test execution
// DEFINE: %{entry} =
// DEFINE: %{check_prefix} =

// DEFINE: %{compile} = heir-opt %s --heir-polynomial-to-llvm
// DEFINE: %{run} = mlir-runner -e %{entry} -entry-point-result=void --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils"
// DEFINE: %{check} = FileCheck %s --check-prefix=%{check_prefix}
//-------------------------------------------------------

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// REDEFINE: %{entry} = test_8
// REDEFINE: %{check_prefix} = CHECK_TEST_8
// RUN: %{compile} | %{run} | %{check}

#ideal_8 = #polynomial.int_polynomial<1 + x**3>
!coeff_ty_8 = !mod_arith.int<7:i32>
#ring_8 = #polynomial.ring<coefficientType=!coeff_ty_8, polynomialModulus=#ideal_8>
!poly_ty_8 = !polynomial.polynomial<ring=#ring_8>

func.func @test_8() {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<-4 + x**1> : !poly_ty_8
  %1 = polynomial.constant int<-1 + 3x**1> : !poly_ty_8
  %2 = polynomial.mul %0, %1 : !poly_ty_8


  %3 = polynomial.to_tensor %2 : !poly_ty_8 -> tensor<3x!coeff_ty_8>
  %tensor = mod_arith.extract %3 : tensor<3x!coeff_ty_8> -> tensor<3xi32>

  %ref = bufferization.to_buffer %tensor : tensor<3xi32> to memref<3xi32>
  %U = memref.cast %ref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// expected_result: Poly(3*x**2 - 13*x + 4, x, domain='ZZ[7]')
// CHECK_TEST_8: {{(4|-3)}}, {{(1|-6)}}, {{(3|-4)}}
