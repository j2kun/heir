// RUN: heir-opt --verify-diagnostics %s

!poly_ty = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=1001 : i32>>
!poly_ty_1 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=7 : i32>>
!poly_ty_2 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=11 : i32>>
!poly_ty_3 = !polynomial.polynomial<ring=<coefficient_type=i32, coefficient_modulus=13 : i32>>

!rns_ty = !rns.rns<!poly_ty_1, !poly_ty_2, !poly_ty_3>

func.func @test_recompose_decompose(%arg0: !poly_ty) -> !poly_ty {
  %0 = rns.decompose %arg0 : !poly_ty -> !rns_ty
  %1 = rns.recompose %0 : !rns_ty -> !poly_ty
  return %1 : !poly_ty
}

#ideal_2 = #polynomial.int_polynomial<1 + x**2048>
#ring_bad = #polynomial.ring<coefficient_type=i32, coefficient_modulus=3180146689:i32, ideal=#ideal_2>
!poly_ty_bad = !polynomial.polynomial<ring=#ring_bad>
// expected-error@+1 {{RNS type has incompatible basis types}}
!ty_bad = !rns.rns<!poly_ty_1, !poly_ty_2, !poly_ty_bad>
