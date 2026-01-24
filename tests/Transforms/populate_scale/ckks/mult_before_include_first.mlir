// RUN: heir-opt %s --secret-insert-mgmt-ckks=before-mul-include-first-mul --populate-scale-ckks=before-mul-include-first-mul | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func @mult
  func.func @mult(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    // check that argument are encrypted in double degree: 45 * 2 = 90
    // CHECK: secret.generic
    // CHECK-SAME: level = 3
    // CHECK-SAME: scale = 90
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    ^body(%input0: f32):
      %1 = arith.mulf %input0, %input0 : f32
      %2 = arith.addf %1, %1 : f32
      %3 = arith.mulf %2, %2 : f32
      secret.yield %3 : f32
    // CHECK: secret.yield
    // CHECK: ->
    // CHECK-SAME: level = 0
    // CHECK-SAME: scale = 45
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}
