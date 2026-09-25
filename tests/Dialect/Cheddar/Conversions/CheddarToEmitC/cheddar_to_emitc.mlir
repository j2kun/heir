// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --cse --canonicalize --drop-equivalent-buffer-results "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// Op coverage for the cheddar -> EmitC lowering over destination-passing ops.

!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!constant = !cheddar.constant
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!user_interface = !cheddar.user_interface
!parameter = !cheddar.parameter
!boot_context = !cheddar.boot_context

// CHECK: emitc.verbatim "namespace heir {
// A private cleartext constant lowers to static C-array storage.
// CHECK: emitc.global static @negative_zero_splat : !emitc.array<4xf32> = dense<-0.000000e+00>
memref.global "private" constant @negative_zero_splat : memref<4xf32> = dense<-0.000000e+00>

// A two-op chain: the intermediate is a local, the last op writes the out-param.
// CHECK: func.func @arith(
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"Context<word>">>
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK: emitc.member_call_opaque %arg0 "Add"(%[[V:.*]], %arg1, %arg2)
// CHECK: emitc.member_call_opaque %arg0 "Mult"(%arg3, %[[V]], %arg2)
func.func @arith(%ctx: !context, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.add %ctx, %a, %b, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %s = cheddar.mult %ctx, %r, %b, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %s : tensor<!ciphertext>
}

// Support values derived from a context and a UserInterface (the runtime
// helpers they call are checked at the top of the file).
// CHECK: func.func @support_values(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">>, %[[UI:.*]]: !emitc.ptr<!emitc.opaque<"UserInterface<word>">>,
// CHECK: %[[MAP:.*]] = emitc.member_call_opaque %[[UI]] "GetEvkMap"() : !emitc.ptr<!emitc.opaque<"UserInterface<word>">>, () -> !emitc.opaque<"const EvkMap<word>&">
// CHECK: %[[KEY:.*]] = emitc.call_opaque "heir::multiplicationKey"(%[[MAP]], %[[CTX]]) : (!emitc.opaque<"const EvkMap<word>&">, !emitc.ptr<!emitc.opaque<"Context<word>">>) -> !emitc.opaque<"const EvaluationKey<word>&">
// CHECK: "Relinearize"(%{{.*}}, %{{.*}}, %[[KEY]])
func.func @support_values(%ctx: !context, %ui: !user_interface, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %map = cheddar.get_evk_map %ui : (!user_interface) -> !evk_map
  %key = cheddar.get_mult_key %map, %ctx : (!evk_map, !context) -> !eval_key
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.relinearize %ctx, %ct, %key, %d0 : (!context, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @support_encoder(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">>,
// CHECK: %[[ENC:.*]] = emitc.call_opaque "heir::getEncoder"(%[[CTX]]) : (!emitc.ptr<!emitc.opaque<"Context<word>">>) -> !emitc.opaque<"const Encoder<word>&">
// CHECK: emitc.verbatim "{}.Encode({}, 1, {}.GetScale(1), {});" args %[[ENC]]
func.func @support_encoder(%ctx: !context, %input: tensor<4xf32>) -> tensor<!plaintext> {
  %enc = cheddar.get_encoder %ctx : (!context) -> !encoder
  %d = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %input, %d {level = 1 : i64} : (!encoder, tensor<4xf32>, tensor<!plaintext>) -> tensor<!plaintext>
  return %pt : tensor<!plaintext>
}

// CHEDDAR's Context overloads Add/Sub/Mult on the second operand type, so the
// `*_plain` / `*_const` ops dispatch to the base method name.
// CHECK: func.func @plain_const
// CHECK: emitc.member_call_opaque %arg0 "Add"
// CHECK: emitc.member_call_opaque %arg0 "Mult"
func.func @plain_const(%ctx: !context, %ct: tensor<!ciphertext>, %pt: tensor<!plaintext>, %c: tensor<!constant>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r1 = cheddar.add_plain %ctx, %ct, %pt, %d0 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %r2 = cheddar.mult_const %ctx, %r1, %c, %d1 : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The level_down target level is appended as a trailing opaque constant arg.
// CHECK: func.func @unary
// CHECK: emitc.member_call_opaque %arg0 "Neg"
// CHECK: emitc.member_call_opaque %arg0 "LevelDown"
// CHECK-SAME: #emitc.opaque<"2">
func.func @unary(%ctx: !context, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %n = cheddar.neg %ctx, %ct, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %l = cheddar.level_down %ctx, %n, %d1 {targetLevel = 2 : i64} : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %l : tensor<!ciphertext>
}

// CHECK: func.func @relin
// CHECK: emitc.member_call_opaque %arg0 "Relinearize"
// CHECK: emitc.member_call_opaque %arg0 "Rescale"
func.func @relin(%ctx: !context, %ct: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r1 = cheddar.relinearize %ctx, %ct, %k, %d0 : (!context, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %r2 = cheddar.rescale %ctx, %r1, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The HMult `rescale` flag is appended as the trailing opaque constant arg.
// CHECK: func.func @hmult
// CHECK: emitc.member_call_opaque %arg0 "HMult"
// CHECK-SAME: #emitc.opaque<"true">
func.func @hmult(%ctx: !context, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hmult %ctx, %a, %b, %k, %d0 {rescale = true} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Encode bridges a float message buffer through a std::vector<Complex> and
// uses CHEDDAR's canonical per-level scale; encrypt is an out-param method
// call.
// CHECK: func.func @enc_chain
// CHECK-SAME: !emitc.ptr<f64>
// CHECK: emitc.verbatim "{}.Encode({}, 5, {}.GetScale(5), {});"
// CHECK: emitc.member_call_opaque %arg2 "Encrypt"
func.func @enc_chain(%enc: !encoder, %msg: tensor<4xf64>, %ui: !user_interface) -> tensor<!ciphertext> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %msg, %dp {level = 5 : i64, logScale = 37 : i64} : (!encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  %dc = tensor.empty() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %dc : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %ct : tensor<!ciphertext>
}

// Decrypt is an out-param method call; decode reads into a temporary
// std::vector<Complex> then copies the real parts into the float buffer.
// CHECK: func.func @dec_chain
// CHECK-SAME: !emitc.ptr<f32>
// CHECK: emitc.member_call_opaque %arg1 "Decrypt"
// CHECK: emitc.verbatim "{}.Decode({}, {});"
// CHECK: emitc.verbatim "for (size_t _i = 0; _i < 4; ++_i) {}[_i] = {}.at(_i).real();"
func.func @dec_chain(%enc: !encoder, %ui: !user_interface, %ct: tensor<!ciphertext>, %dst: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.decrypt %ui, %ct, %dp : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  %msg = cheddar.decode %enc, %pt, %dst : (!encoder, tensor<!plaintext>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %msg : tensor<1x4xf32>
}

// HRot/HConj look the key up on the EvkMap operand; a static distance is baked
// into the format string, a dynamic one threads the SSA value.
// CHECK: func.func @hrot_static
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}.GetRotationKey(5), 5);"
func.func @hrot_static(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %evk, %ct, %d0 {static_distance = 5 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hrot_dyn
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}.GetRotationKey({}), {});"
// CHECK-SAME: %arg3, %arg3
func.func @hrot_dyn(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>, %d: index) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %evk, %ct, %d0, %d : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, index) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hconj_add
// CHECK: emitc.verbatim "{}->HConjAdd({}, {}, {}, {}.GetConjugationKey());"
func.func @hconj_add(%ctx: !context, %evk: !evk_map, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hconj_add %ctx, %evk, %a, %b, %d0 : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// cheddar.boot takes a !cheddar.boot_context, lowered to BootContext<word>*.
// CHECK: func.func @boot
// CHECK: emitc.member_call_opaque %arg0 "Boot"
func.func @boot(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.boot %ctx, %ct, %evk, %d0 : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}
