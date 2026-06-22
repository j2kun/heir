load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "InsertEquivalentConvLayouts",
    srcs = ["InsertEquivalentConvLayouts.cpp"],
    hdrs = ["InsertEquivalentConvLayouts.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils/Layout:ConvolutionLayoutBuilder",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tamagoyaki//:MLIREquivalence",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "InsertEquivalentConvLayouts",
    td_file = "InsertEquivalentConvLayouts.td",
)
