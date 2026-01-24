load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ScaleState",
    hdrs = [
        "ScaleState.h",
    ],
    deps = [
        "@heir//lib/Utils:Overloaded",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ScaleModel",
    srcs = [
        "ScaleModel.cpp",
    ],
    hdrs = [
        "ScaleModel.h",
    ],
    deps = [
        ":ScaleState",
        "@heir//lib/Parameters/BGV:Params",
        "@heir//lib/Parameters/CKKS:Params",
        "@heir//lib/Utils:APIntUtils",
        "@heir//lib/Utils:Overloaded",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ScaleAnalysis",
    srcs = [
        "ScaleAnalysis.cpp",
    ],
    hdrs = [
        "ScaleAnalysis.h",
    ],
    deps = [
        ":ScaleModel",
        ":ScaleState",
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:APIntUtils",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils:Overloaded",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
