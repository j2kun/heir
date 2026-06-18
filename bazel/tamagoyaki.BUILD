load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

# TD files
td_library(
    name = "EquivalenceTdFiles",
    srcs = ["include/EquivalenceDialect.td"],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

td_library(
    name = "EmatchTdFiles",
    srcs = ["include/EmatchDialect.td"],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PDLDialectTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

# TableGen generations
gentbl_cc_library(
    name = "EquivalenceDialectGen",
    tbl_outs = {
        "include/EquivalenceDialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=equivalence",
        ],
        "include/EquivalenceDialect.cpp.inc": [
            "-gen-dialect-defs",
            "-dialect=equivalence",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/EquivalenceDialect.td",
    deps = [":EquivalenceTdFiles"],
)

gentbl_cc_library(
    name = "EquivalenceOpsGen",
    tbl_outs = {
        "include/EquivalenceOps.h.inc": ["-gen-op-decls"],
        "include/EquivalenceOps.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/EquivalenceDialect.td",
    deps = [":EquivalenceTdFiles"],
)

gentbl_cc_library(
    name = "EquivalencePassesGen",
    tbl_outs = {
        "include/EquivalencePasses.h.inc": [
            "--gen-pass-decls",
            "-name=Equivalence",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/EquivalenceDialect.td",
    deps = [":EquivalenceTdFiles"],
)

gentbl_cc_library(
    name = "EquivalenceAttrsGen",
    tbl_outs = {
        "include/EquivalenceAttrs.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=equivalence",
        ],
        "include/EquivalenceAttrs.cpp.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect=equivalence",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/EquivalenceDialect.td",
    deps = [":EquivalenceTdFiles"],
)

gentbl_cc_library(
    name = "EmatchDialectGen",
    tbl_outs = {
        "include/EmatchDialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=ematch",
        ],
        "include/EmatchDialect.cpp.inc": [
            "-gen-dialect-defs",
            "-dialect=ematch",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/EmatchDialect.td",
    deps = [":EmatchTdFiles"],
)

gentbl_cc_library(
    name = "EmatchOpsGen",
    tbl_outs = {
        "include/EmatchOps.h.inc": ["-gen-op-decls"],
        "include/EmatchOps.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/EmatchDialect.td",
    deps = [":EmatchTdFiles"],
)

gentbl_cc_library(
    name = "EmatchPassesGen",
    tbl_outs = {
        "include/EmatchPasses.h.inc": [
            "--gen-pass-decls",
            "-name=Ematch",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/EmatchDialect.td",
    deps = [":EmatchTdFiles"],
)

# Libraries
cc_library(
    name = "TamagoyakiTiming",
    srcs = ["src/TamagoyakiTiming.cpp"],
    hdrs = ["include/TamagoyakiTiming.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "MLIREquivalence",
    srcs = ["src/EquivalenceDialect.cpp"],
    hdrs = [
        "include/EquivalenceDialect.h",
        "include/EquivalenceUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":EquivalenceAttrsGen",
        ":EquivalenceDialectGen",
        ":EquivalenceOpsGen",
        ":EquivalencePassesGen",
        ":TamagoyakiTiming",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:RegisterAllPasses",
    ],
)

cc_library(
    name = "MLIREmatch",
    srcs = [
        "src/EmatchDialect.cpp",
        "src/Utils/ClassOpUnionFind.cpp",
        "src/Utils/HashConsPatternRewriter.cpp",
        "src/Utils/MutableScopedHashTable.cpp",
    ],
    hdrs = [
        "include/EmatchDialect.h",
        "include/EmatchUtils.h",
        "include/Utils/ClassOpUnionFind.h",
        "include/Utils/HashConsPatternRewriter.h",
        "include/Utils/MutableScopedHashTable.h",
        "vendor/mlir/Bytecode.h",
        "vendor/mlir/SimpleOperationInfo.h",
    ],
    includes = ["include"],
    deps = [
        ":EmatchDialectGen",
        ":EmatchOpsGen",
        ":EmatchPassesGen",
        ":MLIREquivalence",
        ":TamagoyakiTiming",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:PDLInterpDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:RegisterAllDialects",
    ],
)

cc_binary(
    name = "tamagoyaki-opt",
    srcs = ["src/tamagoyaki-opt.cpp"],
    deps = [
        ":MLIREmatch",
        ":MLIREquivalence",
        ":TamagoyakiTiming",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:RegisterAllPasses",
    ],
)
