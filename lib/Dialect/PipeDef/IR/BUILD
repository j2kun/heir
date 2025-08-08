# PipeDef dialect implementation

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "PipeDefDialect.cpp",
    ],
    hdrs = [
        "PipeDefAttributes.h",
        "PipeDefDialect.h",
        "PipeDefOps.h",
        "PipeDefTypes.h",
    ],
    deps = [
        "attributes_inc_gen",
        "dialect_inc_gen",
        "ops_inc_gen",
        "types_inc_gen",
        ":PipeDefAttributes",
        ":PipeDefOps",
        ":PipeDefTypes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "PipeDefAttributes",
    srcs = [
        "PipeDefAttributes.cpp",
    ],
    hdrs = [
        "PipeDefAttributes.h",
        "PipeDefDialect.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "PipeDefTypes",
    srcs = [
        "PipeDefTypes.cpp",
    ],
    hdrs = [
        "PipeDefAttributes.h",
        "PipeDefDialect.h",
        "PipeDefTypes.h",
    ],
    deps = [
        ":PipeDefAttributes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "PipeDefOps",
    srcs = [
        "PipeDefOps.cpp",
    ],
    hdrs = [
        "PipeDefDialect.h",
        "PipeDefOps.h",
        "PipeDefTypes.h",
    ],
    deps = [
        ":PipeDefAttributes",
        ":PipeDefTypes",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "PipeDefAttributes.td",
        "PipeDefDialect.td",
        "PipeDefOps.td",
        "PipeDefTypes.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "PipeDef",
    kind = "dialect",
    td_file = "PipeDefDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "PipeDef",
    kind = "attribute",
    td_file = "PipeDefAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "PipeDef",
    kind = "type",
    td_file = "PipeDefTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "PipeDef",
    kind = "op",
    td_file = "PipeDefOps.td",
    deps = [
        ":td_files",
    ],
)
