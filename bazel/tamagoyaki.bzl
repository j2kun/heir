"""Custom dependency for tamagoyaki."""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def _tamagoyaki_dep_impl(_):
    new_local_repository(
        name = "tamagoyaki",
        path = "/usr/local/google/home/jkun/fhe/Tamagoyaki",
        build_file = "@heir//bazel:tamagoyaki.BUILD",
    )

tamagoyaki_dep = module_extension(
    implementation = _tamagoyaki_dep_impl,
)
