"""Custom dependency for tamagoyaki."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _tamagoyaki_dep_impl(_):
    new_git_repository(
        name = "tamagoyaki",
        remote = "https://github.com/jumerckx/Tamagoyaki.git",
        commit = "94c395cf8cec6d4b8c3e2ac92eee857b00bc9c15",
        build_file = "@heir//bazel:tamagoyaki.BUILD",
        patches = ["@heir//patches:tamagoyaki.patch"],
        patch_args = ["-p1"],
    )

tamagoyaki_dep = module_extension(
    implementation = _tamagoyaki_dep_impl,
)
