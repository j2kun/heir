"""Custom dependency for tamagoyaki."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _tamagoyaki_dep_impl(_):
    new_git_repository(
        name = "tamagoyaki",
        remote = "https://github.com/jumerckx/Tamagoyaki.git",
        commit = "30f1b4ef66575800c556b74abb73ad2f4e9020ef",
        build_file = "@heir//bazel:tamagoyaki.BUILD",
        patches = ["@heir//patches:tamagoyaki.patch"],
        patch_args = ["-p1"],
    )

tamagoyaki_dep = module_extension(
    implementation = _tamagoyaki_dep_impl,
)
