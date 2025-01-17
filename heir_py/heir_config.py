"""Configuration of HEIR compiler toolchain."""

import dataclasses
import os
import shutil

dataclass = dataclasses.dataclass


@dataclass(frozen=True)
class HEIRConfig:
  heir_opt_path: str
  heir_translate_path: str


DEVELOPMENT_HEIR_CONFIG = HEIRConfig(
    heir_opt_path="bazel-bin/tools/heir-opt",
    heir_translate_path="bazel-bin/tools/heir-translate",
)


def from_os_env() -> HEIRConfig:
  """Create a HEIRConfig from environment variables.

  Note, this is required for running tests under bazel, as the locations
  of the various binaries are determined by bazel.

  The order of preference is:

  1. Environment variable HEIR_OPT_PATH or HEIR_TRANSLATE_PATH
  2. The path to the heir-opt or heir-translate binary on the PATH
  3. The default development configuration (relative to the project root, in
     bazel-bin)

  Returns:
    The HEIRConfig
  """
  which_heir_opt = shutil.which("heir-opt")
  which_heir_translate = shutil.which("heir-translate")
  resolved_heir_opt_path = os.environ.get(
      "HEIR_OPT_PATH",
      which_heir_opt or DEVELOPMENT_HEIR_CONFIG.heir_opt_path,
  )
  resolved_heir_translate_path = os.environ.get(
      "HEIR_TRANSLATE_PATH",
      which_heir_translate or DEVELOPMENT_HEIR_CONFIG.heir_translate_path,
  )

  return HEIRConfig(
      heir_opt_path=resolved_heir_opt_path,
      heir_translate_path=resolved_heir_translate_path,
  )
