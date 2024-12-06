"""The compilation pipeline."""

from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
import tempfile

from heir_py.openfhe_config import OpenFHEConfig, DEFAULT_INSTALLED_OPENFHE_CONFIG
from heir_py.mlir_emitter import TextualMlirEmitter
from heir_py.pybind_helpers import pybind11_includes, pyconfig_ext_suffix
from numba.core.bytecode import ByteCode, FunctionIdentity
from numba.core.interpreter import Interpreter
from heir_py.heir_backend import HeirOptBackend, HeirTranslateBackend
from heir_py.clang import ClangBackend


@dataclass
class CompilationResult:
    # The module object containing the compiled functions
    module: object

    # The function name used to generate the various compiled functions
    func_name: str

    # A list of arg names (in order)
    arg_names: list[str]

    # A mapping from argument name to the compiled encryption function
    arg_enc_funcs: dict[str, object]

    # The compiled decryption function for the function result
    result_dec_func: object

    # The main compiled function
    main_func: object

    # Backend setup functions, if any
    setup_funcs: dict[str, object]


def run_compiler(
    function,
    openfhe_config: OpenFHEConfig = DEFAULT_INSTALLED_OPENFHE_CONFIG
) -> CompilationResult:
    """Run the compiler."""
    # The temporary workspace dir is so that heir-opt, heir-translate, and
    # clang can have places to write their output files. It is cleaned up once
    # the function returns, at which point the compiled python module has been
    # loaded into memory and the raw files are not needed.
    #
    # For debugging, add delete=False to TemporaryDirectory (python3.12+)
    # to leave the directory around after the context manager closes.
    # Otherwise, replace the context manager with `workspace_dir =
    # tempfile.mkdtemp()` and manually clean it up.
    with tempfile.TemporaryDirectory() as workspace_dir:
        func_id = FunctionIdentity.from_function(function)
        bytecode = ByteCode(func_id)
        ssa_ir = Interpreter(func_id).interpret(bytecode)
        mlir_textual = TextualMlirEmitter(ssa_ir).emit()
        func_name = func_id.func_name
        module_name = f"_heir_{func_name}"

        # FIXME: allow user to configure heir-opt path
        heir_opt = HeirOptBackend(binary_path="tools/heir-opt")
        # FIXME: construct heir-opt pipeline options from decorator
        heir_opt_options = [
            f"--secretize=function={func_name}",
            "--mlir-to-openfhe-bgv="
            f"entry-function={func_name} ciphertext-degree=32",
        ]
        heir_opt_output = heir_opt.run_binary(
            input=mlir_textual,
            options=heir_opt_options,
        )

        heir_translate = HeirTranslateBackend(binary_path="tools/heir-translate")
        cpp_filepath = Path(workspace_dir) / f"{func_name}.cpp"
        h_filepath = Path(workspace_dir) / f"{func_name}.h"
        pybind_filepath = Path(workspace_dir) / f"{func_name}_bindings.cpp"
        # FIXME: construct heir-translate pipeline options from decorator
        include_type_flag = "--openfhe-include-type=" + openfhe_config.include_type
        heir_translate.run_binary(
            input=heir_opt_output,
            options=["--emit-openfhe-pke-header", include_type_flag, "-o", h_filepath],
        )
        heir_translate.run_binary(
            input=heir_opt_output,
            options=["--emit-openfhe-pke", include_type_flag, "-o", cpp_filepath],
        )
        heir_translate.run_binary(
            input=heir_opt_output,
            options=[
                "--emit-openfhe-pke-pybind",
                f"--pybind-header-include={h_filepath.name}",
                f"--pybind-module-name={module_name}",
                "-o",
                pybind_filepath,
            ],
        )

        clang = ClangBackend()
        so_filepath = Path(workspace_dir) / f"{func_name}.so"
        linker_search_paths = [openfhe_config.lib_dir]
        clang.compile_to_shared_object(
            cpp_source_filepath=cpp_filepath,
            shared_object_output_filepath=so_filepath,
            include_paths=openfhe_config.include_dirs,
            linker_search_paths=linker_search_paths,
            link_libs=openfhe_config.link_libs,
        )

        ext_suffix = pyconfig_ext_suffix()
        pybind_so_filepath = Path(workspace_dir) / f"{module_name}{ext_suffix}"
        clang.compile_to_shared_object(
            cpp_source_filepath=pybind_filepath,
            shared_object_output_filepath=pybind_so_filepath,
            include_paths=openfhe_config.include_dirs
            + pybind11_includes()
            + [workspace_dir],
            linker_search_paths=linker_search_paths,
            link_libs=openfhe_config.link_libs,
            linker_args=["-rpath", ":".join(linker_search_paths)],
            abs_link_lib_paths=[so_filepath],
        )

        sys.path.append(workspace_dir)
        importlib.invalidate_caches()
        bound_module = importlib.import_module(module_name)

    result = CompilationResult(
        module=bound_module,
        func_name=func_name,
        arg_names=func_id.arg_names,
        arg_enc_funcs={
            arg_name: getattr(bound_module, f"{func_name}__encrypt__arg{i}")
            for i, arg_name in enumerate(func_id.arg_names)
        },
        result_dec_func=getattr(bound_module, f"{func_name}__decrypt__result0"),
        main_func=getattr(bound_module, func_name),
        setup_funcs={
            "generate_crypto_context": getattr(
                bound_module, f"{func_name}__generate_crypto_context"
            ),
            "configure_crypto_context": getattr(
                bound_module, f"{func_name}__configure_crypto_context"
            ),
        },
    )

    return result
