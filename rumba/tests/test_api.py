from pathlib import Path

import pytest

import rumba
from rumba import RumbaUnsupportedError


def test_import_and_version():
    assert rumba.__version__


def test_njit_direct_decorator_executes_scalar_add():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add.py_func(1, 2) == 3
    assert add(1, 2) == 3
    assert add.signatures


def test_njit_call_decorator_executes_float_branch():
    @rumba.njit(cache=True, debug=True)
    def choose(a, b):
        if a > b:
            return a - b
        return b - a

    assert choose(1.5, 4.0) == pytest.approx(2.5)
    assert "if" in choose.inspect_c()


def test_jit_alias():
    @rumba.jit
    def add(a, b):
        return a + b

    assert add(4, 5) == 9


def test_explicit_signature():
    @rumba.njit(signature=("int64", "int64"))
    def add(a, b):
        return a + b

    assert add(2, 7) == 9


def test_unsupported_option_raises():
    with pytest.raises(RumbaUnsupportedError, match="unsupported njit option"):
        rumba.njit(parallel=True)


def test_loop_execution():
    @rumba.njit
    def total(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc

    assert total(6) == 15


def test_inspection_helpers():
    @rumba.njit
    def add(a, b):
        return a + b

    bytecode = add.inspect_bytecode()
    ast_summary = add.inspect_rumba_ast()
    assert any(inst["opname"] == "RETURN_VALUE" for inst in bytecode)
    assert ast_summary["name"] == "add"
    assert ast_summary["body"] == ["Return"]


def test_inspect_compile_command_requires_compilation():
    @rumba.njit
    def add(a, b):
        return a + b

    with pytest.raises(RumbaUnsupportedError, match="before compilation"):
        add.inspect_compile_command()


def test_inspect_compile_command_returns_single_compiled_artifact_command():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    command = add.inspect_compile_command()
    artifact = next(iter(add._compiled.values()))
    assert isinstance(command, list)
    assert command == artifact.compile_command
    assert command[0]


def test_inspect_compile_command_requires_signature_for_multiple_artifacts():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add(1.5, 2.5) == pytest.approx(4.0)
    with pytest.raises(RumbaUnsupportedError, match="requires a signature"):
        add.inspect_compile_command()

    int_command = add.inspect_compile_command(("int64", "int64"))
    assert int_command == list(add._compiled.values())[0].compile_command


def test_inspect_cache_path_returns_build_directory():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    cache_path = Path(add.inspect_cache_path())
    artifact = next(iter(add._compiled.values()))
    assert cache_path.is_dir()
    assert str(cache_path) == artifact.cache_path
    assert artifact.key in str(cache_path)


def test_inspect_cache_path_requires_compiled_signature():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    with pytest.raises(RumbaUnsupportedError, match="signature has not been compiled"):
        add.inspect_cache_path(("float64", "float64"))


def test_inspect_cache_path_requires_signature_for_multiple_artifacts():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add(1.5, 2.5) == pytest.approx(4.0)
    with pytest.raises(RumbaUnsupportedError, match="requires a signature"):
        add.inspect_cache_path()

    float_path = add.inspect_cache_path(("float64", "float64"))
    assert float_path == list(add._compiled.values())[1].cache_path


def test_unsupported_list_argument_raises():
    @rumba.njit
    def first(x):
        return x[0]

    with pytest.raises(RumbaUnsupportedError, match="unsupported argument type"):
        first([1])
