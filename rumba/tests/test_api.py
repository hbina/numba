from pathlib import Path
import math
import math as m

import numpy as np
import pytest

import rumba
from rumba import RumbaUnsupportedError

_COMPARISON_OPERATORS = ("<", "<=", "==", "!=", ">", ">=")
_ORDERABLE_SCALAR_PAIRS = (
    (1, 2),
    (1.5, 2.5),
)
_EQUALITY_SCALAR_PAIRS = _ORDERABLE_SCALAR_PAIRS + (
    (False, True),
)


@rumba.njit
def _typed_ast_helper(a):
    return a + 1.5


@rumba.njit
def _uncalled_decorated_helper(a):
    return a * 2


@rumba.njit(signature=("int64",))
def _explicit_int_helper(a):
    return a + 1


def _plain_python_helper(a):
    return a + 1


def _plain_python_gen(n):
    for i in range(n):
        yield i


@rumba.njit
def _gen_times_two(n):
    for i in range(n):
        yield i * 2


@rumba.njit
def _gen_branch(n):
    for i in range(n):
        if i == 1:
            yield i
        yield i + 2


@rumba.njit
def _gen_mixed(n):
    yield 1
    yield 2.5


@rumba.njit
def _gen_return_value(n):
    yield n
    return n


@rumba.njit
def _gen_float(n):
    for i in range(n):
        yield float(i) + 0.5


@rumba.njit
def _gen_bool(n):
    for i in range(n):
        yield i == 1


@rumba.njit
def _recursive_helper(a):
    if a == 0:
        return 0
    return _recursive_helper(a - 1)


@rumba.njit
def _helper_with_default(a=1):
    return a


@rumba.njit
def _helper_with_varargs(*args):
    return 1


@rumba.njit
def _helper_with_keyword_only(*, a):
    return a


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


def test_print_supports_static_string_literals_and_scalars(capfd):
    @rumba.njit
    def report(value):
        print("hello", value, "world")
        return value + 1

    assert report(5) == 6
    assert capfd.readouterr().out == "hello 5 world\n"

    c_source = report.inspect_c()
    assert 'fputs("hello", stdout);' in c_source
    assert 'fputs("world", stdout);' in c_source


def test_print_rejects_array_arguments():
    @rumba.njit
    def report(values):
        print("values", values)
        return 0

    values = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(RumbaUnsupportedError, match="print arguments must be scalar"):
        report(values)


@pytest.mark.parametrize("op", _COMPARISON_OPERATORS)
@pytest.mark.parametrize(("left", "right"), _ORDERABLE_SCALAR_PAIRS)
def test_scalar_comparisons_return_python_bool(op, left, right):
    namespace = {}
    exec(f"def compare(a, b):\n    return a {op} b\n", namespace)
    compare = rumba.njit(namespace["compare"])

    result = compare(left, right)

    assert result == compare.py_func(left, right)
    assert type(result) is bool


@pytest.mark.parametrize("op", ("==", "!="))
@pytest.mark.parametrize(("left", "right"), _EQUALITY_SCALAR_PAIRS)
def test_scalar_equality_comparisons_return_python_bool(op, left, right):
    namespace = {}
    exec(f"def compare(a, b):\n    return a {op} b\n", namespace)
    compare = rumba.njit(namespace["compare"])

    result = compare(left, right)

    assert result == compare.py_func(left, right)
    assert type(result) is bool


@pytest.mark.parametrize(("left", "right"), [(1, 2.5), (1, True), (1.5, 2), (False, 1)])
def test_mixed_scalar_comparisons_require_explicit_cast(left, right):
    @rumba.njit
    def compare(a, b):
        return a == b

    with pytest.raises(RumbaUnsupportedError, match="exact matching scalar types"):
        compare(left, right)


def test_bool_ordering_comparisons_are_rejected():
    @rumba.njit
    def compare(a, b):
        return a < b

    with pytest.raises(RumbaUnsupportedError, match="ordering comparisons require"):
        compare(False, True)


def test_unary_not_returns_python_bool_for_bool_input():
    @rumba.njit
    def invert(a):
        return not a

    result = invert(True)

    assert result is False
    assert type(result) is bool


@pytest.mark.parametrize("value", [1, 0.0])
def test_unary_not_requires_bool(value):
    @rumba.njit
    def invert(a):
        return not a

    with pytest.raises(RumbaUnsupportedError, match="not requires bool"):
        invert(value)


def test_debug_option_emits_compilation_and_runtime_details(capfd):
    @rumba.njit(debug=True)
    def add(a, b):
        tmp = a + b
        return tmp

    assert add(1, 2) == 3

    captured = capfd.readouterr()
    assert "[rumba-debug] call: selected signature: [int64, int64]" in captured.err
    assert "[rumba-debug] compile: typed function:" in captured.err
    assert "locals:" in captured.err
    assert "[rumba-debug] compile: generated C source follows" in captured.err
    assert "[rumba-debug] runtime: native wrapper argument count: 2" in captured.err
    assert "[rumba-debug] dispatcher: artifact return type: int64" in captured.err


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


def test_native_wrapper_handles_mixed_three_scalar_signature():
    @rumba.njit
    def combine(a, b, c):
        return float(a) + b + float(c)

    assert combine(1, 2.5, 3) == pytest.approx(6.5)
    c_source = combine.inspect_c()
    assert "double rumba_entry(int64_t a, double b, int64_t c)" in c_source
    assert (
        "*(double *)ret = rumba_entry(*(int64_t *)args[0], *(double *)args[1], *(int64_t *)args[2]);"
        in c_source
    )


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


def test_generator_helper_loop_executes_expression_yields():
    @rumba.njit
    def total(n):
        acc = 0
        for value in _gen_times_two(n):
            acc += value
        return acc

    assert total(5) == 20
    typed = total.inspect_typed_ast()
    assert typed["body"][1]["kind"] == "ForGenerator"
    assert typed["body"][1]["target_type"] == "int64"


def test_generator_helper_loop_executes_branch_and_multiple_yields():
    @rumba.njit
    def total(n):
        acc = 0
        for value in _gen_branch(n):
            acc += value
        return acc

    assert total(3) == 10


def test_generator_helper_loop_executes_float_and_bool_yields():
    @rumba.njit
    def total_float(n):
        acc = 0.0
        for value in _gen_float(n):
            acc += value
        return acc

    @rumba.njit
    def count_true(n):
        acc = 0
        for value in _gen_bool(n):
            if value:
                acc += 1
        return acc

    assert total_float(3) == pytest.approx(4.5)
    assert count_true(4) == 1


def test_generator_mixed_yield_types_raise():
    @rumba.njit
    def total(n):
        acc = 0
        for value in _gen_mixed(n):
            acc += value
        return acc

    with pytest.raises(RumbaUnsupportedError, match="generator yield values require exact matching"):
        total(1)


def test_generator_return_value_raises():
    @rumba.njit
    def total(n):
        acc = 0
        for value in _gen_return_value(n):
            acc += value
        return acc

    with pytest.raises(RumbaUnsupportedError, match="return values from generator helpers"):
        total(1)


def test_plain_python_generator_helper_raises():
    @rumba.njit
    def total(n):
        acc = 0
        for value in _plain_python_gen(n):
            acc += value
        return acc

    with pytest.raises(
        RumbaUnsupportedError,
        match="helper calls require @rumba.njit-decorated functions",
    ):
        total(1)


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


def test_inspect_typed_ast_requires_compilation():
    @rumba.njit
    def add(a, b):
        return a + b

    with pytest.raises(RumbaUnsupportedError, match="before compilation"):
        add.inspect_typed_ast()


def test_inspect_typed_ast_returns_single_compiled_artifact():
    @rumba.njit
    def add(a, b):
        tmp = float(a) + b
        return tmp

    assert add(1, 2.5) == pytest.approx(3.5)
    typed = add.inspect_typed_ast()

    assert typed["name"] == "add"
    assert typed["signature"] == ["int64", "float64"]
    assert typed["return_type"] == "float64"
    assert typed["args"] == [
        {"name": "a", "type": "int64"},
        {"name": "b", "type": "float64"},
    ]
    assert typed["locals"] == {"tmp": "float64"}
    assign = typed["body"][0]
    assert assign["kind"] == "Assign"
    assert assign["target"] == "tmp"
    assert assign["target_type"] == "float64"
    assert assign["value"]["kind"] == "BinOp"
    assert assign["value"]["type"] == "float64"
    assert assign["value"]["reason"] == "exact_scalar_op"
    assert assign["value"]["left"]["kind"] == "IntrinsicCall"
    assert assign["value"]["left"]["intrinsic"] == "float"


def test_inspect_typed_ast_requires_signature_for_multiple_artifacts():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add(1.5, 2.5) == pytest.approx(4.0)
    with pytest.raises(RumbaUnsupportedError, match="requires a signature"):
        add.inspect_typed_ast()

    typed = add.inspect_typed_ast(("int64", "int64"))
    assert typed["signature"] == ["int64", "int64"]
    assert typed["return_type"] == "int64"


def test_inspect_typed_ast_requires_compiled_signature():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    with pytest.raises(RumbaUnsupportedError, match="signature has not been compiled"):
        add.inspect_typed_ast(("float64", "float64"))


def test_inspect_typed_ast_records_if_test_bool_and_loop_index():
    @rumba.njit
    def choose_total(n, flag):
        acc = 0
        for i in range(n):
            acc += i
        if flag:
            return acc
        return 0

    assert choose_total(4, True) == 6
    typed = choose_total.inspect_typed_ast()

    assert typed["locals"]["i"] == "int64"
    loop = typed["body"][1]
    assert loop["kind"] == "ForRange"
    assert loop["target"] == "i"
    assert loop["target_type"] == "int64"
    assert loop["reason"] == "range_index"
    branch = typed["body"][2]
    assert branch["kind"] == "If"
    assert branch["test_type"] == "bool"
    assert branch["test"]["reason"] == "environment"


def test_inspect_typed_ast_records_while_test_and_body():
    @rumba.njit
    def countdown(n):
        acc = 0
        while n > 0:
            acc += n
            n -= 1
        return acc

    assert countdown(4) == 10
    typed = countdown.inspect_typed_ast()

    loop = typed["body"][1]
    assert loop["kind"] == "While"
    assert loop["test_type"] == "bool"
    assert loop["test"]["kind"] == "Compare"
    assert [stmt["kind"] for stmt in loop["body"]] == ["AugAssign", "AugAssign"]


def test_inspect_typed_ast_exposes_helper_return_type():
    @rumba.njit
    def use_helper(a):
        return _typed_ast_helper(a)

    assert use_helper(2.0) == pytest.approx(3.5)
    call = use_helper.inspect_typed_ast()["body"][0]["value"]

    assert call["kind"] == "Call"
    assert call["type"] == "float64"
    assert call["reason"] == "helper_return"
    assert call["helper"]["name"] == "_typed_ast_helper"
    assert call["helper"]["return_type"] == "float64"


def test_builtin_scalar_intrinsics_execute_and_inspect_as_intrinsics():
    @rumba.njit
    def use_intrinsics(a, b):
        return max(a, b) + min(a, b) + abs(a)

    assert use_intrinsics(-5, 2) == 2
    typed = use_intrinsics.inspect_typed_ast()
    value = typed["body"][0]["value"]

    assert value["kind"] == "BinOp"
    assert value["left"]["kind"] == "BinOp"
    assert value["left"]["left"]["kind"] == "IntrinsicCall"
    assert value["left"]["left"]["intrinsic"] == "max"
    assert value["left"]["right"]["kind"] == "IntrinsicCall"
    assert value["left"]["right"]["intrinsic"] == "min"
    assert value["right"]["kind"] == "IntrinsicCall"
    assert value["right"]["intrinsic"] == "abs"
    assert "rumba_max_int64_2" in use_intrinsics.inspect_c()
    assert "rumba_min_int64_2" in use_intrinsics.inspect_c()


def test_explicit_scalar_casts_execute_and_inspect_as_intrinsics():
    @rumba.njit
    def use_casts(a, b, c):
        if bool(c):
            return int(a) + int(b)
        return int(False)

    assert use_casts(1.5, True, 1) == 2
    assert use_casts(1.5, True, 0) == 0
    typed = use_casts.inspect_typed_ast()
    branch = typed["body"][0]
    ret = branch["body"][0]["value"]
    c_source = use_casts.inspect_c()

    assert branch["test"]["kind"] == "IntrinsicCall"
    assert branch["test"]["intrinsic"] == "bool"
    assert ret["left"]["intrinsic"] == "int"
    assert ret["right"]["intrinsic"] == "int"
    assert "(int64_t)(a)" in c_source
    assert "(int64_t)(b)" in c_source
    assert "((c) != 0)" in c_source


def test_explicit_float_and_bool_casts_execute():
    @rumba.njit
    def use_casts(a, b):
        return float(a) + b

    @rumba.njit
    def truthy(a):
        return bool(a)

    assert use_casts(False, 1.5) == pytest.approx(1.5)
    assert truthy(0) is False
    assert truthy(1) is True
    assert truthy(0.0) is False
    assert truthy(1.25) is True


def test_int64_true_division_returns_float64():
    @rumba.njit
    def divide(a, b):
        return a / b

    assert divide(3, 2) == pytest.approx(1.5)
    typed = divide.inspect_typed_ast()
    c_source = divide.inspect_c()

    assert typed["return_type"] == "float64"
    assert "((double)(a) / (double)(b))" in c_source


def test_global_shadowing_builtin_intrinsic_requires_decorated_helper():
    namespace = {"max": _plain_python_helper}
    exec("def use_shadowed(a):\n    return max(a)\n", namespace)
    use_shadowed = rumba.njit(namespace["use_shadowed"])

    with pytest.raises(
        RumbaUnsupportedError,
        match="helper calls require @rumba.njit-decorated functions",
    ):
        use_shadowed(1)


def test_math_module_intrinsics_execute_for_module_and_alias():
    @rumba.njit
    def use_math(a):
        return math.sqrt(a) + math.sin(a) + m.cos(a)

    assert use_math(4.0) == pytest.approx(math.sqrt(4.0) + math.sin(4.0) + math.cos(4.0))
    typed = use_math.inspect_typed_ast()
    c_source = use_math.inspect_c()

    assert typed["body"][0]["value"]["left"]["left"]["intrinsic"] == "math.sqrt"
    assert "sqrt(a)" in c_source
    assert "sin(a)" in c_source
    assert "cos(a)" in c_source


def test_unsupported_calls_still_fail_clearly():
    @rumba.njit
    def use_sum(a):
        return sum(a)

    @rumba.njit
    def use_gamma(a):
        return math.gamma(a)

    with pytest.raises(RumbaUnsupportedError, match="unsupported call to sum"):
        use_sum(1)
    with pytest.raises(RumbaUnsupportedError, match="attribute access"):
        use_gamma(1.0)


def test_undecorated_module_level_helper_call_raises():
    @rumba.njit
    def use_helper(a):
        return _plain_python_helper(a)

    with pytest.raises(
        RumbaUnsupportedError,
        match="helper calls require @rumba.njit-decorated functions",
    ):
        use_helper(1)


def test_decorated_helper_works_before_direct_compilation():
    assert _uncalled_decorated_helper.signatures == []

    @rumba.njit
    def use_helper(a):
        return _uncalled_decorated_helper(a)

    assert use_helper(3) == 6
    assert _uncalled_decorated_helper.signatures == []


def test_debug_output_includes_generated_helper_c(capfd):
    @rumba.njit(debug=True)
    def use_helper(a):
        return _uncalled_decorated_helper(a)

    assert use_helper(3) == 6

    captured = capfd.readouterr()
    assert "[rumba-debug] compile: typed function:" in captured.err
    assert "static int64_t rumba_helper_0" in captured.err


def test_explicit_helper_signature_match_works():
    @rumba.njit
    def use_helper(a):
        return _explicit_int_helper(a)

    assert use_helper(4) == 5


def test_explicit_helper_signature_mismatch_raises():
    @rumba.njit
    def use_helper(a):
        return _explicit_int_helper(a)

    with pytest.raises(RumbaUnsupportedError, match="does not match explicit helper signature"):
        use_helper(1.5)


def test_recursive_jitted_helper_call_remains_unsupported():
    @rumba.njit
    def use_helper(a):
        return _recursive_helper(a)

    with pytest.raises(RumbaUnsupportedError, match="recursive function calls are not supported"):
        use_helper(3)


def test_helper_with_default_args_remains_unsupported():
    @rumba.njit
    def use_helper(a):
        return _helper_with_default(a)

    with pytest.raises(RumbaUnsupportedError, match="default arguments are not supported"):
        use_helper(1)


def test_helper_with_varargs_remains_unsupported():
    @rumba.njit
    def use_helper(a):
        return _helper_with_varargs(a)

    with pytest.raises(RumbaUnsupportedError, match="varargs and kwargs are not supported"):
        use_helper(1)


def test_helper_with_keyword_only_args_remains_unsupported():
    @rumba.njit
    def use_helper(a):
        return _helper_with_keyword_only(a)

    with pytest.raises(RumbaUnsupportedError, match="keyword-only arguments are not supported"):
        use_helper(1)


def test_unsupported_list_argument_raises():
    @rumba.njit
    def first(x):
        return x[0]

    with pytest.raises(RumbaUnsupportedError, match="unsupported argument type"):
        first([1])
