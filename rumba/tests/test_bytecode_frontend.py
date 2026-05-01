import pytest

import rumba
from rumba import RumbaUnsupportedError

_COMPARISON_OPERATORS = ("<", "<=", "==", "!=", ">", ">=")


def _helper_add(a, b):
    return a + b


def _helper_weighted(a, b, c):
    return a + b * c


def test_inspect_bytecode_uses_rust_decoder_without_bookkeeping_opcodes():
    @rumba.njit
    def add(a, b):
        tmp = a + b
        return tmp

    bytecode = add.inspect_bytecode()
    opnames = [inst["opname"] for inst in bytecode]

    assert "LOAD_FAST" in opnames
    assert "BINARY_OP" in opnames
    assert "STORE_FAST" in opnames
    assert "RETURN_VALUE" in opnames
    assert "CACHE" not in opnames
    assert "RESUME" not in opnames


def test_decode_branch_to_rumba_if():
    @rumba.njit
    def choose(a, b):
        if a > b:
            return a - b
        return b - a

    summary = choose.inspect_rumba_ast()
    assert summary["body"][0] == "If"


@pytest.mark.parametrize("op", _COMPARISON_OPERATORS)
def test_decode_compare_op_argrepr(op):
    namespace = {}
    exec(f"def compare(a, b):\n    return a {op} b\n", namespace)
    compare = rumba.njit(namespace["compare"])

    bytecode = compare.inspect_bytecode()
    comparisons = [inst for inst in bytecode if inst["opname"] == "COMPARE_OP"]

    assert len(comparisons) == 1
    assert comparisons[0]["argrepr"] == op


def test_less_than_compare_op_does_not_decode_as_equals():
    namespace = {}
    exec("def compare(a, b):\n    return a < b\n", namespace)
    compare = rumba.njit(namespace["compare"])

    bytecode = compare.inspect_bytecode()
    comparisons = [inst for inst in bytecode if inst["opname"] == "COMPARE_OP"]

    assert comparisons[0]["argrepr"] == "<"
    assert comparisons[0]["argrepr"] != "=="


def test_decode_if_else_assignment_with_duplicated_tail_return():
    @rumba.njit
    def choose_flag(a):
        if a > 0:
            value = 1
        else:
            value = 2
        return value

    summary = choose_flag.inspect_rumba_ast()
    assert summary["body"] == ["If"]
    assert choose_flag(3) == 1
    assert choose_flag(-1) == 2


def test_decode_range_loop_to_rumba_for_range():
    @rumba.njit
    def total(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc

    summary = total.inspect_rumba_ast()
    assert summary["body"] == ["Assign", "For", "Return"]
    assert total(6) == 15


def test_decode_global_function_calls():
    @rumba.njit
    def combined(a, b, c):
        return _helper_add(a, b) + _helper_weighted(a, b, c)

    assert combined(3, 5, 7) == 46
    c_source = combined.inspect_c()
    assert "rumba_helper_0" in c_source
    assert "rumba_helper_1" in c_source


def test_promoted_float_return_survives_typing_pass():
    @rumba.njit
    def choose(a):
        if a > 0:
            return 1
        return 2.5

    assert choose(3.0) == pytest.approx(1.0)
    assert choose(-1.0) == pytest.approx(2.5)


def test_if_condition_must_be_bool():
    @rumba.njit
    def choose(a):
        if a:
            return 1
        return 0

    with pytest.raises(RumbaUnsupportedError, match="if condition must be boolean"):
        choose(1)


def test_use_before_assignment_still_raises():
    @rumba.njit
    def total(n):
        acc += n
        return acc

    with pytest.raises(RumbaUnsupportedError, match="used before assignment"):
        total(1)


def test_source_unavailable_function_compiles_from_bytecode():
    namespace = {}
    exec(
        "def generated(a, b):\n"
        "    value = a * b\n"
        "    return value + 1\n",
        namespace,
    )
    generated = rumba.njit(namespace["generated"])

    assert generated(3, 4) == 13


def test_unsupported_list_indexing_raises_during_frontend_parsing():
    @rumba.njit(signature=("int64",))
    def first(x):
        return x[0]

    with pytest.raises(RumbaUnsupportedError, match="indexing requires an array"):
        first(1)


def test_unsupported_call_other_than_range_raises():
    @rumba.njit
    def use_abs(x):
        return abs(x)

    with pytest.raises(RumbaUnsupportedError, match="unsupported call to abs"):
        use_abs(1)


def test_unsupported_closure_raises():
    value = 10

    @rumba.njit
    def add_value(x):
        return x + value

    with pytest.raises(RumbaUnsupportedError, match="closures are not supported"):
        add_value(1)


def test_unsupported_default_args_raises():
    @rumba.njit
    def with_default(x=1):
        return x

    with pytest.raises(RumbaUnsupportedError, match="default arguments are not supported"):
        with_default()


def test_unsupported_varargs_raises():
    @rumba.njit
    def with_varargs(*args):
        return 1

    with pytest.raises(RumbaUnsupportedError, match="varargs and kwargs are not supported"):
        with_varargs()


def test_unsupported_keyword_only_args_raises_on_frontend_inspection():
    @rumba.njit
    def keyword_only(*, x):
        return x

    with pytest.raises(RumbaUnsupportedError, match="keyword-only arguments are not supported"):
        keyword_only.inspect_rumba_ast()


def test_unsupported_comprehension_raises():
    @rumba.njit
    def comprehension(n):
        return sum(i for i in range(n))

    with pytest.raises(RumbaUnsupportedError):
        comprehension(3)


def test_unsupported_exception_handling_raises():
    @rumba.njit
    def catches(x):
        try:
            return x + 1
        except Exception:
            return 0

    with pytest.raises(RumbaUnsupportedError):
        catches(1)
