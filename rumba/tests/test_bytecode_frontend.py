import pytest

import rumba
from rumba import RumbaUnsupportedError

_COMPARISON_OPERATORS = ("<", "<=", "==", "!=", ">", ">=")


@rumba.njit
def _helper_add(a, b):
    return a + b


@rumba.njit
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


def test_nested_if_inside_if_executes():
    @rumba.njit
    def choose(a, b, c):
        if a > b:
            if b > c:
                return a - c
            return a - b
        return b - a

    assert choose(9, 5, 2) == 7
    assert choose(9, 5, 7) == 4
    assert choose(2, 5, 7) == 3


def test_nested_if_inside_else_executes():
    @rumba.njit
    def choose(a, b, c):
        if a > b:
            return a - b
        else:
            if b > c:
                return b - c
            return c - b

    assert choose(9, 5, 2) == 4
    assert choose(2, 5, 3) == 2
    assert choose(2, 5, 8) == 3


def test_if_elif_else_executes_and_normalizes_to_nested_if():
    @rumba.njit
    def classify(a):
        if a < 0:
            return -1
        elif a == 0:
            return 0
        else:
            return 1

    assert classify(-3) == -1
    assert classify(0) == 0
    assert classify(8) == 1

    typed = classify.inspect_typed_ast()
    outer = typed["body"][0]
    assert outer["kind"] == "If"
    assert outer["orelse"][0]["kind"] == "If"


def test_multiple_elif_branches_execute():
    @rumba.njit
    def classify(a):
        if a < 0:
            return -1
        elif a == 0:
            return 0
        elif a == 1:
            return 10
        return 2

    assert classify(-3) == -1
    assert classify(0) == 0
    assert classify(1) == 10
    assert classify(4) == 2


def test_branch_assignment_followed_by_shared_return():
    @rumba.njit
    def choose_flag(a):
        if a > 0:
            value = 1
        else:
            value = 2
        return value + 10

    assert choose_flag(3) == 11
    assert choose_flag(-1) == 12


def test_branch_only_local_use_after_branch_raises():
    @rumba.njit
    def choose_flag(a):
        if a > 0:
            value = 1
        return value

    with pytest.raises(RumbaUnsupportedError, match="assigned in only one branch"):
        choose_flag(3)


def test_incompatible_branch_assignment_types_raise():
    @rumba.njit
    def choose_flag(a):
        if a > 0:
            value = 1
        else:
            value = 2.5
        return value

    with pytest.raises(RumbaUnsupportedError, match="incompatible branch assignment types"):
        choose_flag(3)


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


def test_range_start_stop_loop_executes():
    @rumba.njit
    def total(start, stop):
        acc = 0
        for i in range(start, stop):
            acc += i
        return acc

    assert total(2, 6) == 14


def test_range_start_stop_step_loop_executes():
    @rumba.njit
    def total(start, stop, step):
        acc = 0
        for i in range(start, stop, step):
            acc += i
        return acc

    assert total(2, 10, 3) == 15


def test_range_constant_negative_step_loop_executes():
    @rumba.njit
    def total(n):
        acc = 0
        for i in range(n, 0, -2):
            acc += i
        return acc

    assert total(7) == 16


def test_range_dynamic_positive_and_negative_step_loop_executes():
    @rumba.njit
    def total(start, stop, step):
        acc = 0
        for i in range(start, stop, step):
            acc += i
        return acc

    assert total(1, 8, 2) == 16
    assert total(8, 1, -3) == 15

    c_source = total.inspect_c()
    assert "? i < stop : i > stop" in c_source


def test_decode_while_loop_to_rumba_while():
    @rumba.njit
    def total(n):
        acc = 0
        while n > 0:
            acc += n
            n -= 1
        return acc

    summary = total.inspect_rumba_ast()
    assert summary["body"] == ["Assign", "While", "Return"]
    assert total(5) == 15
    assert "while (" in total.inspect_c()


def test_float_accumulator_while_loop_executes():
    @rumba.njit
    def total(n):
        acc = 0.5
        while n > 0:
            acc += 1.25
            n -= 1
        return acc

    assert total(4) == pytest.approx(5.5)


def test_while_with_nested_if_executes():
    @rumba.njit
    def total(n):
        acc = 0
        while n > 0:
            if n > 2:
                acc += n
            else:
                acc += 1
            n -= 1
        return acc

    assert total(4) == 9


def test_while_with_nested_for_range_executes():
    @rumba.njit
    def total(n):
        acc = 0
        while n > 0:
            for i in range(n):
                acc += i
            n -= 1
        return acc

    assert total(4) == 10


def test_while_with_early_return_executes():
    @rumba.njit
    def find_total(n):
        acc = 0
        while n > 0:
            if n == 2:
                return acc
            acc += n
            n -= 1
        return acc

    assert find_total(5) == 12
    assert find_total(1) == 1


def test_while_condition_must_be_bool():
    @rumba.njit
    def total(n):
        acc = 0
        while n:
            acc += n
            n -= 1
        return acc

    with pytest.raises(RumbaUnsupportedError, match="while condition must be boolean"):
        total(3)


def test_break_inside_while_remains_unsupported():
    @rumba.njit
    def total(n):
        acc = 0
        while n > 0:
            break
        return acc

    with pytest.raises(RumbaUnsupportedError, match="break is not supported"):
        total(3)


def test_continue_inside_while_remains_unsupported():
    @rumba.njit
    def total(n):
        acc = 0
        while n > 0:
            n -= 1
            if n == 2:
                continue
            acc += n
        return acc

    with pytest.raises(RumbaUnsupportedError, match="continue is not supported"):
        total(4)


def test_source_unavailable_continue_bytecode_shape_is_rejected():
    namespace = {}
    exec(
        "def generated(n):\n"
        "    acc = 0\n"
        "    while n > 0:\n"
        "        n -= 1\n"
        "        if n == 2:\n"
        "            continue\n"
        "        acc += n\n"
        "    return acc\n",
        namespace,
    )
    generated = rumba.njit(namespace["generated"])

    with pytest.raises(RumbaUnsupportedError, match="unsupported .* control flow"):
        generated(4)


def test_range_rejects_float_stop():
    @rumba.njit
    def total(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc

    with pytest.raises(RumbaUnsupportedError, match="range stop must be an int64 scalar"):
        total(4.5)


def test_range_rejects_float_step():
    @rumba.njit
    def total(step):
        acc = 0
        for i in range(0, 5, step):
            acc += i
        return acc

    with pytest.raises(RumbaUnsupportedError, match="range step must be an int64 scalar"):
        total(1.5)


def test_range_rejects_constant_zero_step():
    @rumba.njit
    def total(n):
        acc = 0
        for i in range(0, n, 0):
            acc += i
        return acc

    with pytest.raises(RumbaUnsupportedError, match="range step cannot be zero"):
        total(5)


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
    def use_sum(x):
        return sum(x)

    with pytest.raises(RumbaUnsupportedError, match="unsupported call to sum"):
        use_sum(1)


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
