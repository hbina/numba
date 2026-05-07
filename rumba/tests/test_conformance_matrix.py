from dataclasses import dataclass
import math
from typing import Callable

import numba
import numpy as np
import rumba


@dataclass(frozen=True)
class TypeSpec:
    name: str
    group: str
    factory: Callable[[], object]


@dataclass(frozen=True)
class Case:
    name: str
    py_func: Callable[..., object]
    args_factory: Callable[[], tuple[object, ...]]
    source: str


@dataclass(frozen=True)
class CaseResult:
    case: Case
    status: str
    detail: str


FAILURE_STATUSES = frozenset(
    {
        "python_error",
        "numba_mismatch",
        "rumba_error",
        "rumba_mismatch",
    }
)

EXPECTED_RUMBA_FAILURES = frozenset(
    {
        "generated_binary_add_int64_float64",
        "generated_binary_mul_int64_float64",
        "generated_compare_less_int64_float64",
        "generated_branch_distance_int64_float64",
        "generated_binary_add_int64_bool",
        "generated_binary_mul_int64_bool",
        "generated_compare_less_int64_bool",
        "generated_branch_distance_int64_bool",
        "generated_binary_add_float64_int64",
        "generated_binary_mul_float64_int64",
        "generated_compare_less_float64_int64",
        "generated_branch_distance_float64_int64",
        "generated_binary_add_float64_bool",
        "generated_binary_mul_float64_bool",
        "generated_compare_less_float64_bool",
        "generated_branch_distance_float64_bool",
        "generated_binary_add_bool_float64",
        "generated_binary_mul_bool_float64",
        "generated_compare_less_bool_float64",
        "generated_branch_distance_bool_float64",
        "generated_binary_add_bool_int64",
        "generated_binary_mul_bool_int64",
        "generated_compare_less_bool_int64",
        "generated_branch_distance_bool_int64",
        "generated_binary_add_bool_bool",
        "generated_binary_mul_bool_bool",
        "generated_compare_less_bool_bool",
        "generated_branch_distance_bool_bool",
        "generated_array_plus_scalar_first_array_int64_float64",
        "generated_scalar_plus_array_first_array_int64_float64",
        "generated_array_plus_scalar_first_array_float64_int64",
        "generated_scalar_plus_array_first_array_float64_int64",
        "generated_array_plus_array_first_array_int64_array_float64",
        "generated_array_plus_array_first_array_float64_array_int64",
    }
)

SCALAR_SPECS = (
    TypeSpec("int64", "scalar", lambda: 7),
    TypeSpec("float64", "scalar", lambda: 2.5),
    TypeSpec("bool", "scalar", lambda: True),
)

NUMERIC_SCALAR_SPECS = (
    TypeSpec("int64", "scalar", lambda: 7),
    TypeSpec("float64", "scalar", lambda: 2.5),
)

ARRAY_SPECS = (
    TypeSpec("array_int64", "array", lambda: np.arange(1, 7, dtype=np.int64)),
    TypeSpec(
        "array_float64",
        "array",
        lambda: np.array([1.25, 2.5, 3.75, 5.0], dtype=np.float64),
    ),
)

STRUCT_DTYPE = np.dtype([("a", np.uint64), ("b", np.float64)])
STRUCT_ARRAY_SPEC = TypeSpec(
    "array_struct_u64_f64",
    "structured_array",
    lambda: np.array([(1, 1.25), (2, 2.5), (3, 3.75)], dtype=STRUCT_DTYPE),
)

NESTED_ARGS = (3, 5, 7)


@rumba.njit
def add_i64(a, b):
    return a + b


@rumba.njit
def distance_f64(a, b):
    if a > b:
        return a - b
    return b - a


@rumba.njit
def triangular_i64(n):
    total = 0
    for i in range(n):
        total += i
    return total


@rumba.njit
def weighted_sum_i64(a, b, c):
    return a + b * c


def combined_functions(a, b, c):
    return (
        add_i64(a, b)
        + distance_f64(a, b)
        + triangular_i64(c)
        + weighted_sum_i64(a, b, c)
    )


def _make_case(
    name: str,
    args_factory: Callable[[], tuple[object, ...]],
    body: str,
) -> Case:
    source = f"def {name}({', '.join(f'a{i}' for i in range(_arg_count(args_factory)))}):\n"
    source += "\n".join(f"    {line}" if line else "" for line in body.splitlines())
    source += "\n"
    exec(source, globals())
    return Case(name, globals()[name], args_factory, source)


def _arg_count(args_factory: Callable[[], tuple[object, ...]]) -> int:
    return len(args_factory())


def _binary_factory(left: TypeSpec, right: TypeSpec) -> Callable[[], tuple[object, ...]]:
    return lambda: (left.factory(), right.factory())


def _array_scalar_factory(
    array_spec: TypeSpec, scalar_spec: TypeSpec
) -> Callable[[], tuple[object, ...]]:
    return lambda: (array_spec.factory(), scalar_spec.factory())


def _scalar_array_factory(
    scalar_spec: TypeSpec, array_spec: TypeSpec
) -> Callable[[], tuple[object, ...]]:
    return lambda: (scalar_spec.factory(), array_spec.factory())


def generate_scalar_cases() -> list[Case]:
    cases: list[Case] = []

    for spec in SCALAR_SPECS:
        name = f"generated_unary_identity_{spec.name}"
        cases.append(_make_case(name, lambda spec=spec: (spec.factory(),), "return a0"))

    for left in SCALAR_SPECS:
        for right in SCALAR_SPECS:
            suffix = f"{left.name}_{right.name}"
            factory = _binary_factory(left, right)
            cases.append(
                _make_case(f"generated_binary_add_{suffix}", factory, "return a0 + a1")
            )
            cases.append(
                _make_case(f"generated_binary_mul_{suffix}", factory, "return a0 * a1")
            )
            cases.append(
                _make_case(
                    f"generated_compare_less_{suffix}", factory, "return a0 < a1"
                )
            )
            cases.append(
                _make_case(
                    f"generated_branch_distance_{suffix}",
                    factory,
                    "if a0 > a1:\n"
                    "    return a0 - a1\n"
                    "return a1 - a0",
                )
            )

    cases.append(
        _make_case(
            "generated_while_countdown_int64",
            lambda: (7,),
            "total = 0\n"
            "while a0 > 0:\n"
            "    total += a0\n"
            "    a0 -= 1\n"
            "return total",
        )
    )
    cases.append(
        _make_case(
            "generated_while_float_accumulator_int64",
            lambda: (4,),
            "total = 0.5\n"
            "while a0 > 0:\n"
            "    total += 1.25\n"
            "    a0 -= 1\n"
            "return total",
        )
    )
    cases.append(
        _make_case(
            "generated_while_break_int64",
            lambda: (4,),
            "total = 0\n"
            "while a0 > 0:\n"
            "    if a0 == 2:\n"
            "        break\n"
            "    total += a0\n"
            "    a0 -= 1\n"
            "total += 10\n"
            "return total",
        )
    )
    cases.append(
        _make_case(
            "generated_while_continue_int64",
            lambda: (4,),
            "total = 0\n"
            "while a0 > 0:\n"
            "    a0 -= 1\n"
            "    if a0 == 2:\n"
            "        continue\n"
            "    total += a0\n"
            "return total",
        )
    )
    cases.append(
        _make_case(
            "generated_for_range_break_int64",
            lambda: (7,),
            "total = 0\n"
            "for i in range(a0):\n"
            "    if i == 4:\n"
            "        break\n"
            "    total += i\n"
            "total += 10\n"
            "return total",
        )
    )
    cases.append(
        _make_case(
            "generated_for_range_continue_int64",
            lambda: (7,),
            "total = 0\n"
            "for i in range(a0):\n"
            "    if i == 4:\n"
            "        continue\n"
            "    total += i\n"
            "return total",
        )
    )
    cases.append(
        _make_case(
            "generated_explicit_cast_int_float64",
            lambda: (1.5,),
            "return int(a0) + 1",
        )
    )
    cases.append(
        _make_case(
            "generated_explicit_cast_float_int64",
            lambda: (1,),
            "return float(a0) + 2.5",
        )
    )
    cases.append(
        _make_case(
            "generated_explicit_cast_bool_int64",
            lambda: (0,),
            "return bool(a0)",
        )
    )

    return cases


def generate_array_cases() -> list[Case]:
    cases: list[Case] = []

    for array_spec in ARRAY_SPECS:
        suffix = array_spec.name
        cases.append(
            _make_case(
                f"generated_array_len_{suffix}",
                lambda array_spec=array_spec: (array_spec.factory(),),
                "return len(a0)",
            )
        )
        cases.append(
            _make_case(
                f"generated_array_first_{suffix}",
                lambda array_spec=array_spec: (array_spec.factory(),),
                "return a0[0]",
            )
        )
        cases.append(
            _make_case(
                f"generated_array_sum_{suffix}",
                lambda array_spec=array_spec: (array_spec.factory(),),
                _array_sum_body(array_spec),
            )
        )

        for scalar_spec in NUMERIC_SCALAR_SPECS:
            scalar_suffix = f"{array_spec.name}_{scalar_spec.name}"
            cases.append(
                _make_case(
                    f"generated_array_plus_scalar_first_{scalar_suffix}",
                    _array_scalar_factory(array_spec, scalar_spec),
                    "return a0[0] + a1",
                )
            )
            cases.append(
                _make_case(
                    f"generated_scalar_plus_array_first_{scalar_suffix}",
                    _scalar_array_factory(scalar_spec, array_spec),
                    "return a0 + a1[0]",
                )
            )
            cases.append(
                _make_case(
                    f"generated_array_mutate_{scalar_suffix}",
                    _array_scalar_factory(array_spec, scalar_spec),
                    "a0[1] = a1\n"
                    "return a0[1]",
                )
            )

    for left in ARRAY_SPECS:
        for right in ARRAY_SPECS:
            suffix = f"{left.name}_{right.name}"
            cases.append(
                _make_case(
                    f"generated_array_plus_array_first_{suffix}",
                    lambda left=left, right=right: (left.factory(), right.factory()),
                    "return a0[0] + a1[0]",
                )
            )

    return cases


def _array_sum_body(array_spec: TypeSpec) -> str:
    zero = "0.0" if array_spec.name.endswith("float64") else "0"
    return (
        f"total = {zero}\n"
        "for i in range(len(a0)):\n"
        "    total += a0[i]\n"
        "return total"
    )


def generate_structured_cases() -> list[Case]:
    return [
        _make_case(
            "generated_structured_field_sum_array_struct_u64_f64",
            lambda: (STRUCT_ARRAY_SPEC.factory(),),
            "total = 0.0\n"
            "for i in range(len(a0)):\n"
            "    total += float(a0[i]['a']) + a0[i]['b']\n"
            "return total",
        ),
        _make_case(
            "generated_structured_weighted_array_struct_u64_f64_float64",
            lambda: (STRUCT_ARRAY_SPEC.factory(), 2.5),
            "total = 0.0\n"
            "for i in range(len(a0)):\n"
            "    total += float(a0[i]['a']) * a1 + a0[i]['b']\n"
            "return total",
        ),
        _make_case(
            "generated_structured_mutate_array_struct_u64_f64_float64",
            lambda: (STRUCT_ARRAY_SPEC.factory(), 9.5),
            "a0[1]['b'] = a1\n"
            "return a0[1]['b']",
        ),
    ]


def generate_cases() -> list[Case]:
    return generate_scalar_cases() + generate_array_cases() + generate_structured_cases()


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()

    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def _short_error(exc: BaseException) -> str:
    first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return f"{exc.__class__.__name__}: {first_line}"


def run_case(case: Case) -> CaseResult:
    try:
        py_value = case.py_func(*case.args_factory())
    except Exception as exc:  # noqa: BLE001 - this is a conformance report.
        return CaseResult(case, "python_error", _short_error(exc))

    try:
        numba_func = numba.njit(case.py_func)
        numba_value = numba_func(*case.args_factory())
    except Exception as exc:  # noqa: BLE001 - Numba determines the baseline.
        return CaseResult(case, "numba_unsupported", _short_error(exc))

    if not _same_value(py_value, numba_value):
        return CaseResult(
            case,
            "numba_mismatch",
            f"Python={py_value!r}, Numba={numba_value!r}",
        )

    try:
        rumba_func = rumba.njit(case.py_func)
        rumba_value = rumba_func(*case.args_factory())
    except Exception as exc:  # noqa: BLE001 - report all Rumba gaps.
        return CaseResult(case, "rumba_error", _short_error(exc))

    if not _same_value(py_value, rumba_value):
        return CaseResult(
            case,
            "rumba_mismatch",
            f"Python={py_value!r}, Numba={numba_value!r}, Rumba={rumba_value!r}",
        )

    return CaseResult(case, "passed", repr(py_value))


def run_nested_helper_case() -> CaseResult:
    try:
        py_value = combined_functions(*NESTED_ARGS)
        rumba_combined = rumba.njit(combined_functions)
        rumba_value = rumba_combined(*NESTED_ARGS)
    except Exception as exc:  # noqa: BLE001 - keep reporting style consistent.
        return CaseResult(
            Case(
                "combined_functions",
                combined_functions,
                lambda: NESTED_ARGS,
                "manual nested helper case",
            ),
            "rumba_error",
            _short_error(exc),
        )

    if py_value != rumba_value:
        return CaseResult(
            Case(
                "combined_functions",
                combined_functions,
                lambda: NESTED_ARGS,
                "manual nested helper case",
            ),
            "rumba_mismatch",
            f"Python={py_value!r}, Rumba={rumba_value!r}",
        )

    return CaseResult(
        Case(
            "combined_functions",
            combined_functions,
            lambda: NESTED_ARGS,
            "manual nested helper case",
        ),
        "passed",
        repr(py_value),
    )


def run_conformance_matrix() -> list[CaseResult]:
    results = [run_case(case) for case in generate_cases()]
    results.append(run_nested_helper_case())
    return results


def failure_results(results: list[CaseResult]) -> list[CaseResult]:
    return [result for result in results if result.status in FAILURE_STATUSES]


def unexpected_failure_results(results: list[CaseResult]) -> list[CaseResult]:
    return [
        result
        for result in failure_results(results)
        if result.case.name not in EXPECTED_RUMBA_FAILURES
    ]


def format_failures(failures: list[CaseResult]) -> str:
    lines = [f"{len(failures)} Numba-supported case(s) failed under Rumba:"]
    for result in failures:
        lines.append(f"  - {result.case.name}: {result.status}: {result.detail}")
    return "\n".join(lines)


def test_python_numba_rumba_conformance_matrix():
    results = run_conformance_matrix()
    failures = unexpected_failure_results(results)

    assert not failures, format_failures(failures)
