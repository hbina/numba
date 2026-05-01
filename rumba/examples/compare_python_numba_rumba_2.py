"""Dynamic Python, Numba, and Rumba conformance comparison.

This example generates a finite matrix of simple functions over scalar, array,
and structured-array inputs. Each case is compiled and run with Numba first.
If Numba supports the case, Rumba is required to compile and produce the same
result; otherwise the case is recorded as outside the current baseline.

Run from the repository root after installing Rumba:

    uv run python rumba/examples/compare_python_numba_rumba_2.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
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
GENERATED_GLOBALS = globals()


def add_i64(a, b):
    return a + b


def distance_f64(a, b):
    if a > b:
        return a - b
    return b - a


def triangular_i64(n):
    total = 0
    for i in range(n):
        total += i
    return total


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
    exec(source, GENERATED_GLOBALS)
    return Case(name, GENERATED_GLOBALS[name], args_factory, source)


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
            "    total += a0[i]['a'] + a0[i]['b']\n"
            "return total",
        ),
        _make_case(
            "generated_structured_weighted_array_struct_u64_f64_float64",
            lambda: (STRUCT_ARRAY_SPEC.factory(), 2.5),
            "total = 0.0\n"
            "for i in range(len(a0)):\n"
            "    total += a0[i]['a'] * a1 + a0[i]['b']\n"
            "return total",
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
        rumba_func = rumba.njit(debug=True)(case.py_func)
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


def _type_name(typ):
    return typ if isinstance(typ, str) else typ.name


def print_result(name, args, py_value, rumba_func):
    signature = ", ".join(_type_name(t) for t in rumba_func.signatures[0])
    print(
        f"{name}({', '.join(map(repr, args))}) -> {py_value!r} "
        f"[Rumba signature: ({signature})]"
    )
    print(rumba_func.inspect_c())


def run_nested_helper_case() -> CaseResult:
    # This intentionally remains separate from the Numba-gated matrix.
    # Rumba supports plain module-level helper calls here; Numba does not.
    try:
        py_value = combined_functions(*NESTED_ARGS)
        rumba_combined = rumba.njit(debug=True)(combined_functions)
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

    print_result("combined_functions", NESTED_ARGS, py_value, rumba_combined)
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


def print_summary(results: list[CaseResult]) -> None:
    groups = (
        ("passed", "passed"),
        ("numba_unsupported", "skipped because Numba did not support the case"),
        ("rumba_error", "Rumba unsupported/error"),
        ("rumba_mismatch", "result mismatches"),
        ("python_error", "Python errors"),
        ("numba_mismatch", "Numba result mismatches"),
    )

    print("\nSummary")
    print("=======")
    for status, label in groups:
        selected = [result for result in results if result.status == status]
        print(f"\n{label}: {len(selected)}")
        for result in selected:
            print(f"  - {result.case.name}: {result.detail}")


def main() -> int:
    print(f"Numba version: {numba.__version__}")
    print(f"Rumba version: {rumba.__version__}")

    results = [run_case(case) for case in generate_cases()]
    results.append(run_nested_helper_case())
    print_summary(results)

    failure_statuses = {
        "python_error",
        "numba_mismatch",
        "rumba_error",
        "rumba_mismatch",
    }
    failures = [result for result in results if result.status in failure_statuses]
    if failures:
        print(f"\nFAILED: {len(failures)} Numba-supported case(s) failed under Rumba.")
        return 1

    print("\nPython <=> Numba <=> Rumba comparisons passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
