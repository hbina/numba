"""Compare Python, Numba, and Rumba including nested (JIT-calls-JIT) function calls
and numpy.ndarray arguments.

Run from the rumba directory after installing Rumba:

    maturin develop
    python examples/compare_python_numba_rumba_2.py
"""

from __future__ import annotations

import math

import numba
import numpy as np
import rumba


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


def sum_array_i64(a):
    total = 0
    for i in range(len(a)):
        total += a[i]
    return total


def sum_array_f64(a):
    total = 0.0
    for i in range(len(a)):
        total += a[i]
    return total


def offset_sum_i64(a, offset):
    total = 0
    for i in range(len(a)):
        total += a[i] + offset
    return total


def combined_functions(a, b, c):
    return (
        add_i64(a, b)
        + distance_f64(a, b)
        + triangular_i64(c)
        + weighted_sum_i64(a, b, c)
    )


SIMPLE_CASES = (
    ("add_i64", add_i64, (11, 31)),
    ("distance_f64", distance_f64, (10.5, 2.25)),
    ("triangular_i64", triangular_i64, (12,)),
    ("weighted_sum_i64", weighted_sum_i64, (3, 5, 7)),
)

_INT_ARR = np.arange(6, dtype=np.int64)          # [0, 1, 2, 3, 4, 5]
_FLOAT_ARR = np.array([1.25, 2.5, 3.75], dtype=np.float64)

ARRAY_CASES = (
    ("sum_array_i64", sum_array_i64, (_INT_ARR,)),
    ("sum_array_f64", sum_array_f64, (_FLOAT_ARR,)),
    ("offset_sum_i64", offset_sum_i64, (_INT_ARR, 10)),
)

NESTED_ARGS = (3, 5, 7)


def assert_same(name, py_value, numba_value, rumba_value):
    if isinstance(py_value, float):
        ok = math.isclose(py_value, numba_value) and math.isclose(py_value, rumba_value)
    else:
        ok = py_value == numba_value == rumba_value

    if not ok:
        raise AssertionError(
            f"{name} mismatch: Python={py_value!r}, "
            f"Numba={numba_value!r}, Rumba={rumba_value!r}"
        )


def _type_name(typ):
    return typ if isinstance(typ, str) else typ.name


def print_result(name, args, py_value, rumba_func):
    signature = ", ".join(_type_name(t) for t in rumba_func.signatures[0])
    print(
        f"{name}({', '.join(map(repr, args))}) -> {py_value!r} "
        f"[Rumba signature: ({signature})]"
    )
    print(rumba_func.inspect_c())



def main():
    print(f"Numba version: {numba.__version__}")
    print(f"Rumba version: {rumba.__version__}")

    # Step 1: compile and verify simple (non-nested) functions.
    for name, py_func, args in SIMPLE_CASES:
        numba_func = numba.njit(py_func)
        rumba_func = rumba.njit(py_func)

        py_value = py_func(*args)
        numba_value = numba_func(*args)
        rumba_value = rumba_func(*args)

        assert_same(name, py_value, numba_value, rumba_value)
        print_result(name, args, py_value, rumba_func)

    # Step 2: compile and verify array (numpy.ndarray) functions.
    for name, py_func, args in ARRAY_CASES:
        numba_func = numba.njit(py_func)
        rumba_func = rumba.njit(py_func)

        py_value = py_func(*args)
        numba_value = numba_func(*args)
        rumba_value = rumba_func(*args)

        assert_same(name, py_value, numba_value, rumba_value)
        print_result(name, args, py_value, rumba_func)

    # Step 3: test nested calls — combined_functions calls plain module-level
    # helpers. Numba cannot resolve plain Python globals from njit context, so
    # only Python <=> Rumba is compared here.
    rumba_combined = rumba.njit(combined_functions)

    py_value = combined_functions(*NESTED_ARGS)
    rumba_value = rumba_combined(*NESTED_ARGS)

    if py_value != rumba_value:
        raise AssertionError(
            f"combined_functions mismatch: Python={py_value!r}, Rumba={rumba_value!r}"
        )
    print_result("combined_functions", NESTED_ARGS, py_value, rumba_combined)

    print("Python <=> Numba <=> Rumba comparisons passed.")


if __name__ == "__main__":
    main()
