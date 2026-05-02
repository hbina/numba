"""Compare Python, Numba, and Rumba on the current scalar subset.

Run from this directory after installing Rumba:

    maturin develop
    python examples/compare_python_numba_rumba.py
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


def intrinsic_score_f64(values):
    a = values[0]
    b = values[1]
    span = max(a, b) - min(a, b)
    magnitude = math.sqrt(abs(a))
    wave = math.sin(b) + math.cos(a)
    reduction = np.sum(values) + np.max(values) - np.min(values)
    return magnitude + wave + reduction + len(values) + span


rumba_add_i64 = rumba.njit(add_i64)
rumba_distance_f64 = rumba.njit(distance_f64)
rumba_triangular_i64 = rumba.njit(triangular_i64)
rumba_weighted_sum_i64 = rumba.njit(weighted_sum_i64)
rumba_intrinsic_score_f64 = rumba.njit(intrinsic_score_f64)


def combined_functions(a, b, c):
    return (
        rumba_add_i64(a, b)
        + rumba_distance_f64(a, b)
        + rumba_triangular_i64(c)
        + rumba_weighted_sum_i64(a, b, c)
    )


CASES = (
    ("add_i64", add_i64, (11, 31), True),
    ("distance_f64", distance_f64, (10.5, 2.25), True),
    ("triangular_i64", triangular_i64, (12,), True),
    ("weighted_sum_i64", weighted_sum_i64, (3, 5, 7), True),
    (
        "intrinsic_score_f64",
        intrinsic_score_f64,
        (np.array([9.0, 2.5, 4.75], dtype=np.float64),),
        True,
    ),
    ("combined_functions", combined_functions, (3, 5, 7), False),
)


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


def main():
    print(f"Numba version: {numba.__version__}")
    print(f"Rumba version: {rumba.__version__}")

    for name, py_func, args, compare_numba in CASES:
        rumba_func = rumba.njit(py_func)

        py_value = py_func(*args)
        if compare_numba:
            numba_func = numba.njit(py_func)
            numba_value = numba_func(*args)
        else:
            numba_value = py_value
        rumba_value = rumba_func(*args)

        assert_same(name, py_value, numba_value, rumba_value)
        signature = ", ".join(str(typ) for typ in rumba_func.signatures[0])

        numba_note = (
            "" if compare_numba else "; Numba skipped for Rumba jitted helper calls"
        )
        print(
            f"{name}({', '.join(map(repr, args))}) -> {py_value!r} "
            f"[Rumba signature: ({signature}){numba_note}]"
        )
        if name == "intrinsic_score_f64":
            print("\nGenerated C for intrinsic_score_f64:")
            print(rumba_func.inspect_c())

    print("Python <=> Rumba comparisons passed; Numba compared where supported.")


if __name__ == "__main__":
    main()
