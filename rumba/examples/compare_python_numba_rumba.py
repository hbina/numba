"""Compare Python, Numba, and Rumba on the current supported subset.

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


def while_countdown_i64(n):
    total = 0
    while n > 0:
        total += n
        n -= 1
    return total


def stepped_total_i64(start, stop, step):
    total = 0
    for i in range(start, stop, step):
        total += i
    return total


def weighted_sum_i64(a, b, c):
    return a + b * c


def nested_branch_i64(a, b, c):
    if a > b:
        if b > c:
            return a - c
        return a - b
    return b - a


def classify_i64(a):
    if a < 0:
        return -1
    elif a == 0:
        return 0
    else:
        return 1


def combined_control_flow_i64(a, b, c, step):
    if a > b:
        start = b
        stop = a
    elif a == b:
        return c
    else:
        start = a
        stop = b

    total = 0
    for i in range(start, stop, step):
        if i > c:
            total += i - c
        else:
            total += c - i
    return total


def intrinsic_score_f64(values):
    a = values[0]
    b = values[1]
    span = max(a, b) - min(a, b)
    magnitude = math.sqrt(abs(a))
    wave = math.sin(b) + math.cos(a)
    reduction = np.sum(values) + np.max(values) - np.min(values)
    return magnitude + wave + reduction + float(len(values)) + span


def structured_field_sum(records):
    total = 0.0
    for i in range(len(records)):
        total += float(records[i]["count"]) + records[i]["weight"]
    return total


def structured_weighted_score(records, scale):
    total = 0.0
    for i in range(len(records)):
        total += float(records[i]["count"]) * scale + records[i]["weight"]
    return total


def structured_update_weight(records, replacement):
    records[1]["weight"] = replacement
    return records[1]["weight"]


def while_mutate_array(values, n):
    i = 0
    while i < n:
        values[i] = i + 10
        i += 1
    return values[n - 1]


def while_structured_update(records, n):
    i = 0
    total = 0.0
    while i < n:
        records[i]["weight"] = float(records[i]["count"]) * 2.0
        total += records[i]["weight"]
        i += 1
    return total


rumba_add_i64 = rumba.njit(add_i64)
rumba_distance_f64 = rumba.njit(distance_f64)
rumba_triangular_i64 = rumba.njit(triangular_i64)
rumba_while_countdown_i64 = rumba.njit(while_countdown_i64)
rumba_stepped_total_i64 = rumba.njit(stepped_total_i64)
rumba_weighted_sum_i64 = rumba.njit(weighted_sum_i64)
rumba_nested_branch_i64 = rumba.njit(nested_branch_i64)
rumba_classify_i64 = rumba.njit(classify_i64)
rumba_combined_control_flow_i64 = rumba.njit(combined_control_flow_i64)
rumba_intrinsic_score_f64 = rumba.njit(intrinsic_score_f64)
rumba_structured_field_sum = rumba.njit(structured_field_sum)
rumba_structured_weighted_score = rumba.njit(structured_weighted_score)
rumba_structured_update_weight = rumba.njit(structured_update_weight)
rumba_while_mutate_array = rumba.njit(while_mutate_array)
rumba_while_structured_update = rumba.njit(while_structured_update)


def combined_functions(a, b, c):
    return (
        rumba_add_i64(a, b)
        + rumba_distance_f64(a, b)
        + rumba_triangular_i64(c)
        + rumba_stepped_total_i64(a, c * 2, 2)
        + rumba_weighted_sum_i64(a, b, c)
        + rumba_nested_branch_i64(a, b, c)
        + rumba_classify_i64(c)
    )


CASES = (
    ("add_i64", add_i64, (11, 31), True),
    ("distance_f64", distance_f64, (10.5, 2.25), True),
    ("triangular_i64", triangular_i64, (12,), True),
    ("while_countdown_i64", while_countdown_i64, (12,), True),
    ("stepped_total_i64", stepped_total_i64, (2, 12, 3), True),
    ("weighted_sum_i64", weighted_sum_i64, (3, 5, 7), True),
    ("nested_branch_i64", nested_branch_i64, (9, 5, 2), True),
    ("classify_i64", classify_i64, (0,), True),
    ("combined_control_flow_i64", combined_control_flow_i64, (3, 11, 5, 2), True),
    (
        "intrinsic_score_f64",
        intrinsic_score_f64,
        (np.array([9.0, 2.5, 4.75], dtype=np.float64),),
        True,
    ),
    (
        "structured_field_sum",
        structured_field_sum,
        (
            np.array(
                [(1, 1.25), (2, 2.5), (3, 3.75)],
                dtype=np.dtype([("count", np.uint64), ("weight", np.float64)]),
            ),
        ),
        True,
    ),
    (
        "structured_weighted_score",
        structured_weighted_score,
        (
            np.array(
                [(1, 1.25), (2, 2.5), (3, 3.75)],
                dtype=np.dtype([("count", np.uint64), ("weight", np.float64)]),
            ),
            2.5,
        ),
        True,
    ),
    (
        "structured_update_weight",
        structured_update_weight,
        (
            np.array(
                [(1, 1.25), (2, 2.5), (3, 3.75)],
                dtype=np.dtype([("count", np.uint64), ("weight", np.float64)]),
            ),
            9.5,
        ),
        True,
    ),
    (
        "while_mutate_array",
        while_mutate_array,
        (np.zeros(5, dtype=np.int64), 3),
        True,
    ),
    (
        "while_structured_update",
        while_structured_update,
        (
            np.array(
                [(1, 0.0), (2, 0.0), (3, 0.0)],
                dtype=np.dtype([("count", np.uint64), ("weight", np.float64)]),
            ),
            3,
        ),
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
        print(f"\nGenerated C for {name}:")
        print(rumba_func.inspect_c())

    print("Python <=> Rumba comparisons passed; Numba compared where supported.")


if __name__ == "__main__":
    main()
