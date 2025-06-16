from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Generator

from numba import njit
import numpy as np


@contextmanager
def timer(
    f: Callable[[float], object] = lambda _: None,
) -> Generator[Callable[[], float], None, None]:
    _ = perf_counter()

    def t() -> float:
        return perf_counter() - _

    yield t
    f(t())
    del _


@njit
def test(x, y):
    return np.cov(x, y)


with timer(print):
    test(np.array([0.0]), np.array([0.0]))  # First call (JIT compilation expected)
# Example output: 3.549982499331236

with timer(print):
    test(np.array([0.0]), np.array([0.0]))  # Second call (should be fast)
# Example output: 1.730024814605713e-05

with timer(print):
    test(np.array([0.0, 0.0]), np.array([0.0, 0.0]))  # First shape change
# Example output: 0.02204340137541294  <-- Unexpectedly high execution time

with timer(print):
    test(
        np.array([0.0, 0.0]), np.array([0.0, 0.0])
    )  # Second call with same new shape (should be fast)
# Example output: 2.85004333428382874e-05


print(test.nopython_signatures)
# Output: [(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True)) -> array(float64, 2d, C)]
