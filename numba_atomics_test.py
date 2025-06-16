import numba.cpu
from numba import njit
import numpy as np


@njit
def atomic_operations():
    arr = np.array([10, 20, 30], dtype=np.uint64)

    # Atomic load/store
    val = numba.cpu.atomic.load(arr, 0, "acquire")
    numba.cpu.atomic.store(arr, 1, np.uint64(42), "release")

    # Atomic arithmetic (returns previous value)
    old = numba.cpu.atomic.add(arr, 2, np.uint64(5), "acq_rel")
    old = numba.cpu.atomic.sub(arr, 0, np.uint64(3))

    return val, old


atomic_operations()
