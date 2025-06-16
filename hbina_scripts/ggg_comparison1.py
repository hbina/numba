import numpy as np
import numba


@numba.njit(cache=False)
def sample_generator():
    for i in range(10):
        yield i


@numba.njit(cache=False)
def numba_generator():
    a = 0
    for i in sample_generator():
        a += i
    return a


xxx = numba_generator()
print(xxx)
