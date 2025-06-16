import numpy as np
import numba


@numba.njit(cache=False)
def numba_loop():
    a = 0

    i = 0
    while True:
        if i == 10:
            break
        a += i
        i += 1

    return a


xxx = numba_loop()
print(xxx)
