import dis
import numba


@numba.njit(cache=False)
def hello():
    result = 0
    xxx = range(10)
    for i in xxx:
        result += i
    return result


dis.dis(hello)

hello()

hello()
