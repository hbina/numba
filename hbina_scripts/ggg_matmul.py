import numba
import numpy as np


@numba.njit(debug=True, cache=False)
def hello(a1, b1):
    out = np.zeros(1, dtype=np.uint64)
    # c1 = np.dot(a1, b1, out=out])
    c1 = np.dot(out, out)
    # d1 = np.matmul(a1, b1)
    return c1


@numba.njit(debug=True, cache=False)
def center(X):
    means = np.array([np.float64(x.mean()) for x in X.T])
    ones = np.array([np.float64(1) for _ in X.T])
    return X - np.dot(ones, means)


# X = np.random.random((10, 10))
# res = center(X)

a = np.fromiter([9, 5, 7, 4], dtype=np.uint64)
b = np.fromiter([9, 5, 4, 3], dtype=np.uint64)
b = hello(a, b)
print(b)
