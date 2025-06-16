import numba


@numba.njit(debug=True, cache=False)
def hello():
    yield hello()
    yield hello()


a = all(hello.py_func())
print(a)
b = all(hello())
print(b)
