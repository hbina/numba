# Rumba

Rumba is an experimental, independent implementation of a very small
Numba-like API. It does not import or depend on Numba at runtime.

Current development command:

```bash
cd rumba
maturin develop
python -m pytest
```

There are no Python implementation files for the `rumba` package. The only
Python files kept in this tree are public API tests.

The initial execution path is:

```text
Python function -> Rust/PyO3 frontend -> Rumba AST -> typed Rumba AST
  -> generated C -> shared library -> Rust native invocation
```

Generated Rumba code is intended to be allocation-free. Rust/PyO3 validates
Python values and prepares any argument or return storage before entering the
compiled native function; generated C must not allocate Python objects, NumPy
arrays, heap buffers, or runtime containers.

## Return Values And Output Buffers

Rumba functions only support returning scalar values such as `int64`, `float64`,
and `bool`. Returning a NumPy array is unsupported and raises
`RumbaUnsupportedError`.

## No Implicit Scalar Conversions

Rumba intentionally rejects implicit scalar conversions. Scalar arithmetic,
comparisons, branch-local variable merging, and return merging require exact
matching scalar types. Mixed expressions such as `1 + 1.0`, `1 < 2.0`, `not 1`,
or returning `int64` on one path and `float64` on another raise
`RumbaUnsupportedError`.

Use explicit Python casts when conversion is intended:

```python
@rumba.njit
def score(n, weight):
    return float(n) + weight
```

The supported scalar casts are `int(...)`, `float(...)`, and `bool(...)` for
all combinations of `int64`, `float64`, and `bool`. True division is the one
operator-defined widening rule: `int64 / int64` returns `float64`, matching
Python `/`. Floor division and modulo require identical numeric operand types
and return that type.

Scalar intrinsics are strict too. `min` and `max` require at least two non-bool
scalar arguments of the same type, `abs` supports only `int64` and `float64`,
and `math.*` intrinsics require `float64` arguments.

If a function needs to produce array output, the caller must allocate the NumPy
array and pass it as an output argument. The compiled function can then mutate
that caller-owned buffer in place:

```python
@rumba.njit
def fill(values, out):
    for i in range(len(values)):
        out[i] = values[i] * 2
    return 0
```

This follows directly from Rumba's allocation-free execution model: generated
code cannot allocate return arrays, so the caller is responsible for providing
any array storage.

Only scalar `int64`, `float64`, and `bool` values and supported 1D NumPy arrays
are accepted by the current execution slice. Rust owns the importable module,
decorator API, dispatcher, C emission, cache metadata, compiler selection,
shared-library compilation, and native invocation.
