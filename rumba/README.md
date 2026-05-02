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

Only scalar `int64`, `float64`, and `bool` arguments and scalar returns are
supported in this first slice. Rust owns the importable module, decorator API,
dispatcher, C emission, cache metadata, compiler selection, shared-library
compilation, and native invocation.
