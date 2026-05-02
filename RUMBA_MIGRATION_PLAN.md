# Comprehensive Rumba Migration Plan

## Summary

`rumba` is a new top-level package whose purpose is to reimplement a focused,
useful subset of Numba in Rust. The goal is not full Numba feature parity.
Rumba should deliberately support a small Python syntax surface well, with
clear unsupported-operation diagnostics outside that surface. It must not
import, call, or depend on `numba`; Numba is only a reference for bytecode
handling, supported semantics, NumPy overload behavior, diagnostics, and
compatibility tests for features Rumba explicitly chooses to support.

The long-term ownership model is Rust-first:

- Rust owns the public extension module through PyO3.
- Rust owns dispatcher state, signature specialization, bytecode decoding,
  Rumba AST construction, type inference, lowering, code generation, cache
  metadata, compiler invocation, diagnostics, and runtime ABI handling.
- There are no Python implementation files under `rumba/python/rumba`.
- Python remains only the consumer language and the language used by public API
  tests. Optional examples should live in documentation snippets rather than
  checked-in Python implementation files.

The target compiler pipeline is:

```text
Python function object
  -> Rust/PyO3 reads CPython code object and metadata
  -> Rust CPython bytecode decoder
  -> Rust-owned Rumba AST
  -> Rust-owned typed Rumba AST
  -> Rust generated C source
  -> Rust compiler driver builds shared library
  -> Rust/PyO3 dispatcher invokes compiled artifact
```

## Current Implementation

- Added `rumba/` as an independent `maturin` project with a PyO3 extension
  module named `rumba`.
- Added Rust-owned `rumba.__version__`, `rumba.njit`, and `rumba.jit`.
- Added a PyO3 dispatcher object with `.py_func`, `.signatures`, `_compiled`,
  `.inspect_bytecode()`, `.inspect_rumba_ast()`, `.inspect_c()`,
  `.inspect_compile_command()`, and `.inspect_cache_path()`.
- Added lazy compile-on-first-call for observed signatures.
- Added a Rust bytecode frontend for the currently supported Python 3.12
  scalar and 1D array subset.
- Added a dedicated Rust typing pass that feeds C emission.
- Moved scalar and 1D array type handling, C emission, cache key creation,
  generated-source writing, C compiler discovery, C compiler invocation, and
  shared-library artifact metadata into Rust.
- Moved decorator ergonomics, argument type discovery, exception types, compiled
  library loading, and native invocation into Rust.
- Removed `rumba/python/rumba`; no Python files implement the package.
- Added clear Rust-defined `RumbaUnsupportedError` and
  `RumbaCompilationError` exception types.

The first implemented subset supports scalar `int64`, `float64`, and `bool`
arguments, arithmetic, comparisons, simple `if` statements, `range` loops,
local assignment, augmented assignment, helper function calls, and scalar
returns. It also supports initial 1D contiguous NumPy array handling for
`int64` and `float64`, including `len(array)`, element load/store, dtype and
layout validation, selected builtin/math/NumPy reduction intrinsics, and
scalar-returning kernels that mutate arrays in place.

Current known gap in the chosen Python syntax subset:

- Helper function calls are supported only for top-level helpers already
  decorated with `@rumba.njit`. Plain Python helpers, local nested helpers,
  closures, recursive helpers, and runtime dispatcher calls from generated C
  remain unsupported.

## Non-Negotiable Direction

Rumba is not a Python reimplementation with a Rust helper. Rumba is a Rust
compiler/runtime exposed to Python.

The following pieces are now Rust-owned in the initial slice:

- Argument type discovery and signature construction.
- Dispatcher object and specialization cache.
- Public exception classes.
- Bytecode decoding and Rumba AST construction for the currently supported
  subset.
- Dedicated type inference and unsupported-operation diagnostics for the
  current scalar and 1D array subset.
- ABI conversion and native invocation for the currently supported scalar and
  1D array signatures.
- Cache key generation and shared-library compilation.

The following pieces still need deeper Rust implementations:

- Broader CPython bytecode coverage across supported Python versions.
- Full typed AST inspection/debug representations.
- Broader ABI argument conversion beyond the current scalar and 1D array view
  subset.
- Cache metadata serialization and invalidation.

## Public API

Initial public API:

- `rumba.__version__`
- `rumba.njit(fn=None, **options)`
- `rumba.jit(fn=None, **options)`

Supported options for now:

- `cache=False`
- `debug=False`
- `signature=None`

Unsupported Numba options must raise `RumbaUnsupportedError` instead of being
silently accepted or falling back to Python.

## Target Python Syntax Subset

Rumba intentionally targets a limited Python syntax subset. This subset is the
near-term language contract; broad Python or Numba parity is out of scope unless
explicitly added to this table.

| Feature | Target | Current status | Notes |
| ------- | ------ | -------------- | ----- |
| `if` / `else` branches | Supported | Partial | Simple branches are supported. Broaden bytecode/control-flow handling and tests for nested branches and branch-local assignments. |
| `elif` | Supported | Not complete | Treat as nested `else: if ...` in the AST/control-flow frontend and verify generated C preserves Python semantics. |
| `for` loops | Supported for `range(...)` only | Partial | Keep support focused on `for i in range(...)`; iteration over lists, tuples, arrays, generators, and arbitrary objects remains unsupported. |
| `range(start/stop/step)` | Supported | Partial | Support one-, two-, and three-argument integer ranges, including negative constant steps where practical. Reject non-integer range bounds. |
| `while` loops | Supported | Not started | Add bytecode control-flow recognition, AST node, type checking for boolean conditions, C lowering, and tests. |
| `break` / `continue` | Supported inside supported loops | Not started | Add structured loop exits in the AST/lowering. Reject use outside loops and unsupported nested-control-flow cases clearly. |
| Local assignment | Supported | Implemented | Continue to require statically typed local variables in the Rust typing pass. |
| Augmented assignment | Supported | Implemented | Current support is scalar-focused; array element augmented assignment should remain explicit future work. |
| Arithmetic operators | Supported for scalar numeric values | Partial | Maintain support for the scalar numeric subset first: `+`, `-`, `*`, `/`, `//`, `%`, unary `+`, unary `-`. Broader operators are not implied. |
| Comparisons | Supported for scalar values | Partial | Support equality and ordering comparisons for scalar numeric/bool combinations where typing and C lowering are defined. |
| Boolean conditions | Supported | Partial | Conditions must type as `bool`. Truthiness for arrays, objects, lists, tuples, and arbitrary values remains unsupported. |
| Unary operators | Supported for scalar values | Partial | Support unary numeric signs and boolean `not` for typed scalar expressions. |
| Function calls | Supported for selected helper and intrinsic calls | Partial | Support direct calls to top-level `@rumba.njit` helper functions and explicit native intrinsics for `len`, scalar `min`/`max`/`abs`, selected `math.*` calls, and 1D NumPy `max`/`min`/`sum`, all without Python fallback. Plain Python helpers, local nested helpers/closures, and recursive helpers are rejected. |

Everything outside this table should be treated as unsupported by default. New
syntax must be added deliberately with frontend tests, typing tests, C codegen
tests, execution tests, and unsupported-operation diagnostics.

## Architecture Milestones

### Milestone 1: Importable Rust Extension

Status: Implemented.

Acceptance criteria:

```bash
cd rumba
maturin develop
python -m pytest
python -c "import rumba; print(rumba.__version__)"
```

### Milestone 2: Minimal User API

Status: Implemented in Rust.

The current API supports `@rumba.njit`, `@rumba.njit(...)`, and `rumba.jit` as
an alias. The dispatcher class is a PyO3 `#[pyclass]`.

### Milestone 3: Rust-Owned Scalar Codegen Path

Status: Implemented.

Rust now owns scalar C source generation, cache key creation, generated-source
writing, C compiler selection, C compiler invocation, artifact metadata, and a
dedicated typing pass feeding C emission.

Remaining work:

- Continue broadening typed IR diagnostics as the language surface grows.

### Milestone 4: Rust Bytecode Frontend

Status: Partial.

The current implementation reads CPython code objects through PyO3 and decodes
the Python 3.12 bytecode needed for the supported scalar and 1D array subset.
It builds a Rust-owned Rumba AST for straight-line code, branches, simple
`range` loops, `len`, scalar locals, helper calls, and array indexing.

Required capabilities:

- Decode instructions, offsets, constants, names, locals, freevars, and source
  locations. Partial.
- Build control-flow blocks and stack effects. Partial.
- Reject closures, generators, exceptions, comprehensions, object operations,
  and unsupported opcodes with structured diagnostics. Partial.
- Produce a Rust-owned Rumba AST for straight-line code, branches, simple
  loops, `range`, `len`, scalar locals, and array indexing. Implemented for the
  current subset.
- Keep bytecode frontend tests independent from Numba imports. Implemented.

### Milestone 5: Rust Dispatcher And Runtime Invocation

Status: Partial.

Implement the dispatcher as a PyO3 class.

Required capabilities:

- Store the original Python function as `.py_func`. Implemented.
- Discover argument types in Rust. Implemented for scalars and 1D contiguous
  `int64`/`float64` NumPy arrays.
- Compile lazily on first call for an observed signature. Implemented.
- Cache compiled artifacts by bytecode hash, constants, closure-free globals
  used, signature, Python version, Rumba version, target platform, and compiler
  flags. Partial.
- Invoke compiled functions from Rust instead of Python `ctypes`. Implemented
  for the current scalar and 1D array ABI combinations.
- Preserve inspection helpers for bytecode, Rumba AST, generated C, compiler
  command, and cache path. Implemented except typed AST.

### Milestone 6: Rust Type Inference And Diagnostics

Status: Dedicated pass implemented for current scalar and 1D array slice.

Current typing is handled by a dedicated Rust type inference pass over the Rumba
AST before C emission.

Required capabilities:

- Scalar types: `int64`, `float64`, `bool`.
- 1D array view types for contiguous `int64` and `float64` NumPy arrays.
- Clear ambiguity errors and unsupported-operation diagnostics.
- Diagnostic spans tied to bytecode offsets and source line information when
  available.

### Milestone 7: NumPy Array Interop In Rust

Status: Partial.

Initial array support is implemented for:

- 1D contiguous `int64` and `float64` arrays. Implemented.
- Rust/PyO3 validation of dtype, dimensionality, contiguity, and mutability.
  Implemented.
- ABI representation containing data pointer and length. Implemented.
- Lower array element load/store and `len(array)` to C. Implemented.
- Support scalar-returning kernels that mutate arrays in place. Implemented.

Remaining work:

- Carry item size and stride metadata if non-contiguous or strided views become
  supported.
- Broaden supported array ABI combinations deliberately.
- Keep array-returning functions unsupported until ownership and lifetime rules
  are designed.

### Milestone 8: Focused Python Syntax Growth

Status: Partial.

Grow only the selected Python syntax subset, with Rust implementation and tests
for each feature. This milestone replaces any broad Numba parity goal.

Required capabilities:

- Complete simple `if` / `else` support, including nested branches and branch
  merge diagnostics.
- Add `elif` support through normalized nested branch handling.
- Complete `range(start/stop/step)` handling for integer scalar bounds.
- Add `while` loops with boolean typed conditions.
- Add `break` and `continue` for supported `for range` and `while` loops.
- Keep local assignment and augmented assignment stable as the control-flow
  surface grows.
- Broaden scalar arithmetic, comparison, boolean, and unary operator tests
  within the supported scalar type set.
- Keep helper function calls reliable for supported top-level `@rumba.njit`
  helper functions, and preserve explicit tests for unsupported plain Python
  helpers, local nested helpers/closures, recursive helpers, default args,
  kwargs, varargs, and keyword-only args.

Out of scope for this milestone:

- Object mode or fallback to Python execution.
- General Python iterators or iteration over containers.
- Lists, dicts, sets, tuples, comprehensions, generators, exceptions, classes,
  recursion, and closures unless a later milestone explicitly adds them.
- Broad NumPy, array allocation/constructors such as `np.empty`/`np.zeros`,
  broadcasting, GPU, or parallel support.

### Milestone 9: Packaging And Developer Workflow

Status: Partial.

Keep the project independently buildable:

```bash
cd rumba
maturin develop
python -m pytest
```

The source-tree fallback used during development may symlink a Cargo-built
extension for local testing, but `maturin develop` is the supported workflow.

## Explicitly Unsupported

Unsupported behavior must raise `RumbaUnsupportedError`, not fall back to
Python:

- Python objects beyond supported scalars and planned array views.
- Lists, dicts, sets, tuples, exceptions, comprehensions, generators, closures,
  recursion, object mode, GPU, parallel mode, and general NumPy broadcasting.
- Array-returning functions in the initial native path.
- Bool arrays and non-contiguous NumPy arrays.
- Windows native compilation until the shared-library and compiler-driver path
  is stable.

## Test Plan

- Package tests for import, version, decorator forms, and unsupported options.
- Rust unit tests for bytecode decoding, Rumba AST construction, type inference,
  C generation, cache keys, and compiler command construction.
- Frontend tests for simple arithmetic, branches, loops, `range`, `len`, and
  array indexing.
- Focused Python syntax tests for nested `if` / `else`, `elif`, one-, two-, and
  three-argument `range`, `while`, `break`, `continue`, scalar arithmetic,
  comparisons, boolean conditions, unary operators, local assignment, augmented
  assignment, and supported jitted-only helper calls.
- Execution tests for scalar arithmetic, branches, loops, cache reuse, and
  distinct signatures.
- NumPy tests for 1D array reads/writes, dtype mismatch, non-contiguous arrays,
  unsupported dimensions, and unsupported returns.
- Compatibility tests compare against Python first and against Numba only for
  explicitly supported semantics. Rumba implementation code must not import
  Numba.

## Progress Log

| Date | Update |
| ---- | ------ |
| 2026-04-26 | Initial migration plan created. |
| 2026-04-26 | Added independent `rumba/` package scaffold and first scalar C/ctypes execution slice. |
| 2026-04-27 | Moved scalar C generation, cache key generation, generated source writing, compiler discovery, compiler invocation, and artifact metadata into Rust. |
| 2026-04-27 | Updated project direction: Rumba is a Rust compiler/runtime with Python only as a thin package boundary and temporary glue. |
| 2026-04-27 | Removed the Python package implementation and made `rumba` a top-level Rust/PyO3 extension module. |
| 2026-04-27 | Removed the Python package implementation files; Python remains limited to public API tests and optional examples. |
| 2026-04-28 | Added dedicated Rust typing pass, compiler/cache inspection helpers, Python 3.12 bytecode frontend coverage for the current subset, and initial 1D NumPy array interop. |
| 2026-05-02 | Added explicit native intrinsic resolution, typing, inspection, and C lowering for selected builtins, `math` calls, and 1D NumPy reductions while preserving rejection of arbitrary Python helpers. |
