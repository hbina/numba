# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build and Setup
```bash
# Create development environment (conda recommended)
conda create -n numbaenv python=3.10 numba/label/dev::llvmlite numpy scipy jinja2 cffi

# Activate the development environment
conda activate numbaenv

# Build extensions in-place for development
python setup.py build_ext --inplace --debug

# Install for development (use this in the numba folder)
python -m pip install -e .
```

### Testing
```bash
# Run full test suite
python -m numba.runtests

# Run with specific options
python -m numba.runtests -b -v -g -m <nprocs> -- numba.tests

# Run with CUDA simulation
NUMBA_ENABLE_CUDASIM=1 python -m numba.runtests

# List all available tests
python -m numba.runtests -l

# Run specific test shards (CI approach)
python -m numba.runtests -j "start_index:total_count" --exclude-tags='long_running'
```

### Code Quality
```bash
# Run code style checks
flake8 -j auto numba

# Run type checking (selective enforcement)
mypy
```

### Useful Environment Variables
```bash
NUMBA_DEVELOPER_MODE=1        # Enable dev mode with full tracebacks
NUMBA_ENABLE_CUDASIM=1        # Enable CUDA simulator for GPU testing
NUMBA_THREADING_LAYER=tbb     # Set threading backend (tbb/omp/workqueue)
```

### Critical: After Source Code Changes
**Every time you modify Numba source code, you MUST reinstall:**
```bash
python -m pip install -e .
```
This rebuilds compiled extensions. Changes won't take effect without reinstalling.

## Architecture Overview

### Core Components

**Compiler Pipeline** (`numba/core/`):
- Frontend: Python bytecode → Numba IR → Type inference
- Backend: Typed Numba IR → LLVM IR → Machine code
- Two compilation modes: object mode (with Python fallback) vs nopython mode (pure compiled)

**Type System** (`numba/core/types/` and `numba/core/typing/`):
- Comprehensive type hierarchy for Python and NumPy types
- Type inference engine that determines variable types during compilation

**Targets**:
- CPU target: `numba/core/cpu.py`
- CUDA target: `numba/cuda/`
- Extensible target system for different hardware backends

**Runtime** (`numba/core/runtime/`):
- Numba Runtime (NRT) handles memory management
- Reference counting for Python objects in compiled code

### Key Directories

- `numba/core/` - Compiler infrastructure, types, and typing
- `numba/cpython/` - CPython object implementations for compiled code
- `numba/np/` - NumPy support and ufunc infrastructure
- `numba/cuda/` - GPU/CUDA compilation target
- `numba/experimental/` - Experimental features (jitclass, structref)
- `numba/parfors/` - Parallel for-loop optimization
- `numba/typed/` - Typed containers (List, Dict)

### Extension Points

**Function Overloading** (`numba/core/extending.py`):
- `@overload` - Implement functions for Numba types
- `@intrinsic` - Define low-level LLVM intrinsics
- Target extension API for new hardware backends

**Testing Infrastructure**:
- Tests located in `numba/tests/` and module-specific `tests/` directories
- Test tagging system for categorization and selective running
- Parallel test execution with sharding support
- CUDA simulator allows GPU testing without hardware

## Adding New Functionality to Numba

### Component Architecture for Extensions

When adding new functionality, you typically need these components:

**1. Stubs (`*_stubs.py`)**
- Define the Python API using `Stub` classes from `numba.cuda.stubs`
- Create namespaced APIs (e.g., `numba.something.operation()`)
- Document function signatures and parameters

**2. Type Declarations (`numba/core/typing/*_decl.py`)**
- Create `AbstractTemplate` classes with `generic()` methods for type inference
- Use `Registry()` pattern: `registry = Registry(); register = registry.register`
- Handle different argument patterns and optional parameters
- Follow CUDA's `_gen()` pattern for similar operations (generates template classes)
- For 1D arrays, normalize index to `types.intp` in signatures

**3. LLVM Implementations (`*_impl.py`)**
- Use `@lower_builtin` decorators to map stub functions to LLVM code
- Implement both pointer-based and array-based versions
- Use LLVM instructions via `builder.atomic_rmw()`, `builder.load_atomic()`, etc.
- Handle memory ordering conversion (`'acquire'` → `'acquire'`, `'relaxed'` → `'monotonic'`)

**4. Module Registration**
- **For CPU**: Extend the typing context by creating a custom `CPUTypingContext` that includes your registry
- **For CUDA**: Add to the CUDA typing context's `load_additional_registries()`
- Register global modules using `registry.register_global(module, types.Module(module))`
- Create `AttributeTemplate` classes to resolve module attributes

### Type System Integration

**Template Classes**:
- `AbstractTemplate` with `generic(args, kws)` for flexible type matching
- `ConcreteTemplate` with `cases = [signature(...)]` for fixed signatures
- `AttributeTemplate` with `resolve_*` methods for module attribute resolution

**Signature Patterns**:
- Use `signature(return_type, *arg_types)` from `numba.core.typing.templates`
- Handle optional arguments by checking `len(args)`
- Validate string literals for configuration parameters
- CUDA pattern: normalize 1D arrays to `types.intp`, keep original types for multi-D

### Target Integration

**CPU Target**:
- Modify `numba/core/registry.py` to use extended typing context
- Import implementations in target context's `load_additional_registries()`
- CPU uses standard LLVM atomic instructions

**CUDA Target**:
- Add registries to `CUDATypingContext.load_additional_registries()`
- CUDA uses NVVM intrinsics via `nvvmutils.declare_*` functions

### Testing Strategy

**Test File Structure**:
- Create `numba/tests/test_<feature>.py` with comprehensive test classes
- Test basic functionality, thread safety, edge cases, error conditions
- Use `@unittest.skipUnless(FEATURE_AVAILABLE, "reason")` for optional features
- Include both compilation tests and runtime behavior validation

### Common Patterns Observed

**CUDA as Reference**:
- CUDA implementation is often the most complete reference for patterns
- Look at `numba/cuda/cudadecl.py` and `numba/cuda/cudaimpl.py` for examples
- Module attribute resolution pattern: `resolve_<attr>` returning `types.Module` or `types.Function`

**Error Debugging**:
- "Untyped global name" → Missing module/global registration
- "No implementation found" → Missing `@lower_builtin` decorators or signature mismatch
- Signature mismatch → Check template `generic()` return values vs `@lower_builtin` signatures

**Performance Considerations**:
- Use LLVM's native atomic instructions when possible
- Minimize Python object creation in compiled code
- Consider memory ordering semantics for correctness vs performance

## Dependencies

- **Required**: NumPy ≥1.24, llvmlite ≥0.45.0dev0, Python 3.10-3.13
- **Optional**: TBB, OpenMP for parallel backends
- **Build**: setuptools, Cython for some extensions

## Development Notes

- Numba is a JIT compiler that translates Python functions to optimized machine code using LLVM
- The codebase supports both CPU and GPU (CUDA) targets
- Compilation happens in two main modes: object mode (mixed Python/compiled) and nopython mode (fully compiled)
- The type system is central to compilation - understanding `numba/core/types/` is crucial for most development
- Use `python -m numba -s` to check system configuration and available features

### Development Workflow Tips

**Iterative Development Process**:
1. Study existing CUDA implementations for reference patterns
2. Create stubs → type declarations → LLVM implementations → registration → tests
3. Test frequently with simple cases before adding complexity
4. Always reinstall after source changes: `python -m pip install -e .`

**Debugging Common Issues**:
- Always check CUDA implementations first for established patterns
- Use `NUMBA_DEVELOPER_MODE=1` for detailed error tracebacks
- Template signatures must exactly match `@lower_builtin` signatures
- Global name resolution requires module registration with `AttributeTemplate`
- Array operations need both pointer and array-based lowering rules

**Testing Integration**:
- Compile functions first in tests to catch typing errors early
- Test both basic functionality and thread safety for concurrent features
- Use `@unittest.skipUnless` for features that may not be available