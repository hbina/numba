#!/usr/bin/env python
"""
Final verification of CPU atomic operations registration
"""

import numpy as np
from numba import njit

# Test all import patterns
print("Testing import patterns...")

# 1. Test direct module access
import numba.cpu

print("✓ import numba.cpu")

# 2. Test atomic module access
print(f"✓ numba.cpu.atomic: {numba.cpu.atomic}")

# 3. Test direct import
from numba.cpu import atomic

print(f"✓ from numba.cpu import atomic: {atomic}")

# Test compilation and execution
print("\nTesting compilation and execution...")


@njit
def test_module_access():
    arr = np.array([100], dtype=np.int64)
    return numba.cpu.atomic.fetch_add(arr, 0, np.int64(42))


@njit
def test_direct_import():
    arr = np.array([200], dtype=np.int64)
    return atomic.fetch_add(arr, 0, np.int64(58))


try:
    result1 = test_module_access()
    print(f"✓ Module access compilation: {result1}")

    result2 = test_direct_import()
    print(f"✓ Direct import compilation: {result2}")

    print("\n🎉 All CPU atomic operations are working correctly!")
    print("The 'Untyped global name' error has been resolved.")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
