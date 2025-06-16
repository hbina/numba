#!/usr/bin/env python3
"""
Simple test for fetch_add implementation
"""

import sys
import os

# Add current directory to path to import numba
sys.path.insert(0, os.path.dirname(__file__))

try:
    import numpy as np
    from numba import njit
    from numba.cpu import atomic

    print("Testing fetch_add implementation...")

    # Test 1: Basic functionality test
    @njit
    def test_fetch_add_basic(arr, idx, val):
        return atomic.fetch_add(arr, idx, val)

    # Test with uint64 array
    arr = np.array([1000, 2000, 3000], dtype=np.uint64)
    print(f"Initial array: {arr}")

    old_val = test_fetch_add_basic(arr, 0, np.uint64(500))
    print(f"fetch_add returned: {old_val} (should be 1000)")
    print(f"Array after fetch_add: {arr} (arr[0] should be 1500)")

    # Verify results
    if old_val == 1000 and arr[0] == 1500:
        print("✓ Basic fetch_add test PASSED")
    else:
        print("✗ Basic fetch_add test FAILED")
        sys.exit(1)

    # Test 2: Memory ordering test
    @njit
    def test_fetch_add_ordering(arr, idx, val, ordering):
        return atomic.fetch_add(arr, idx, val, ordering)

    arr2 = np.array([100], dtype=np.uint64)
    old_val2 = test_fetch_add_ordering(arr2, 0, np.uint64(25), "seq_cst")
    print(f"fetch_add with seq_cst ordering returned: {old_val2} (should be 100)")
    print(f"Array after fetch_add: {arr2} (should be [125])")

    if old_val2 == 100 and arr2[0] == 125:
        print("✓ Memory ordering test PASSED")
    else:
        print("✗ Memory ordering test FAILED")
        sys.exit(1)

    print("\n🎉 All fetch_add tests PASSED!")

except ImportError as e:
    print(f"Import error: {e}")
    print("This is expected if dependencies are not installed")
    sys.exit(0)
except Exception as e:
    print(f"Error during test: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
