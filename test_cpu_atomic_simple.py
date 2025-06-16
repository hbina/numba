#!/usr/bin/env python

"""
Simple test for CPU atomic operations
"""

import numpy as np
import numba
from numba import njit

print("Testing CPU atomic operations...")


@njit
def test_atomic_fetch_add():
    # Test with int64 array
    arr = np.array([10], dtype=np.int64)

    # This should work if our registration is correct
    old_val = numba.cpu.atomic.fetch_add(arr, 0, np.int64(5))
    return old_val, arr[0]


if __name__ == "__main__":
    try:
        # Test that the module can be imported
        import numba.cpu

        print("✓ numba.cpu module imported successfully")

        # Test that atomic can be accessed
        print(f"✓ numba.cpu.atomic available: {numba.cpu.atomic}")

        # Test compilation and execution
        old_val, new_val = test_atomic_fetch_add()
        if old_val == 10 and new_val == 15:
            print("✓ CPU atomic fetch_add works correctly!")
            print(f"  Previous value: {old_val}, New value: {new_val}")
        else:
            print(f"✗ CPU atomic fetch_add failed: old={old_val}, new={new_val}")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
