#!/usr/bin/env python
"""
Comprehensive test for CPU atomic operations
"""

import numpy as np
import numba
from numba import njit

print("Testing comprehensive CPU atomic operations...")


@njit
def test_fetch_add():
    arr = np.array([100], dtype=np.int64)
    old_val = numba.cpu.atomic.fetch_add(arr, 0, np.int64(42))
    return old_val, arr[0]


@njit
def test_add():
    arr = np.array([200], dtype=np.int64)
    old_val = numba.cpu.atomic.add(arr, 0, np.int64(58))
    return old_val, arr[0]


@njit
def test_sub():
    arr = np.array([300], dtype=np.int64)
    old_val = numba.cpu.atomic.sub(arr, 0, np.int64(50))
    return old_val, arr[0]


@njit
def test_load_store():
    arr = np.array([400], dtype=np.int64)
    # Store a value atomically
    numba.cpu.atomic.store(arr, 0, np.int64(500))
    # Load a value atomically
    loaded_val = numba.cpu.atomic.load(arr, 0)
    return loaded_val, arr[0]


@njit
def test_compare_and_swap():
    arr = np.array([600], dtype=np.int64)
    # Try to swap 600 -> 700
    old_val = numba.cpu.atomic.compare_and_swap(arr, 0, np.int64(600), np.int64(700))
    return old_val, arr[0]


if __name__ == "__main__":
    try:
        print("1. Testing fetch_add...")
        old_val, new_val = test_fetch_add()
        print(f"   fetch_add: {old_val} -> {new_val} (expected 100 -> 142)")
        assert old_val == 100 and new_val == 142, (
            f"fetch_add failed: {old_val}, {new_val}"
        )

        print("2. Testing add...")
        old_val, new_val = test_add()
        print(f"   add: {old_val} -> {new_val} (expected 200 -> 258)")
        assert old_val == 200 and new_val == 258, f"add failed: {old_val}, {new_val}"

        print("3. Testing sub...")
        old_val, new_val = test_sub()
        print(f"   sub: {old_val} -> {new_val} (expected 300 -> 250)")
        assert old_val == 300 and new_val == 250, f"sub failed: {old_val}, {new_val}"

        print("4. Testing load/store...")
        loaded_val, final_val = test_load_store()
        print(f"   load/store: {loaded_val}, {final_val} (expected 500, 500)")
        assert loaded_val == 500 and final_val == 500, (
            f"load/store failed: {loaded_val}, {final_val}"
        )

        print("5. Testing compare_and_swap...")
        old_val, new_val = test_compare_and_swap()
        print(f"   compare_and_swap: {old_val} -> {new_val} (expected 600 -> 700)")
        assert old_val == 600 and new_val == 700, (
            f"compare_and_swap failed: {old_val}, {new_val}"
        )

        print("\n✓ All CPU atomic operations working correctly!")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
