"""
Example demonstrating CPU atomic operations in Numba

This example shows how to use the new CPU atomic operations for
thread-safe programming with Numba.
"""

import numpy as np
import threading
from numba import njit
import numba.cpu


def example_basic_atomic_operations():
    """Basic example of atomic operations"""

    @njit
    def demo_atomic_ops():
        # Create a shared array
        shared_data = np.array([10, 20, 30], dtype=np.uint64)

        # Atomic load - safely read a value
        val = numba.cpu.atomic.load(shared_data, 0)
        print(f"Loaded value: {val}")

        # Atomic store - safely write a value
        numba.cpu.atomic.store(shared_data, 1, np.uint64(99))
        print(f"Stored 99, array now: {shared_data}")

        # Atomic add - safely increment and get previous value
        old_val = numba.cpu.atomic.add(shared_data, 2, np.uint64(5))
        print(f"Added 5 to index 2, old value: {old_val}, new array: {shared_data}")

        # Note: compare_and_swap not yet implemented in this demo
        # old_val = numba.cpu.atomic.compare_and_swap(shared_data, 1, np.uint64(99), np.uint64(777))
        # print(f"CAS: old={old_val}, array now: {shared_data}")

        return shared_data

    print("=== Basic Atomic Operations ===")
    result = demo_atomic_ops()
    print(f"Final array: {result}")
    print()


def example_thread_safe_counter():
    """Thread-safe counter using atomic operations"""

    @njit
    def increment_counter(counter_arr, num_increments):
        """Atomically increment a counter"""
        for _ in range(num_increments):
            numba.cpu.atomic.add(counter_arr, 0, np.uint64(1))

    print("=== Thread-Safe Counter ===")

    # Create shared counter
    counter = np.array([0], dtype=np.uint64)

    # Create multiple threads that increment the counter
    num_threads = 4
    increments_per_thread = 1000

    threads = []
    for i in range(num_threads):
        t = threading.Thread(
            target=increment_counter, args=(counter, increments_per_thread)
        )
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    expected = num_threads * increments_per_thread
    actual = counter[0]

    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    print(f"Success: {actual == expected}")
    print()


def example_memory_ordering():
    """Example demonstrating memory ordering options"""

    @njit
    def demo_memory_ordering():
        data = np.array([1, 2, 3], dtype=np.uint8)

        # Different memory orderings
        val1 = numba.cpu.atomic.load(data, 0, "acquire")  # Acquire semantics
        numba.cpu.atomic.store(data, 1, np.uint8(42), "release")  # Release semantics
        old = numba.cpu.atomic.add(data, 2, np.uint8(10), "acq_rel")  # Acquire-release

        return val1, data[1], old, data[2]

    print("=== Memory Ordering ===")
    val1, stored_val, old_val, new_val = demo_memory_ordering()
    print(f"Acquired value: {val1}")
    print(f"Released value: {stored_val}")
    print(f"Add old value: {old_val}, new value: {new_val}")
    print()


def example_uint8_operations():
    """Example with uint8 operations"""

    @njit
    def uint8_operations():
        # Work with uint8 data
        data = np.array([100, 200, 50], dtype=np.uint8)

        # Test boundary conditions
        numba.cpu.atomic.store(data, 0, np.uint8(255))  # Max uint8 value

        # This will overflow and wrap around
        old = numba.cpu.atomic.add(data, 0, np.uint8(1))  # 255 + 1 = 0 (wrap)

        return old, data[0]

    print("=== uint8 Operations (with overflow) ===")
    old_val, new_val = uint8_operations()
    print(f"Added 1 to 255: old={old_val}, new={new_val} (wrapped)")
    print()


if __name__ == "__main__":
    print("Numba CPU Atomic Operations Examples")
    print("=" * 40)

    try:
        example_basic_atomic_operations()
        example_thread_safe_counter()
        example_memory_ordering()
        example_uint8_operations()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nNote: These examples require:")
        print("- NumPy")
        print("- Numba with CPU atomic support")
        print("- Python threading support")
