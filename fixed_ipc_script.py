#!/usr/bin/env python3
"""
IPC counter shared via a real, on-disk mmap using CPU atomics.
"""

import mmap
import multiprocessing
import random
import struct
import tempfile
import time

import numpy as np
import numba
import numba.cpu.atomic as atomic

ITERATION_COUNT = 16  # number of processes
ADD_COUNT = 1024  # increments per process
INT64_FMT = "<q"  # little-endian signed int64, 8 bytes
BYTES_NEEDED = struct.calcsize(INT64_FMT)


def create_mmap_file(file_size):
    tmp = tempfile.NamedTemporaryFile(prefix="shared_counter_", delete=False)
    tmp.truncate(file_size)  # make sure file is big enough
    tmp.close()

    f = open(tmp.name, "r+b")
    f.write(b"\x00" * file_size)
    f.flush()
    f.close()

    return tmp.name


@numba.njit(cache=False)
def _numba_run(sync_buffer, add_buffer, idx, iterations):
    total = 0
    
    # Atomically set this process as ready
    atomic.store(sync_buffer, idx, 1)

    # Wait for all processes to be ready using atomic loads
    while True:
        all_ready = True
        for i in range(ITERATION_COUNT):
            if atomic.load(sync_buffer, i) != 1:
                all_ready = False
                break
        
        if all_ready:
            break

    # Perform atomic increments and reads
    for i in range(iterations):
        # Atomically read the current value
        current_val = atomic.load(add_buffer, 0)
        total += current_val
        
        # Atomically increment the counter (fetch-and-add)
        atomic.add(add_buffer, 0, 1)

    return total


def worker(sync_path: str, add_path: str, idx: int, iterations: int) -> None:
    # Open sync file and map it
    sync_file = open(sync_path, "r+b")
    sync_mm = mmap.mmap(sync_file.fileno(), ITERATION_COUNT, access=mmap.ACCESS_WRITE)
    sync_buffer = np.frombuffer(sync_mm, dtype=np.uint8)

    # Open add file and map it
    add_file = open(add_path, "r+b")
    add_mm = mmap.mmap(add_file.fileno(), 8, access=mmap.ACCESS_WRITE)
    add_buffer = np.frombuffer(add_mm, dtype=np.int64)

    # Run the atomic operations
    result = _numba_run(sync_buffer, add_buffer, idx, iterations)
    
    # Clean up
    sync_mm.close()
    sync_file.close()
    add_mm.close()
    add_file.close()
    
    print(f"Process {idx} local total: {result}")


def main():
    # Create shared memory files
    sync_path = create_mmap_file(ITERATION_COUNT)
    add_path = create_mmap_file(8)
    
    print(f"Starting {ITERATION_COUNT} processes, each doing {ADD_COUNT} increments...")
    
    # Create and start processes
    procs = [
        multiprocessing.Process(
            target=worker, args=(sync_path, add_path, idx, ADD_COUNT)
        )
        for idx in range(ITERATION_COUNT)
    ]

    start_time = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    end_time = time.time()

    # Read final result
    add_file = open(add_path, "rb")
    add_buffer = np.memmap(add_file, dtype=np.int64, mode="r")
    final_val = add_buffer[0]
    add_file.close()
    
    expected = ITERATION_COUNT * ADD_COUNT
    print(f"Final value: {final_val}")
    print(f"Expected: {expected}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Verify the result
    if final_val == expected:
        print("✅ SUCCESS: Atomic operations worked correctly!")
    else:
        print(f"❌ FAILURE: Expected {expected}, got {final_val}")
        print(f"Difference: {final_val - expected}")


if __name__ == "__main__":
    main()