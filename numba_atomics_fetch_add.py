#!/usr/bin/env python3
"""
IPC counter shared via a real, on-disk mmap
(no NumPy — only Python’s standard library).
"""

import mmap
import multiprocessing
import random
import struct
import tempfile
import time

import numpy as np

import numba

# from multiprocessing import Process, Lock, set_start_method

ITERATION_COUNT = 16  # number of processes
ADD_COUNT = 1024 * 1024 * 1024  # increments per process
INT64_FMT = "<q"  # little-endian signed int64, 8 bytes
BYTES_NEEDED = struct.calcsize(INT64_FMT)


def create_mmap_file(file_size):
    tmp = tempfile.NamedTemporaryFile(prefix="shared_counter_", delete=False)
    tmp.truncate(BYTES_NEEDED)  # make sure file is big enough
    tmp.close()

    f = open(tmp.name, "r+b")
    f.write(b"\x00" * file_size)
    f.flush()

    return tmp.name


@numba.njit(cache=False)
def _numba_run(sync_buffer, add_buffer, idx, iterations):
    total = 0
    sync_buffer[idx] = 1

    while True:
        all_ready = True
        for i in range(ITERATION_COUNT):
            all_ready = all_ready and (sync_buffer[i] == 1)

        if all_ready:
            break

    for i in range(iterations):
        total += add_buffer[0]
        add_buffer[0] += 1

    return total


def worker(sync_path: str, add_path: str, idx: int, iterations: int) -> None:
    sync_file = open(sync_path, "r+b")
    sync_mm = mmap.mmap(sync_file.fileno(), ITERATION_COUNT, access=mmap.ACCESS_WRITE)
    sync_buffer = np.frombuffer(sync_mm, dtype=np.uint8)

    add_file = open(add_path, "r+b")
    add_mm = mmap.mmap(add_file.fileno(), 8, access=mmap.ACCESS_WRITE)
    add_buffer = np.frombuffer(add_mm, dtype=np.int64)

    _numba_run(sync_buffer, add_buffer, idx, iterations)


def __main():
    sync_path = create_mmap_file(ITERATION_COUNT)
    add_path = create_mmap_file(8)
    procs = [
        multiprocessing.Process(
            target=worker, args=(sync_path, add_path, idx, ADD_COUNT)
        )
        for idx in range(ITERATION_COUNT)
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    add_file = open(add_path, "rb")
    add_buffer = np.memmap(add_file, dtype=np.int64, mode="r")
    final_val = add_buffer[0]
    expected = ITERATION_COUNT * ADD_COUNT
    print(f"Final value: {final_val}, expected: {expected}")
    assert final_val == expected + ITERATION_COUNT


if __name__ == "__main__":
    __main()
