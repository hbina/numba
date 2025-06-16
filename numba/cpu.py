"""
CPU-specific atomic operations module for Numba

This module provides a CPU equivalent to CUDA's atomic operations,
using LLVM's atomic instructions for lock-free operations on CPU targets.
"""

# Import the atomic stubs to make them available as numba.cpu.atomic
from numba.core.cpu_atomic_stubs import atomic

# This creates the numba.cpu.atomic namespace that users can import
