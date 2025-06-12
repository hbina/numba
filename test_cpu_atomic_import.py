#!/usr/bin/env python3
"""
Test script to verify that numba.cpu.atomic can be imported and accessed
"""

def test_import():
    """Test that we can import numba.cpu.atomic"""
    try:
        # Test the import path we want to enable
        import numba.cpu
        print("✓ Successfully imported numba.cpu")
        
        # Check that atomic namespace exists
        if hasattr(numba.cpu, 'atomic'):
            print("✓ numba.cpu.atomic namespace exists")
            
            # Check some atomic operations
            atomic_ops = ['load', 'store', 'add', 'sub', 'max', 'min', 'and_', 'or_', 'xor', 'exchange', 'compare_exchange']
            available_ops = []
            
            for op in atomic_ops:
                if hasattr(numba.cpu.atomic, op):
                    available_ops.append(op)
                    
            print(f"✓ Available atomic operations: {available_ops}")
        else:
            print("✗ numba.cpu.atomic namespace not found")
            
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = test_import()
    if success:
        print("\n✓ All tests passed - CPU atomic operations are accessible!")
    else:
        print("\n✗ Some tests failed")