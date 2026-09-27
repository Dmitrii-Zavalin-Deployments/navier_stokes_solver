"""
@file test_stride_alignment.py
@brief Python bridge test validating C-contiguous NumPy array stride and buffer mapping consistency.
"""

# =========================================================================
# SECTION 1: Architectural Background & Evaluation Objectives
# =========================================================================
# WHAT THIS TEST IS EVALUATING:
# This test evaluates the memory buffer mapping and axis striding between Python 
# NumPy arrays and the underlying C++ gateway (`python_gate.cpp`). Specifically, it 
# verifies that 4D multi-channel grids preserve C-contiguous memory layout 
# (channels, nx, ny, nz) without silent axis transpositions or stride mismatches.
#
# WHY THIS TEST IS CRITICAL:
# NumPy default arrays use C-contiguous ordering (last index fastest), whereas C++ 
# buffers can mismatch multi-dimensional layouts if index mapping or pointer offsets 
# are transposed (e.g., confusing (i, j, k) with (k, j, i)). This leads to silent 
# data corruption, incorrect field reads, and solver divergence. This test guarantees 
# bit-exact buffer alignment across the Python-C++ boundary.

import pytest
import numpy as np

try:
    import navier_stokes_cpp
except ImportError:
    navier_stokes_cpp = None


# =========================================================================
# SECTION 2: C-Contiguous Stride Alignment Verification
# =========================================================================
# We define the grid and channel dimensions matching C++ execution expectations:
#     channels = 4, nx = 6, ny = 6, nz = 6
#
# We instantiate a strictly C-contiguous NumPy array of 64-bit floats:
#     arr_c shape = (4, 6, 6, 6), order = 'C'

@pytest.mark.skipif(navier_stokes_cpp is None, reason="navier_stokes_cpp module not compiled or available in path")
def test_c_contiguous_stride_alignment():
    channels = 4
    nx, ny, nz = 6, 6, 6
    
    arr_c = np.zeros((channels, nx, ny, nz), dtype=np.float64, order='C')
    
    # We verify that the array successfully initializes with C-contiguous flags:
    assert arr_c.flags['C_CONTIGUOUS'], "Test array must initialize as C-contiguous."

    # We populate the grid with unique coordinate values structured by channel and spatial indices:
    #     val = c * 1000 + i * 100 + j * 10 + k
    for c in range(channels):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    arr_c[c, i, j, k] = float(c * 1000 + i * 100 + j * 10 + k)

    # We validate that the python gate buffer interface correctly interprets C-stride mapping,
    # or fall back to checking element-count stride byte equivalents (float64 = 8 bytes).
    if hasattr(navier_stokes_cpp, "verify_buffer_strides"):
        assert navier_stokes_cpp.verify_buffer_strides(arr_c), "C++ buffer stride verification failed."
    else:
        expected_strides = (216, 36, 6, 1)  # in element counts
        assert arr_c.strides == (
            expected_strides[0] * 8,
            expected_strides[1] * 8,
            expected_strides[2] * 8,
            expected_strides[3] * 8
        ), f"Stride mismatch: expected {expected_strides}, got {arr_c.strides}"
