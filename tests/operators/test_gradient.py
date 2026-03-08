# tests/operators/test_gradient.py

import numpy as np
import pytest

from src.step3.ops.gradient import gradient_p_n, gradient_p_n_plus_1


@pytest.fixture
def grid_setup():
    """Provides a consistent setup for operator verification."""
    nx, ny, nz = 5, 5, 5
    p = np.random.rand(nx, ny, nz)
    dx, dy, dz = 0.1, 0.1, 0.1
    return p, dx, dy, dz

def test_gradient_p_n_equivalence(grid_setup):
    p, dx, dy, dz = grid_setup
    
    # Audit Reference (Loop-based)
    ref_x = np.zeros((3, 3, 3))
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                ref_x[i-1, j-1, k-1] = -(p[i+1, j, k] - p[i-1, j, k]) / (2 * dx)
    
    # Production Optimized
    opt_x, _, _ = gradient_p_n(p, dx, dy, dz)
    
    # Proof
    assert np.allclose(ref_x, opt_x)

def test_gradient_p_n_plus_1_equivalence(grid_setup):
    p, dx, dy, dz = grid_setup
    
    # Audit Reference
    ref_x = np.zeros((3, 3, 3))
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                ref_x[i-1, j-1, k-1] = -(p[i+1, j, k] - p[i-1, j, k]) / (2 * dx)
    
    # Production Optimized
    opt_x, _, _ = gradient_p_n_plus_1(p, dx, dy, dz)
    
    # Proof
    assert np.allclose(ref_x, opt_x)