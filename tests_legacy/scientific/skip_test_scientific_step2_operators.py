# tests/scientific/test_scientific_step2_operators.py

import numpy as np
import pytest
import scipy.sparse as sp

from src.step2.operators import build_numerical_operators


def test_scientific_operators_dof_handshake(state_3d_small, capsys):
    """Rule 2.1: Verify DOF mapping and Debug Handshake prints."""
    # Fixture state_3d_small already initialized to 3x3x3 via conftest
    build_numerical_operators(state_3d_small)
    captured = capsys.readouterr().out
    
    assert "Target DOFs - P:27, U:36, V:36, W:36" in captured
    assert state_3d_small.operators.divergence.shape == (27, 108)
    assert state_3d_small.operators.grad_x.shape == (36, 27)

def test_scientific_gradient_coefficients(state_3d_small):
    """Rule 2.2: Verify finite difference coefficients (1/dx) in Gradient."""
    build_numerical_operators(state_3d_small)
    Gx = state_3d_small.operators.grad_x
    
    # Coefficients should be 10.0 (1/0.1) and -10.0
    assert Gx[17, 13] == pytest.approx(10.0)
    assert Gx[17, 12] == pytest.approx(-10.0)

def test_scientific_divergence_nullspace(state_3d_small):
    """Rule 2.3: Divergence of a constant velocity field must be zero."""
    build_numerical_operators(state_3d_small)
    D = state_3d_small.operators.divergence
    
    div_v = D @ np.ones(D.shape[1])
    assert div_v[13] == pytest.approx(0.0)

def test_scientific_composite_laplacian(state_3d_small, capsys):
    """Rule 2.4: Verify L = D @ G composition and non-emptiness."""
    build_numerical_operators(state_3d_small)
    captured = capsys.readouterr().out
    
    L = state_3d_small.operators.laplacian
    assert "Laplacian nnz:" in captured
    assert L.nnz > 0
    assert L.shape == (27, 27)
    
    # L_ii = -(2/dx^2 + 2/dy^2 + 2/dz^2) = -(200 + 200 + 200) = -600
    assert L[13, 13] == pytest.approx(-600.0)

def test_scientific_ppe_state_transfer(state_3d_small):
    """Rule 2.5: Ensure Laplacian is correctly committed to PPE solver state."""
    build_numerical_operators(state_3d_small)
    assert state_3d_small.ppe._A is not None
    assert state_3d_small.ppe._A.shape == (27, 27)
    assert isinstance(state_3d_small.ppe._A, sp.csr_matrix)

def test_scientific_operators_format_and_stack(state_3d_small, capsys):
    """Rule 2.6: Verify Matrix formats (CSR) and Global Gradient Stacking."""
    build_numerical_operators(state_3d_small)
    captured = capsys.readouterr().out
    
    assert isinstance(state_3d_small.operators.grad_x, sp.csr_matrix)
    assert isinstance(state_3d_small.operators.divergence, sp.csr_matrix)
    assert "Global Gradient V-Stack shape: (108, 27)" in captured
    assert "!!! CRITICAL" not in captured

def test_scientific_operator_orthogonality(state_3d_small):
    """Rule 2.7: Verify Gy and Gz coefficients (Dimensional separation)."""
    build_numerical_operators(state_3d_small)
    Gy = state_3d_small.operators.Gy if hasattr(state_3d_small.operators, 'Gy') else state_3d_small.operators.grad_y
    Gz = state_3d_small.operators.Gz if hasattr(state_3d_small.operators, 'Gz') else state_3d_small.operators.grad_z
    
    assert Gy[16, 13] == pytest.approx(10.0)
    assert Gy[16, 10] == pytest.approx(-10.0)
    assert Gz[13, 13] == pytest.approx(10.0)
    assert Gz[13, 4] == pytest.approx(-10.0)

def test_scientific_laplacian_symmetry(state_3d_small):
    """Rule 2.8: Verify L is symmetric (essential for CG solvers)."""
    build_numerical_operators(state_3d_small)
    L_sub = state_3d_small.operators.laplacian[1:, 1:]
    diff = (L_sub - L_sub.T)
    assert diff.nnz == 0 or np.allclose(diff.data, 0, atol=1e-10)

def test_scientific_laplacian_conservation(state_3d_small):
    """Rule 2.9: Every row of L must sum to 0 (Compatibility Condition)."""
    build_numerical_operators(state_3d_small)
    row_sums = np.array(state_3d_small.operators.laplacian.sum(axis=1)).flatten()[1:]
    assert np.allclose(row_sums, 0, atol=1e-10)