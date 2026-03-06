# tests/scientific/test_scientific_step2_operators.py

import pytest
import numpy as np
import scipy.sparse as sp
from src.step2.operators import build_numerical_operators
# No local fixture needed; assumes conftest.py provides state_3d_small


def test_scientific_operators_dof_handshake(state_3d_small, capsys):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.1: Verify DOF mapping and Debug Handshake prints."""
    # P: 3*3*3 = 27
    # U: 4*3*3 = 36
    # V: 3*4*3 = 36
    # W: 3*3*4 = 36
    # Total Vel DOF: 36+36+36 = 108
    build_numerical_operators(state_3d_small)
    captured = capsys.readouterr().out
    
    assert "Target DOFs - P:27, U:36, V:36, W:36" in captured
    assert state_3d_small.operators.divergence.shape == (27, 108)
    assert state_3d_small.operators.grad_x.shape == (36, 27)

def test_scientific_gradient_coefficients(state_3d_small):
    # Ensure 0.3 / 3 = 0.1 spacing
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    state_3d_small.grid.x_max = state_3d_small.grid.y_max = state_3d_small.grid.z_max = 0.3
    
    """Rule 2.2: Verify finite difference coefficients (1/dx) in Gradient."""
    build_numerical_operators(state_3d_small)
    Gx = state_3d_small.operators.grad_x
    
    # For dx=0.1, coefficients should be 10.0 and -10.0
    # Check an internal node: Gx[idx_u, idx_p]
    # Pick i=1, j=1, k=1 -> idx_u = 1 + 1*4 + 1*4*3 = 17
    # P cells: (1,1,1) -> 13 and (0,1,1) -> 12
    assert Gx[17, 13] == pytest.approx(10.0)
    assert Gx[17, 12] == pytest.approx(-10.0)

def test_scientific_divergence_nullspace(state_3d_small):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.3: Divergence of a constant velocity field must be zero."""
    build_numerical_operators(state_3d_small)
    D = state_3d_small.operators.divergence
    
    # Create constant velocity field [1.0, 1.0, 1.0]
    total_dofs = D.shape[1]
    v_const = np.ones(total_dofs)
    
    div_v = D @ v_const
    # Only internal cells are strictly zero due to boundary faces
    # For a 3x3x3, the center cell is (1,1,1) -> index 13
    assert div_v[13] == pytest.approx(0.0)

def test_scientific_composite_laplacian(state_3d_small, capsys):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.4: Verify L = D @ G composition and non-emptiness."""
    build_numerical_operators(state_3d_small)
    captured = capsys.readouterr().out
    
    L = state_3d_small.operators.laplacian
    assert "Laplacian nnz:" in captured
    assert L.nnz > 0
    assert L.shape == (27, 27)
    
    # Check the standard 7-point stencil value for a center cell
    # L_ii = - (2/dx^2 + 2/dy^2 + 2/dz^2)
    # For dx=dy=dz=0.1, L_ii = -(200 + 200 + 200) = -600
    center_idx = 13
    assert L[center_idx, center_idx] == pytest.approx(-600.0)

def test_scientific_ppe_state_transfer(state_3d_small):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.5: Ensure Laplacian is correctly committed to PPE solver state."""
    build_numerical_operators(state_3d_small)
    assert state_3d_small.ppe._A is not None
    assert state_3d_small.ppe._A.shape == (27, 27)
    # Ensure it is a CSR matrix for performance
    assert isinstance(state_3d_small.ppe._A, sp.csr_matrix)
# --- FINAL SCIENTIFIC AUDIT: SPARSITY & STACKING ---

def test_scientific_operators_format_and_stack(state_3d_small, capsys):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.6: Verify Matrix formats (CSR) and Global Gradient Stacking."""
    build_numerical_operators(state_3d_small)
    captured = capsys.readouterr().out
    
    # 1. Verify CSR Conversion (Crucial for solver performance)
    assert isinstance(state_3d_small.operators.grad_x, sp.csr_matrix)
    assert isinstance(state_3d_small.operators.divergence, sp.csr_matrix)
    
    # 2. Verify V-Stack Shape in Debug
    # Total Vel DOFs = 36+36+36 = 108. P DOFs = 27.
    assert "Global Gradient V-Stack shape: (108, 27)" in captured
    
    # 3. Verify the 'Critical' firewall was NOT triggered
    assert "!!! CRITICAL" not in captured

def test_scientific_operator_orthogonality(state_3d_small):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.7: Verify Gy and Gz coefficients (Dimensional separation)."""
    build_numerical_operators(state_3d_small)
    Gy = state_3d_small.operators.Gy if hasattr(state_3d_small.operators, 'Gy') else state_3d_small.operators.grad_y
    Gz = state_3d_small.operators.Gz if hasattr(state_3d_small.operators, 'Gz') else state_3d_small.operators.grad_z
    
    # Check Gy for v-face at center (i=1, j=1, k=1) 
    # idx_v = i + j*nx + k*nx*(ny+1) = 1 + 1*3 + 1*3*4 = 16
    assert Gy[16, 13] == pytest.approx(10.0) # P(1,1,1)
    assert Gy[16, 10] == pytest.approx(-10.0) # P(1,0,1)

    # Check Gz for w-face at center (i=1, j=1, k=1)
    # idx_w = i + j*nx + k*nx*ny = 1 + 1*3 + 1*9 = 13
    assert Gz[13, 13] == pytest.approx(10.0) # P(1,1,1)
    assert Gz[13, 4] == pytest.approx(-10.0) # P(1,1,0)

def test_scientific_laplacian_symmetry(state_3d_small):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.8: Verify L is symmetric (essential for CG solvers)."""
    build_numerical_operators(state_3d_small)
    L = state_3d_small.operators.laplacian
    
    # Check a few off-diagonal elements
    # If L[i, j] exists, L[j, i] must be identical
    # Symmetry is broken at index 0 due to pinning. Check submatrix [1:, 1:]
    L_sub = L[1:, 1:]
    diff = (L_sub - L_sub.T)
    # Check the norm of the difference
    assert diff.nnz == 0 or np.allclose(diff.data, 0, atol=1e-10), "Laplacian is not symmetric!"

def test_scientific_laplacian_conservation(state_3d_small):
    state_3d_small.grid.nx = state_3d_small.grid.ny = state_3d_small.grid.nz = 3
    """Rule 2.9: Every row of L must sum to 0 (Compatibility Condition)."""
    build_numerical_operators(state_3d_small)
    L = state_3d_small.operators.laplacian
    
    # Sum across columns for each row
    row_sums = np.array(L.sum(axis=1)).flatten()[1:]
    
    # In a pure Neumann setup, all rows sum to 0.
    # Note: If you have a Dirichlet point for pressure, one row will be different.
    # For now, we check if the internal nodes follow conservation.
    assert np.allclose(row_sums, 0, atol=1e-10), f"Laplacian rows do not sum to zero: {row_sums}"