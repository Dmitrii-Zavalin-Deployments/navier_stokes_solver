# tests/physics_gate/test_operators.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture
def laplacian_matrix():
    """Setup a standard 4x4x4 grid Laplacian for auditing."""
    raw_data = solver_input_schema_dummy()
    nx, ny, nz = 4, 4, 4
    
    # Manually assemble SolverState using the confirmed keys from your probe
    state = SolverState(
        config={
            'solver_type': 'projection', 
            'precision': 'float64',
            'simulation_parameters': raw_data.get('simulation_parameters', {})
        },
        grid={'nx': nx, 'ny': ny, 'nz': nz, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0},
        boundary_conditions=raw_data.get('boundary_conditions', [])
    )
    
    # Initialize fluid mask (all fluid for clean-room audit)
    state.is_fluid = np.ones(nx * ny * nz, dtype=bool)
    state.operators = {}
    
    build_laplacian_operators(state)
    return state.operators["laplacian"]

def test_gate_2a_calculus_identity(laplacian_matrix):
    """2.A: Identity Verification: Matrix dimensions for 4x4x4."""
    N = 64
    assert laplacian_matrix.shape == (N, N), "Matrix must be square NxN"

def test_gate_2b_symmetry_and_sparsity(laplacian_matrix):
    """2.B: Symmetry (L = L^T) and Sparsity (nnz <= 7N)."""
    # Check Symmetry
    diff = laplacian_matrix - laplacian_matrix.T
    assert diff.nnz == 0 or np.all(np.abs(diff.data) < 1e-12), "Matrix is not symmetric"
    
    # Check Sparsity
    N = laplacian_matrix.shape[0]
    assert laplacian_matrix.nnz <= 7 * N, f"Matrix too dense: {laplacian_matrix.nnz} non-zeros"

def test_gate_2c_null_space_audit(laplacian_matrix):
    """2.C: Null-Space Audit: Laplacian of constant field."""
    N = laplacian_matrix.shape[0]
    ones = np.ones(N)
    result = laplacian_matrix.dot(ones)
    
    # Rows must sum to 0.0 (Poisson) or 1.0 (BC Identity)
    for i, val in enumerate(result):
        assert np.isclose(val, 0.0) or np.isclose(val, 1.0), f"Null-space violation at row {i}: {val}"
