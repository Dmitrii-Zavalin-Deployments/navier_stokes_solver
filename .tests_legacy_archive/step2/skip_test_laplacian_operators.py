# tests/step2/test_laplacian_operators.py

import numpy as np
from scipy.sparse import issparse

from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_state(nx=4, ny=4, nz=4, dx=1.0):
    """
    Create a canonical Step-1 dummy state for Laplacian tests.
    """
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)

    state.grid['dx'] = dx
    state.grid['dy'] = dx
    state.grid['dz'] = dx

    state.mask = np.ones((nx, ny, nz), dtype=int)
    
    # FIX: Initialize empty BCs to avoid AttributeError in build_laplacian_operators
    state.boundary_conditions = {}

    create_fluid_mask(state)

    return state


# ------------------------------------------------------------
# 1. Sparse Check: Verify operator is a CSR matrix
# ------------------------------------------------------------
def test_laplacian_is_sparse():
    state = make_state()
    build_laplacian_operators(state)
    
    lap = state.operators.get("laplacian")
    assert lap is not None
    assert issparse(lap)
    assert lap.shape == (64, 64) 


# ------------------------------------------------------------
# 2. Symmetry Check: Internal Laplacian logic
# ------------------------------------------------------------
def test_laplacian_symmetry():
    # Symmetry holds for interior cells. 
    # Neumann boundaries also maintain symmetry in this implementation.
    state = make_state(nx=3, ny=3, nz=3)
    build_laplacian_operators(state)
    
    lap = state.operators["laplacian"]
    diff = lap - lap.transpose()
    assert diff.nnz == 0 or np.allclose(diff.data, 0.0, atol=1e-10)


# ------------------------------------------------------------
# 3. Sum of Rows: Internal fluid cells sum to 0
# ------------------------------------------------------------
def test_laplacian_row_sum_internal():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz)
    build_laplacian_operators(state)
    
    lap = state.operators["laplacian"].toarray()
    
    # Index of an internal cell (1,1,1) -> no boundary contact
    idx = 1 + 1*nx + 1*nx*ny
    
    # Internal center (-6) cancels 6 neighbors (+1)
    assert np.isclose(np.sum(lap[idx, :]), 0.0, atol=1e-10)


# ------------------------------------------------------------
# 4. Solid Handling: Identity on diagonal
# ------------------------------------------------------------
def test_laplacian_solid_handling():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz)
    
    # Set one cell to solid
    state.mask[21] = 0
    create_fluid_mask(state)
    
    build_laplacian_operators(state)
    lap = state.operators["laplacian"].toarray()
    
    idx = 1 + 1*nx + 1*nx*ny
    # Solid cell row: A[i,i] = 1.0, others = 0.0
    assert lap[idx, idx] == 1.0
    assert np.isclose(np.sum(lap[idx, :]), 1.0, atol=1e-10)


# ------------------------------------------------------------
# 5. Diagonal Dominance
# ------------------------------------------------------------
def test_laplacian_diagonal_dominance():
    state = make_state(nx=3, ny=3, nz=3)
    build_laplacian_operators(state)
    lap = state.operators["laplacian"].toarray()
    
    for i in range(lap.shape[0]):
        diag = np.abs(lap[i, i])
        off_diag = np.sum(np.abs(lap[i, :])) - diag
        assert diag >= off_diag - 1e-10