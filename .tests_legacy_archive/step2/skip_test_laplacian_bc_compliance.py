# tests/step2/test_laplacian_bc_compliance.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators


@pytest.fixture
def state_with_bcs():
    """
    Setup a small 4x4x4 grid with specific BC types.
    Aligns with the 'Matrix Debt' theory where pressure is Dirichlet 
    and walls/inflows are Neumann.
    """
    grid = {"nx": 4, "ny": 4, "nz": 4, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    # x_min is a Wall (Neumann), x_max is Pressure (Dirichlet)
    bcs = {
        "x_min": {"type": "no-slip", "p": 0.0},
        "x_max": {"type": "pressure", "p": 10.0}
    }
    state = SolverState(grid=grid, boundary_conditions=bcs)
    # Ensure fluid mask is fully initialized so the builder doesn't skip cells
    state.is_fluid = np.ones((4, 4, 4), dtype=bool)
    return state

class TestLaplacianBCCompliance:
    """
    Verifies 100% implementation of 'The Matrix Debt' in Step 2.
    Checks that the Laplacian operator A is modified according to BC type.
    """

    def test_dirichlet_pressure_matrix_row(self, state_with_bcs):
        """
        Theory Check: For Pressure-type BCs: Row is identity-like (A[i,i] = 1, others = 0).
        """
        nx, ny = 4, 4
        # Execute operator construction
        build_laplacian_operators(state_with_bcs)
        A = state_with_bcs.operators["laplacian"]
        
        # Test a cell on the x_max face (i=3, j=2, k=2)
        # Flat index: 3 + 2*4 + 2*16 = 3 + 8 + 32 = 43
        idx = 3 + (2 * nx) + (2 * nx * ny)
        
        row = A.getrow(idx).toarray().flatten()
        
        # Identity check
        assert row[idx] == 1.0, f"Dirichlet diagonal at {idx} should be 1.0"
        assert np.count_nonzero(row) == 1, f"Dirichlet row at {idx} must be identity-like"

    def test_neumann_wall_matrix_modification(self, state_with_bcs):
        """
        Theory Check: For Wall-type BCs: Neumann condition reduces the stencil.
        One missing neighbor at the boundary should change the center from -6.0 to -5.0.
        """
        nx, ny = 4, 4
        build_laplacian_operators(state_with_bcs)
        A = state_with_bcs.operators["laplacian"]
        
        # Test a cell on the x_min face (i=0, j=2, k=2)
        # Flat index: 0 + 2*4 + 2*16 = 40
        idx = 0 + (2 * nx) + (2 * nx * ny)
        
        row = A.getrow(idx).toarray().flatten()
        
        # Neumann check: center coefficient should be -5.0 because dx=dy=dz=1.0 
        # and the neighbor at i-1 is out of bounds/ignored.
        assert row[idx] == -5.0, f"Neumann boundary at {idx} should have center -5.0, got {row[idx]}"
        
        # Verify neighbor at i+1 (index 41) still has its weight of 1.0
        assert row[idx + 1] == 1.0, "Neighbor i+1 should still exist in Neumann stencil"