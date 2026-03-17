# tests/step3/test_pressure_and_bc_integrity.py

import numpy as np
from scipy.sparse import eye

from src.step3.solve_pressure import solve_pressure
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_ppe_solid_pressure_zeroing():
    """
    Verify solve_pressure respects solid masks and produces non-zero fluid pressure.
    Ensures that cells marked as solid are anchored to 0.0.
    """
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # NUMERICAL FIX: Provide a valid diagonal for the Laplacian matrix A.
    # Without this, the Jacobi preconditioner in solve_pressure hits a divide-by-zero.
    size = nx * ny * nz
    state.ppe["A"] = eye(size, format="csr") * -6.0 

    state.is_fluid.fill(True)
    state.is_fluid[1, 1, 1] = False # Create a solid island in the center
    
    # Balanced RHS (sum=0) is required for stable convergence in Neumann systems.
    rhs_ppe = np.zeros((nx, ny, nz))
    rhs_ppe[0, 0, 0] = 1.0
    rhs_ppe[2, 2, 2] = -1.0
    
    P_new, metadata = solve_pressure(state, rhs_ppe)

    # Verification 1: Solid cell must be exactly zeroed post-solve
    assert P_new[1, 1, 1] == 0.0, "Pressure at solid cell was not zeroed."
    
    # Verification 2: Fluid cells must contain non-zero results (the solver actually worked)
    assert np.any(P_new[state.is_fluid] != 0.0), "Pressure solver returned all zeros."

def test_ppe_mean_subtraction_logic():
    """
    Ensures that for singular systems (pure Neumann), the pressure field 
    is normalized so the mean is zero.
    """
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.ppe["A"] = eye(nx*ny*nz, format="csr") * -6.0
    state.is_fluid.fill(True)
    state.ppe["ppe_is_singular"] = True
    
    # Balanced RHS
    rhs_ppe = np.zeros((3, 3, 3))
    rhs_ppe[0, 0, 0] = 1.0
    rhs_ppe[-1, -1, -1] = -1.0
    
    P_new, _ = solve_pressure(state, rhs_ppe)
    
    # The mean of the pressure field should be effectively zero
    assert np.isclose(np.mean(P_new), 0.0, atol=1e-10)

def test_ppe_solver_metadata_contract():
    """
    Verify the solver returns the expected metadata keys for orchestration logging.
    """
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    state.ppe["A"] = eye(8, format="csr")
    rhs_ppe = np.zeros((2, 2, 2))
    
    _, metadata = solve_pressure(state, rhs_ppe)
    
    required_keys = ["converged", "solver_status", "method", "tolerance_used"]
    for key in required_keys:
        assert key in metadata, f"Metadata key '{key}' missing from solve_pressure output"