# tests/step3/test_solve_pressure.py

import numpy as np
from scipy.sparse import eye

from src.step3.solve_pressure import solve_pressure
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_config_key_mapping_and_logic():
    """Verify solver correctly utilizes tolerance and atol from config."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    p_size = state.fields["P"].size
    
    # Setup Identity Matrix as mock Laplacian (Non-singular case)
    state.ppe["A"] = eye(p_size, format="csr")
    state.ppe["ppe_is_singular"] = False
    
    # Inject settings into config
    state.config["solver_settings"] = {
        "ppe_tolerance": 1e-9,
        "ppe_atol": 1e-11,
        "ppe_max_iter": 500
    }
    
    # Input RHS = 5.0 everywhere. For A*P = I*P = 5.0, P should be 5.0.
    rhs = np.full((2, 2, 2), 5.0)
    P_new, meta = solve_pressure(state, rhs)

    # Assertions for metadata and numerical correctness
    assert meta["tolerance_used"] == 1e-9
    assert meta["absolute_tolerance_used"] == 1e-11
    assert meta["method"] == "PCG (Jacobi)"
    assert np.allclose(P_new, 5.0)
    assert meta["converged"] is True
    assert meta["is_singular"] is False

def test_singular_mean_subtraction_physics():
    """Ensure that for singular systems, the mean pressure in fluid cells is zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    p_size = state.fields["P"].size
    state.ppe["A"] = eye(p_size, format="csr")
    state.ppe["ppe_is_singular"] = True
    
    # Setting a full fluid mask
    state.is_fluid = np.ones((3, 3, 3), dtype=bool)
    
    # RHS = 10.0 results in P = 10.0 initially, then 0.0 after mean subtraction
    rhs = np.full((3, 3, 3), 10.0)
    P_new, meta = solve_pressure(state, rhs)

    # Check that the fluid mean is zero (Mean-Subtraction Logic)
    assert abs(np.mean(P_new[state.is_fluid])) < 1e-12
    assert meta["is_singular"] is True

def test_fallback_tolerance_values():
    """Ensure solver uses robust defaults if config keys are completely missing."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    state.ppe["A"] = eye(state.fields["P"].size, format="csr")
    state.ppe["ppe_is_singular"] = False
    
    # Empty solver settings to trigger fallbacks in solve_pressure.py
    state.config["solver_settings"] = {}
    
    rhs = np.zeros((2, 2, 2))
    P_new, meta = solve_pressure(state, rhs)
    
    # These should match the hardcoded defaults in solve_pressure.py (1e-6 and 1e-12)
    assert meta["tolerance_used"] == 1e-6
    assert meta["absolute_tolerance_used"] == 1e-12
    assert meta["converged"] is True

def test_minimal_grid_no_crash():
    """Check stability on a 1x1x1 grid with Jacobi preconditioning."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    state.ppe["A"] = eye(1, format="csr")
    state.ppe["ppe_is_singular"] = False

    rhs = np.zeros((1, 1, 1))
    P_new, meta = solve_pressure(state, rhs)

    assert P_new.shape == (1, 1, 1)
    assert "converged" in meta
    assert meta["method"] == "PCG (Jacobi)"