import numpy as np

from src.solver_state import SolverState
from src.step2.advection import build_advection_stencils


def test_scientific_advection_dof_handshake(state_3d_small, capsys):
    """Rule 2.1: Verify total DOF count and Debug Handshake prints."""
    build_advection_stencils(state_3d_small)
    captured = capsys.readouterr().out
    
    assert "Mapping 108 DOFs" in captured
    assert "Handshake check - processed 108/108 DOFs" in captured
    assert state_3d_small.advection.indices.shape == (108, 8)
    assert state_3d_small.advection.weights.shape == (108, 8)

def test_scientific_advection_trilinear_indices(state_3d_small):
    """Rule 2.2: Verify 8-point stencil mapping for an internal U-node."""
    build_advection_stencils(state_3d_small)
    indices = state_3d_small.advection.indices
    
    first_u_stencil = indices[0]
    assert len(np.unique(first_u_stencil)) <= 8
    assert indices.max() < 27, "Indices must not exceed total P-cells (27)"

def test_scientific_advection_weight_consistency(state_3d_small):
    """Rule 2.3: Ensure weights follow the SSoT config and are double precision."""
    build_advection_stencils(state_3d_small)
    weights = state_3d_small.advection.weights
    
    assert weights.dtype == np.float64
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
    assert np.all(weights == 0.125)

def test_scientific_advection_clamping_boundary(state_3d_small):
    """Rule 2.4: Boundary faces must clamp to valid P-indices."""
    # Custom 10x10x10 grid
    n = 10
    state_3d_small.grid._set_safe("nx", n, int)
    state_3d_small.grid._set_safe("ny", n, int)
    state_3d_small.grid._set_safe("nz", n, int)
    build_advection_stencils(state_3d_small)
    
    indices = state_3d_small.advection.indices
    assert indices.max() == 999
    assert indices.min() == 0

def test_scientific_advection_buffer_integrity():
    state = SolverState()
    # Manual setup since this bypasses the conftest fixture
    state.grid._set_safe("nx", 2, int)
    state.grid._set_safe("ny", 2, int)
    state.grid._set_safe("nz", 2, int)
    state.config._simulation_parameters = {"advection_weight_base": 0.0}
    
    build_advection_stencils(state)
    np.testing.assert_array_equal(state.advection.weights, 0.0)

def test_scientific_advection_full_debug_cycle(state_3d_small, capsys):
    """Rule 2.1: Verify specific sequence in DEBUG prints."""
    build_advection_stencils(state_3d_small)
    captured = capsys.readouterr().out
    
    assert "Mapping 108 DOFs (U:36, V:36, W:36)" in captured
    assert "Indices Min/Max: 0/26" in captured
    assert "Weights check sum: 1.0000" in captured

def test_scientific_advection_high_precision_ssot(state_3d_small):
    high_precision_val = 0.123456789012345
    state_3d_small.config._simulation_parameters["advection_weight_base"] = high_precision_val
    
    build_advection_stencils(state_3d_small)
    assert state_3d_small.advection.weights[0, 0] == high_precision_val

def test_scientific_advection_internal_stencil_uniqueness(state_3d_small):
    """Scientific check: Internal stencil must point to 8 distinct neighbors."""
    build_advection_stencils(state_3d_small)
    internal_u_idx = 18
    unique_neighbors = np.unique(state_3d_small.advection.indices[internal_u_idx])
    
    assert len(unique_neighbors) == 8, f"Stencil collapsed! Unique neighbors: {unique_neighbors}"

def test_scientific_advection_ssot_propagation(state_3d_small):
    """Rule 2.6: Verify exact weight propagation."""
    test_val = 0.0625
    state_3d_small.config._simulation_parameters["advection_weight_base"] = test_val
    build_advection_stencils(state_3d_small)
    
    assert np.allclose(state_3d_small.advection.weights, test_val)
    assert state_3d_small.config._simulation_parameters["advection_weight_base"] == test_val

def test_scientific_advection_weights_sum(state_3d_small):
    """Rule 7: Interpolation weights must sum to 1.0."""
    state_3d_small.config._simulation_parameters["advection_weight_base"] = 0.125
    build_advection_stencils(state_3d_small)
    
    row_sums = np.sum(state_3d_small.advection.weights, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12)

def test_scientific_advection_index_bounds(state_3d_small):
    """Rule 7: Stencil indices must never exceed the pressure grid boundaries."""
    # Already set to 3x3x3 via conftest fixture
    build_advection_stencils(state_3d_small)
    
    indices = state_3d_small.advection.indices
    assert indices.max() < 27
    assert indices.min() >= 0