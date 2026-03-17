# tests/step3/test_step3_boundary_integration.py

import numpy as np

from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_mock_operators(state):
    """Inject mock gradient operators with staggered output shapes."""
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    state.operators["grad_x"] = lambda p: np.zeros((nx + 1, ny, nz))
    state.operators["grad_y"] = lambda p: np.zeros((nx, ny + 1, nz))
    state.operators["grad_z"] = lambda p: np.zeros((nx, ny, nz + 1))

def test_bc_precedence_inflow_vs_solid():
    """PROVE: Internal Solid Masks take priority over Domain Inflow."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants = {"rho": 1.0}
    state.config["dt"] = 0.01
    setup_mock_operators(state)
    
    state.config["boundary_conditions"] = [{"location": "x_min", "type": "inflow", "values": {"u": 5.0}}]
    state.is_fluid.fill(True)
    state.is_fluid[0, 0, 0] = False # Solid at the inflow boundary
    
    U_star = np.ones((4, 3, 3))
    V_star = np.zeros((3, 4, 3))
    W_star = np.zeros((3, 3, 4))
    P_new = np.zeros((3, 3, 3))

    U_c, V_c, W_c = correct_velocity(state, U_star, V_star, W_star, P_new)
    fields_final = apply_boundary_conditions_post(state, U_c, V_c, W_c, P_new)

    assert fields_final["U"][0, 0, 0] == 0.0, "Solid mask must override Inflow velocity"

def test_projection_reduces_divergence_logic():
    """PROVE: correct_velocity uses grad_x to modify U_star with correct shapes."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants = {"rho": 1.0}
    state.config["dt"] = 1.0
    state.is_fluid.fill(True)
    
    state.operators["grad_x"] = lambda p: np.ones((4, 3, 3))
    state.operators["grad_y"] = lambda p: np.zeros((3, 4, 3))
    state.operators["grad_z"] = lambda p: np.zeros((3, 3, 4))

    U_star, V_star, W_star = np.ones((4, 3, 3)), np.zeros((3, 4, 3)), np.zeros((3, 3, 4))
    P = np.zeros((3, 3, 3))
    
    U_new, _, _ = correct_velocity(state, U_star, V_star, W_star, P)
    assert np.all(U_new[1:-1, :, :] == 0.0)

def test_solid_mask_indexing_consistency():
    """PROVE: apply_boundary_conditions_post handles staggered shapes without IndexError."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)
    state.is_fluid[:, :, 1] = False 
    
    U, V, W = np.ones((4,3,3)), np.ones((3,4,3)), np.ones((3,3,4))
    P = np.zeros((3,3,3))
    
    fields = apply_boundary_conditions_post(state, U, V, W, P)
    assert np.all(fields["W"][:, :, 1] == 0.0) # Lower face of solid cell
    assert np.all(fields["W"][:, :, 2] == 0.0) # Upper face of solid cell