# tests/scientific/test_scientific_step3_corrector.py

import pytest
import numpy as np
import scipy.sparse as sp
from src.step3.corrector import correct_velocity
from src.step2.operators import build_numerical_operators
from src.solver_state import SolverState

@pytest.fixture
def state_corrector():
    """Setup a 3D state with pre-built operators and dummy U_star."""
    state = SolverState()
    from src.solver_input import FluidInput
    state.config._fluid_properties = FluidInput()
    from src.solver_input import SimParamsInput
    state.config._simulation_parameters = SimParamsInput()
    # 3x3x3 Grid
    state.grid.nx, state.grid.ny, state.grid.nz = 3, 3, 3
    state.grid.x_min, state.grid.x_max = 0.0, 0.3
    state.grid.y_min, state.grid.y_max = 0.0, 0.3
    state.grid.z_min, state.grid.z_max = 0.0, 0.3
    
    # Physics
    state.config.fluid_properties.density = 1000.0
    state.config.simulation_parameters.time_step = 0.01 # coeff = 1e-5
    
    # Initialize operators
    build_numerical_operators(state)
    
    # Initialize Fields with zeros (Fortran order as per Rule 5)
    state.fields.U_star = np.zeros((4, 3, 3), order='F')
    state.fields.V_star = np.zeros((3, 4, 3), order='F')
    state.fields.W_star = np.zeros((3, 3, 4), order='F')
    state.fields.P = np.zeros((3, 3, 3), order='F')
    
    return state

def test_scientific_corrector_math_identity(state_corrector, capsys):
    """Rule 3.1: Verify V_new = V* - (dt/rho)*grad(P) logic and coefficient scaling."""
    # Set a pressure gradient: P increases linearly in x
    # P[i,j,k] = i * 100.0 -> grad_x should be 100 / dx = 100 / 0.1 = 1000
    for i in range(3):
        state_corrector.fields.P[i, :, :] = i * 100.0
    
    # coeff = 0.01 / 1000.0 = 1e-5
    # Expected U correction = - (1e-5 * 1000) = -0.01
    correct_velocity(state_corrector)
    captured = capsys.readouterr().out
    
    # Verify debug output
    assert "Applying correction with coeff (dt/rho)=1.000000e-05" in captured
    assert "Max Grad P: 1.0000e+03" in captured
    
    # Verify U field update at an internal node
    # U[1,1,1] should be -0.01
    assert state_corrector.fields.U[1, 1, 1] == pytest.approx(-0.01)

def test_scientific_corrector_divergence_update(state_corrector):
    """Rule 3.2: Verify Health Vitals update with post-correction divergence."""
    # Start with a divergent U_star field
    # U_star[1,0,0] = 1.0, others 0 -> creates divergence at P[0,0,0]
    state_corrector.fields.U_star[1, 1, 1] = 1.0
    
    correct_velocity(state_corrector)
    
    # Since P is zero, U should equal U_star
    assert state_corrector.fields.U[1, 1, 1] == 1.0
    
    # Divergence norm should be non-zero
    assert state_corrector.health.divergence_norm > 0
    assert state_corrector.health.post_correction_divergence_norm == state_corrector.health.divergence_norm

def test_scientific_corrector_health_vitals_max_u(state_corrector):
    """Rule 3.3: Verify Max Velocity tracking across all components."""
    state_corrector.fields.U_star[0, 0, 0] = 5.0
    state_corrector.fields.V_star[0, 0, 0] = -10.0
    state_corrector.fields.W_star[0, 0, 0] = 2.0
    
    correct_velocity(state_corrector)
    
    # Max absolute velocity should be 10.0
    assert state_corrector.health.max_u == 10.0

def test_scientific_corrector_logic_gate_warning(state_corrector, capsys):
    """Rule 3.4: Verify the logic gate warning for high divergence."""
    # Force a massive divergence that exceeds 1e-10
    state_corrector.fields.U_star[1, 1, 1] = 1e6
    correct_velocity(state_corrector)
    
    captured = capsys.readouterr().out
    assert "!!! WARNING: Divergence above logic gate threshold !!!" in captured