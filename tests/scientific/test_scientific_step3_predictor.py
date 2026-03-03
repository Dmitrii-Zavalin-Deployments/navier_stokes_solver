import pytest
import numpy as np
from scipy import sparse
from src.solver_state import SolverState

@pytest.fixture
def state_predictor():
    """Fixture to set up a valid state for full predictor physics."""
    state = SolverState()
    
    # 1. Physics Setup
    state.config._fluid_properties.density = 1000.0
    state.config._fluid_properties.viscosity = 1.0  # nu = 0.001
    state.config._simulation_parameters.time_step = 0.1
    state.config.external_forces.force_vector = [1.0, 0.0, 0.0]
    
    # 2. Field Allocation (3x3x3 grid)
    # U: (4,3,3) = 36, V: (3,4,3) = 36, W: (3,3,4) = 36
    state.fields._U = np.ones((4, 3, 3)) 
    state.fields._V = np.zeros((3, 4, 3))
    state.fields._W = np.zeros((3, 3, 4))
    
    # 3. Operator Mocking (Using Identity for math verification)
    # Laplacian and Advection operators must exist for U, V, and W
    eye36 = sparse.eye(36, 36)
    state.operators._laplacian = eye36
    state.operators._advection_u = eye36
    state.operators._advection_v = eye36
    state.operators._advection_w = eye36
    
    return state

def test_predict_velocity_full_physics(state_predictor):
    """
    Verify U* = U + dt * (nu * Lap(U) - Adv(U) + force_u)
    Initial U = 1.0, dt = 0.1, nu = 0.001, Lap(U) = 1.0, Adv(U) = 1.0, force_u = 1.0
    Expected: 1.0 + 0.1 * (0.001 * 1.0 - 1.0 + 1.0) = 1.0001
    """
    from src.step3.predictor import predict_velocity
    predict_velocity(state_predictor)
    
    assert np.allclose(state_predictor.fields.U_star, 1.0001)

def test_predict_velocity_missing_advection(state_predictor):
    """Ensure security guard catches missing advection operators."""
    from src.step3.predictor import predict_velocity
    state_predictor.operators._advection_u = None
    
    with pytest.raises(RuntimeError, match="Access Error: 'advection_u'"):
        predict_velocity(state_predictor)

def test_predict_velocity_instability_debug(state_predictor, capsys):
    """Verify that the CRITICAL instability message triggers on Inf/NaN."""
    from src.step3.predictor import predict_velocity
    state_predictor.fields.U[0,0,0] = np.inf
    
    predict_velocity(state_predictor)
    captured = capsys.readouterr()
    assert "CRITICAL: Predictor Instability" in captured.out

def test_predict_velocity_component_isolation(state_predictor):
    """
    Production Check: Ensure U, V, and W use their own force indices and operators.
    If we use force_vector = [1, 2, 3], components should react uniquely.
    """
    from src.step3.predictor import predict_velocity
    state_predictor.config.external_forces.force_vector = [1.0, 2.0, 3.0]
    
    predict_velocity(state_predictor)
    
    # U calculation (from previous test): 1.0001
    # V calculation: 0.0 + 0.1 * (0.001 * 0 - 0 + 2.0) = 0.2
    # W calculation: 0.0 + 0.1 * (0.001 * 0 - 0 + 3.0) = 0.3
    assert np.allclose(state_predictor.fields.U_star, 1.0001)
    assert np.allclose(state_predictor.fields.V_star, 0.2)
    assert np.allclose(state_predictor.fields.W_star, 0.3)

## =========================================================
## RESTORED COMPONENT & DEBUG VALIDATION
## =========================================================

def test_predict_velocity_v_w_components(state_predictor):
    """
    Production Check: Verify V and W use correct force indices and operators.
    Calculations:
    V_star = 0 + 0.1 * (0.001 * 0 - 0 + 5.0) = 0.5
    W_star = 0 + 0.1 * (0.001 * 0 - 0 + 10.0) = 1.0
    """
    from src.step3.predictor import predict_velocity
    # Set unique forces for Y and Z
    state_predictor.config.external_forces.force_vector = [0.0, 5.0, 10.0]
    
    predict_velocity(state_predictor)
    
    assert np.allclose(state_predictor.fields.V_star, 0.5), "V_star update failed or used wrong index"
    assert np.allclose(state_predictor.fields.W_star, 1.0), "W_star update failed or used wrong index"

def test_predict_velocity_nu_debug_format(state_predictor, capsys):
    """
    UI Check: Ensure the 'Nu' and 'dt' debug prints maintain their required precision.
    Formula: Nu = 1.0 / 1000.0 = 1.000000e-03
    """
    from src.step3.predictor import predict_velocity
    predict_velocity(state_predictor)
    
    captured = capsys.readouterr()
    # Verifies scientific notation formatting in the logs
    assert "Nu=1.000000e-03" in captured.out
    assert "dt=0.1" in captured.out
