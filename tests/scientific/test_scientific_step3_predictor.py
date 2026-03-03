import pytest
import numpy as np
from scipy import sparse
from src.solver_state import SolverState

@pytest.fixture
def state_predictor():
    """
    Fixture to set up a valid state for full predictor physics.
    Matches frozen logic in solver_state.py and solver_input.py.
    """
    state = SolverState()
    
    # 1. Physics Setup (Using dictionaries as expected by SolverConfig properties)
    state.config.fluid_properties = {
        "density": 1000.0, 
        "viscosity": 1.0
    }
    state.config.simulation_parameters = {
        "time_step": 0.1, 
        "total_time": 1.0, 
        "output_interval": 1
    }
    state.config.external_forces = {
        "force_vector": [1.0, 0.0, 0.0]
    }
    
    # 2. Field Allocation (3x3x3 grid)
    # Staggered grid components: U(4,3,3), V(3,4,3), W(3,3,4)
    state.fields._U = np.ones((4, 3, 3)) 
    state.fields._V = np.zeros((3, 4, 3))
    state.fields._W = np.zeros((3, 3, 4))
    
    # 3. Operator Mocking (Using Identity for math verification)
    # Size 36 matches the flattened 3x3x3 grid components
    eye36 = sparse.eye(36, 36)
    state.operators._laplacian = eye36
    state.operators._advection_u = eye36
    state.operators._advection_v = eye36
    state.operators._advection_w = eye36
    
    return state

## =========================================================
## PHYSICS & FORMULA VERIFICATION
## =========================================================

def test_predict_velocity_full_physics(state_predictor):
    """
    Verify U* = U + dt * (nu * Lap(U) - Adv(U) + force_u)
    Initial U=1.0, dt=0.1, nu=0.001, Lap=1.0, Adv=1.0, Force=1.0
    Expected: 1.0 + 0.1 * (0.001 * 1.0 - 1.0 + 1.0) = 1.0001
    """
    from src.step3.predictor import predict_velocity
    predict_velocity(state_predictor)
    
    assert np.allclose(state_predictor.fields.U_star, 1.0001)

def test_predict_velocity_missing_operator(state_predictor):
    """Ensures security guard catches missing advection operators."""
    from src.step3.predictor import predict_velocity
    state_predictor.operators._advection_u = None
    
    # Accessing None via property triggers the Security Guard RuntimeError
    with pytest.raises(RuntimeError):
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
    """
    from src.step3.predictor import predict_velocity
    state_predictor.config.external_forces = {"force_vector": [1.0, 2.0, 3.0]}
    
    predict_velocity(state_predictor)
    
    # U calculation: 1.0 + 0.1 * (0.001*1 - 1 + 1) = 1.0001
    # V calculation: 0.0 + 0.1 * (0.001*0 - 0 + 2) = 0.2
    # W calculation: 0.0 + 0.1 * (0.001*0 - 0 + 3) = 0.3
    assert np.allclose(state_predictor.fields.U_star, 1.0001)
    assert np.allclose(state_predictor.fields.V_star, 0.2)
    assert np.allclose(state_predictor.fields.W_star, 0.3)

def test_predict_velocity_nu_debug_format(state_predictor, capsys):
    """
    UI Check: Ensure the 'Nu' and 'dt' debug prints maintain required precision.
    """
    from src.step3.predictor import predict_velocity
    predict_velocity(state_predictor)
    
    captured = capsys.readouterr()
    assert "Nu=1.000000e-03" in captured.out
    assert "dt=0.1" in captured.out

def test_predict_velocity_v_w_components_explicit(state_predictor):
    """
    Dedicated Production Check: Verify V and W components use correct force indices.
    V_star should use force_vector[1], W_star should use force_vector[2].
    Calculations:
    V_star = 0 + 0.1 * (0.001 * 0 - 0 + 5.0) = 0.5
    W_star = 0 + 0.1 * (0.001 * 0 - 0 + 10.0) = 1.0
    """
    from src.step3.predictor import predict_velocity
    # Set unique forces for Y and Z to ensure indexing is correct
    state_predictor.config.external_forces = {"force_vector": [0.0, 5.0, 10.0]}
    
    predict_velocity(state_predictor)
    
    assert np.allclose(state_predictor.fields.V_star, 0.5), "V_star update failed or used wrong force index"
    assert np.allclose(state_predictor.fields.W_star, 1.0), "W_star update failed or used wrong force index"
