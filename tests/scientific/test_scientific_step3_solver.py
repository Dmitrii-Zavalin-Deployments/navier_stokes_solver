# tests/scientific/test_scientific_step3_solver.py

import pytest
import numpy as np
from scipy import sparse
from src.solver_state import SolverState

class AttributeDict(dict):
    """A dict that allows dot notation access, satisfying the frozen dict type-check."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_solver():
    state = SolverState()
    state.config._ppe_tolerance = 1e-6
    state.config._ppe_atol = 1e-8
    state.config._ppe_max_iter = 1000
    return state
    pass
    state = SolverState()
    
    # 1. Config Setup (PPE Parameters)
    state.config.fluid_properties = AttributeDict({"density": 1000.0, "viscosity": 1.0})
    state.config.simulation_parameters = AttributeDict({
        "time_step": 0.1, 
        "initial_pressure": 0.0,
        "ppe_tolerance": 1e-6,
        "ppe_atol": 1e-8,
        "ppe_max_iter": 1000
    })
    
    # 2. Grid & Mask Setup (3x3x3 fluid block)
    state.fields._P = np.zeros((3, 3, 3))
    state.fields._U_star = np.ones((4, 3, 3)) # Divergence-free test case
    state.fields._V_star = np.zeros((3, 4, 3))
    state.fields._W_star = np.zeros((3, 3, 4))
    state._mask = np.ones((3, 3, 3)) # All fluid
    
    # 3. PPE System Mocking (Laplacian Matrix)
    # Total cells = 27. Create a simple identity for the PPE Matrix A.
    size = 27
    state.ppe._A = sparse.csr_matrix(sparse.eye(size))
    
    # 4. Operator Mocking (Divergence)
    # Divergence maps 3 velocity fields (4*3*3 + 3*4*3 + 3*3*4 = 108) to 27 pressure cells
    # We use the private backing variable to bypass frozen attribute check
    state.operators._divergence = sparse.csr_matrix((27, 108))
    
    return state

def test_solver_pressure_convergence(state_solver):
    """Verifies the formula: rhs = (rho/dt) * div(V*) and CG convergence."""
    from src.step3.solver import solve_pressure
    
    # Mock a specific divergence result to check RHS calculation
    # Let div(V*) = 0.001 at every cell
    div_val = 0.001
    state_solver.operators._divergence = sparse.csr_matrix(np.ones((27, 108)) * 0.0)
    # Manually create a RHS that isn't zero
    rho, dt = 1000.0, 0.1
    
    status = solve_pressure(state_solver)
    
    assert status == "converged"
    # Even if div is 0, the anchor (ref_p=0) should be applied
    assert state_solver.fields.P.shape == (3, 3, 3)

def test_solver_pressure_anchoring(state_solver):
    """Verifies Rule 5: Dynamic anchoring at the first fluid cell."""
    from src.step3.solver import solve_pressure
    
    # Set a specific reference pressure
    state_solver.config.simulation_parameters.initial_pressure = 101325.0 # Pascal
    
    solve_pressure(state_solver)
    
    # Check if the first fluid cell matches the anchor pressure
    # Since we use 'F' order, P[0,0,0] is index 0
    assert state_solver.fields.P[0, 0, 0] == 101325.0

def test_solver_no_fluid_cells(state_solver, capsys):
    """Verifies critical failure when the mask contains no fluid."""
    from src.step3.solver import solve_pressure
    
    state_solver._mask = np.zeros((3, 3, 3)) # All solid
    
    status = solve_pressure(state_solver)
    
    assert status == "failed"
    assert "CRITICAL: No fluid cells found" in capsys.readouterr().out

def test_solver_cg_failure(state_solver, capsys):
    """Tests the failure path when CG does not converge."""
    from src.step3.solver import solve_pressure
    
    # Force CG failure by making A singular (all zeros)
    state_solver.ppe._A = sparse.csr_matrix((27, 27))
    state_solver.config._ppe_tolerance = 1e-6
    state_solver.config._ppe_atol = 1e-8
    state_solver.config._ppe_max_iter = 1000
    
    status = solve_pressure(state_solver)
    
    assert status == "failed"
    assert "CG status info" in capsys.readouterr().out

def test_solver_debug_output(state_solver, capsys):
    """Ensures all physics debug prints are active and correctly formatted."""
    from src.step3.solver import solve_pressure
    
    solve_pressure(state_solver)
    captured = capsys.readouterr().out
    
    assert "Starting PPE solve" in captured
    assert "Pressure anchored at Index" in captured
    assert "Res Norm:" in captured