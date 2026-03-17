# tests/scientific/test_scientific_step3_solver.py

import numpy as np
import pytest
from scipy import sparse

from src.solver_state import SolverState


class AttributeDict(dict):
    """A dict that allows dot notation access, satisfying the frozen dict type-check."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_solver():
    state = SolverState()
    
    # 1. Config Setup: Bypassing _get_safe crash with AttributeDict
    # This allows state.config.simulation_parameters.initial_pressure to work
    state.config._fluid_properties = AttributeDict({
        "density": 1000.0, 
        "viscosity": 1e-3
    })
    
    state.config._simulation_parameters = AttributeDict({
        "time_step": 0.1, 
        "initial_pressure": 0.0,
        "ppe_tolerance": 1e-6,
        "ppe_atol": 1e-8,
        "ppe_max_iter": 1000
    })

    # Initialize numerical parameter slots directly for the solver's safety check
    state.config._ppe_tolerance = 1e-6
    state.config._ppe_atol = 1e-8
    state.config._ppe_max_iter = 1000
    
    # 2. Grid & Mask Setup (3x3x3 block = 27 cells)
    # Using 'F' order consistency as expected by the solver
    state.fields.U_star = np.zeros((3, 3, 3))
    state.fields.V_star = np.zeros((3, 3, 3))
    state.fields.W_star = np.zeros((3, 3, 3))
    state.fields.P = np.zeros((3, 3, 3))
    state._mask = np.ones((3, 3, 3)) # Default all fluid
    
    # 3. PPE System Mocking (Laplacian Matrix)
    # Total cells = 27. Identity matrix ensures easy convergence for testing.
    size = 27
    state.ppe._A = sparse.csr_matrix(sparse.eye(size))
    
    # 4. Operator Mocking (Divergence)
    # solver.py concatenates U, V, and W. 
    # For a 3x3x3 grid, that is 27 + 27 + 27 = 81 input elements.
    state.operators._divergence = sparse.csr_matrix((27, 81))
    
    return state

def test_solver_pressure_convergence(state_solver):
    from src.step3.solver import solve_pressure
    
    # We leave Cell 0 as the Anchor (it will be 0.0)
    # We put the divergence in Cell 1 (the neighbor)
    div_data = np.zeros((27, 81))
    div_data[1, 0] = 1.0  
    state_solver.operators._divergence = sparse.csr_matrix(div_data)
    
    state_solver.fields.U_star[0, 0, 0] = 1.0
    
    status = solve_pressure(state_solver)
    
    assert status == "converged"
    
    # Cell 1 should now show the calculated pressure (10,000)
    # P[1, 0, 0] is the 3D coordinate for flat index 1 in Fortran order
    assert state_solver.fields.P[1, 0, 0] == pytest.approx(10000.0)
    
    # Cell 0 should still be exactly 0.0 (the anchor)
    assert state_solver.fields.P[0, 0, 0] == 0.0

def test_solver_pressure_anchoring(state_solver):
    """Verifies Rule 5: Dynamic anchoring at the first fluid cell."""
    from src.step3.solver import solve_pressure
    
    # Set a specific reference pressure using dot notation (enabled by AttributeDict)
    state_solver.config.simulation_parameters.initial_pressure = 101325.0
    
    solve_pressure(state_solver)
    
    # In F-order, the first cell [0,0,0] is flat index 0
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
    
    # Force a numerical failure that SciPy cannot call '0' (success)
    # Injecting NaN into the RHS is the most reliable way to force info > 0
    state_solver.operators._divergence = sparse.csr_matrix(np.ones((27, 81)) * np.nan)
    state_solver.fields.U_star += 1.0 
    
    status = solve_pressure(state_solver)
    
    # Now status will be "failed" because info will be < 0 or > 0 (NaN breakdown)
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