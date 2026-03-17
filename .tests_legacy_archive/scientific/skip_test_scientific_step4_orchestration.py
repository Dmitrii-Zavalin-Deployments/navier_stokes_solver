# tests/scientific/test_scientific_step4_orchestration.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step4.orchestrate_step4 import orchestrate_step4


class MockDiagnostics(dict):
    """Allows dot notation access for diagnostics."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_pre_orchestration():
    """Sets up a physically and numerically complete state for audit validation."""
    state = SolverState()
    
    # 1. Grid Definition (for spatial steps dx, dy, dz)
    state.grid._nx, state.grid._ny, state.grid._nz = 3, 3, 3
    state.grid._x_min, state.grid._x_max = 0.0, 1.0
    state.grid._y_min, state.grid._y_max = 0.0, 1.0
    state.grid._z_min, state.grid._z_max = 0.0, 1.0
    
    # 2. Simulation Parameters (The 'requested' time step)
    # We set it to 0.01, which is safely below the 0.166 CFL limit seen in logs
    state.config._simulation_parameters = {
        "time_step": 0.01,
        "max_iterations": 100
    }
    
    # 3. Fluid Properties (for Diffusion/Viscosity audit)
    state.config._fluid_properties = {
        "viscosity": 0.01,
        "density": 1.0
    }
    
    # 4. Health & Fields (for CFL max_u audit)
    state.health._max_u = 1.0
    state.health._is_stable = True
    state.time = 0.5
    
    state.fields.P = np.ones((3, 3, 3))
    state.fields.U = np.ones((4, 3, 3))
    state.fields.V = np.ones((3, 4, 3))
    state.fields.W = np.ones((3, 3, 4))
    
    # 5. Boundary Conditions Lookup
    state.bc_lookup = {
        "x_min": {"type": "wall"}, "x_max": {"type": "wall"},
        "y_min": {"type": "wall"}, "y_max": {"type": "wall"},
        "z_min": {"type": "wall"}, "z_max": {"type": "wall"}
    }
    
    # 6. Diagnostics Mock
    state.diagnostics = MockDiagnostics()
    state.diagnostics.bc_verification_passed = False
    state.diagnostics.initial_cfl_dt = 0.0
    state.ready_for_time_loop = False
    
    return state

def test_orchestration_state_progression(state_pre_orchestration):
    """
    Scientific check: Verifies that the orchestrator flips the 
    ready_for_time_loop bit and sets diagnostics.
    """
    # Execute the orchestrator
    updated_state = orchestrate_step4(state_pre_orchestration)
    
    # 1. Check the 'Handshake' bit
    assert updated_state.ready_for_time_loop is True
    
    # 2. Check that sub-steps were actually called (via their output flags)
    assert updated_state.diagnostics.bc_verification_passed is True
    
    # 3. Verify fields were actually extended (Allocation check)
    assert updated_state.fields.P_ext.shape == (5, 5, 5)

def test_orchestration_debug_handshake(state_pre_orchestration, capsys):
    """
    Scientific check: Verifies all log signals for the 'Handshake' logic.
    """
    orchestrate_step4(state_pre_orchestration)
    captured = capsys.readouterr().out
    
    # Check for specific Step 4 Orchestrator signals
    assert "DEBUG [Step 4 Orchestrator]: Finalizing Step 4" in captured
    assert "Time: 0.5" in captured
    assert "Boundaries synchronized" in captured
    assert "HANDSHAKE SUCCESSFUL" in captured

def test_orchestration_cfl_reporting(state_pre_orchestration, capsys):
    """
    Scientific check: Verifies that the CFL limit calculation 
    from audit_diagnostics is correctly passed through and logged.
    """
    # Note: run_preflight_audit is expected to set initial_cfl_dt
    # Since we are using the real solver state, we verify the orchestrator prints it.
    orchestrate_step4(state_pre_orchestration)
    captured = capsys.readouterr().out
    
    # Check that the CFL signal is present (formula: dt_cfl = dx / |u_max|)
    assert "Calculated CFL limit:" in captured

def test_orchestration_failure_blocking(state_pre_orchestration):
    """
    Scientific check: Verifies that if a sub-step fails (RuntimeError), 
    the state is NEVER marked ready for the time loop.
    """
    # Corrupt the state to trigger a RuntimeError in fill_ghost_boundaries
    # by removing a required boundary face
    del state_pre_orchestration.bc_lookup["x_min"]
    
    with pytest.raises(RuntimeError):
        orchestrate_step4(state_pre_orchestration)
        
    # Crucially, the bit should still be False because the orchestrator crashed
    assert state_pre_orchestration.ready_for_time_loop is False