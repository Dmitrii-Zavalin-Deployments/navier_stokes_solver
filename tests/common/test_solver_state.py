# tests/common/test_solver_state.py

import pytest
import numpy as np
import logging
from src.common.solver_state import SolverState, PhysicalConstraintsManager, FieldManager
from src.common.field_schema import FI

def test_audit_catches_velocity_explosion(caplog):
    """
    VERIFICATION: Ensure the Audit identifies the 1e10 explosion and logs it.
    """
    state = SolverState()
    
    # 1. Setup Constraints
    pc = PhysicalConstraintsManager()
    pc.max_velocity = 1000.0  # Limit is 1k
    pc.min_pressure, pc.max_pressure = -1e5, 1e5
    state.physical_constraints = pc
    
    # 2. Setup Fields with 1e10 injection
    fm = FieldManager()
    fm.allocate(10)
    # Inject the "Ghost 1e10" into the VX_STAR field of the first cell
    fm.data[0, FI.VX_STAR] = 1e10 
    state.fields = fm

    # 3. Execute Audit and check logs
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ArithmeticError):
            state.audit_physical_bounds()

    # Assert Logger Evidence
    assert "AUDIT [Metric]: V_max observed: 1.0000e+10" in caplog.text
    assert "AUDIT [Explosion]: Velocity 1.0000e+10 > Limit 1000.0" in caplog.text

def test_audit_catches_nan(caplog):
    """VERIFICATION: Audit must catch NaNs immediately."""
    state = SolverState()
    pc = PhysicalConstraintsManager()
    pc.max_velocity = 100.0
    pc.min_pressure, pc.max_pressure = -10, 10
    state.physical_constraints = pc
    
    fm = FieldManager()
    fm.allocate(5)
    fm.data[2, FI.P] = np.nan
    state.fields = fm

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ArithmeticError):
            state.audit_physical_bounds()
            
    assert "AUDIT [Explosion]: Found 1 NaN/Inf values." in caplog.text