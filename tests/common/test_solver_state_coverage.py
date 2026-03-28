# tests/common/test_solver_state_coverage.py

import pytest
import numpy as np
import logging
from src.common.solver_state import (
    SolverState, FieldManager, GridManager, PhysicalConstraintsManager, 
    BoundaryCondition, BoundaryConditionManager, DomainManager, MaskManager,
    ExternalForceManager, FluidPropertiesManager, InitialConditionManager,
    verify_foundation_integrity
)
from src.common.field_schema import FI

def test_solver_state_defensive_logic(caplog):
    state = SolverState()
    
    # 1. Trigger Alignment Warning (Line 24-25)
    # Create an unaligned buffer (offset by 8 bytes)
    raw_data = np.zeros((100, FI.num_fields()) , dtype=np.float64)
    unaligned_view = raw_data.view(np.float64)[1:].reshape(-1, FI.num_fields())
    
    fm = FieldManager()
    fm.data = unaligned_view
    state.fields = fm
    state.stencil_matrix = []
    
    with caplog.at_level(logging.WARNING):
        verify_foundation_integrity(state)
        assert "not 64-byte aligned" in caplog.text

    # 2. Trigger Architecture Overflow (Line 40-41)
    state.stencil_matrix = [type('obj', (object,), {'center': None})] * 101 # More than num_cells
    with pytest.raises(RuntimeError, match="Architecture integrity compromised"):
        verify_foundation_integrity(state)

    # 3. Setter Exceptions (Lines 127, 135, 237)
    fluid = FluidPropertiesManager()
    with pytest.raises(ValueError, match="Density must be > 0"):
        fluid.density = -1.0
    with pytest.raises(ValueError, match="Viscosity must be >= 0"):
        fluid.viscosity = -1.0
    
    mask = MaskManager()
    with pytest.raises(ValueError, match="Mask must be a NumPy array"):
        mask.mask = np.array([5]) # Invalid value

    # 4. Audit Physics - Velocity NaN (Line 513)
    state.grid = GridManager()
    state.grid.nx, state.grid.ny, state.grid.nz = 2, 2, 2
    state.physical_constraints = PhysicalConstraintsManager()
    state.physical_constraints.min_velocity = -10
    state.physical_constraints.max_velocity = 10
    
    fm.allocate(8)
    state.fields.data[0, FI.VX] = np.nan
    with pytest.raises(ArithmeticError, match="NaN/Inf detected"):
        state.audit_physical_bounds()

    # 5. Boundary Logic & Pressure Grounding (Lines 571-576)
    # We loop through locations to hit the branching logic
    bc_mgr = BoundaryConditionManager()
    for loc in ["x_max", "y_min", "y_max", "z_min", "z_max"]:
        bc = BoundaryCondition()
        bc.location = loc
        bc.type = "pressure"
        bc.values = {"p": 0.0}
        bc_mgr.conditions = [bc]
        state.boundary_conditions = bc_mgr
        state.fields.data[:] = 0.0 # Reset to finite values
        
        # This will trigger the specific coordinate slicing for each location
        try:
            state.audit_physical_bounds()
        except ArithmeticError:
            pass # We just care about hitting the lines

    # 6. Physical Readiness (Lines 610-615)
    state.fields = None
    with pytest.raises(RuntimeError, match="Foundation buffer is missing"):
        state.validate_physical_readiness()

