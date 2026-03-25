# tests/common/test_solver_state.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.common.solver_state import (
    BoundaryCondition,
    BoundaryConditionManager,
    GridManager,
    PhysicalConstraintsManager,
    SolverState,
    verify_foundation_integrity,
)


class TestSolverStateFoundation:

    @pytest.fixture
    def populated_state(self):
        """Creates a state with an allocated foundation for testing."""
        state = SolverState()
        
        # Grid Setup: Real Instance required to satisfy _set_safe validation
        gm = GridManager()
        gm.nx, gm.ny, gm.nz = 2, 2, 2
        gm.x_min, gm.x_max = 0.0, 1.0
        gm.y_min, gm.y_max = 0.0, 1.0
        gm.z_min, gm.z_max = 0.0, 1.0
        state.grid = gm
        
        state.fields.allocate(8) # 2x2x2 cells
        return state

    # --- 1. MEMORY INTEGRITY (POST) ---

    def test_verify_foundation_integrity_success(self, populated_state, caplog):
        """Verifies that a healthy state passes the pre-flight check."""
        caplog.set_level(logging.INFO)
        
        class MockBlock:
            def __init__(self, idx):
                self.center = type('obj', (object,), {
                    'is_ghost': False, 
                    'index': idx, 
                    'p': float(idx) + (float(FI.P) / 10.0),
                    'u': [float(idx) + (float(FI.VX) / 10.0), 0, 0]
                })
        
        populated_state.stencil_matrix = [MockBlock(i) for i in range(8)]
        verify_foundation_integrity(populated_state)
        assert "Memory integrity verified" in caplog.text

    def test_verify_foundation_integrity_failure_drift(self, populated_state, caplog):
        """Verifies that the POST catches memory drift/misalignment."""
        caplog.set_level(logging.CRITICAL)
        
        # Create a block with WRONG pointer data (Simulating Memory Drift)
        class CorruptBlock:
            def __init__(self):
                self.center = type('obj', (object,), {
                    'is_ghost': False, 'index': 0, 'p': -999.9, 'u': [0, 0, 0]
                })
        
        populated_state.stencil_matrix = [CorruptBlock()]
        
        with pytest.raises(RuntimeError, match="Memory Swap"):
            verify_foundation_integrity(populated_state)
        assert "MEMORY DRIFT [P]" in caplog.text

    # --- 2. ANTI-FRANKENSTEIN PROTOCOL (RULE 9) ---

    def test_rollback_recovery(self, populated_state, caplog):
        """Ensures rollback wipes 'numerical pollution' and restores stability."""
        caplog.set_level(logging.WARNING)
        populated_state.capture_stable_state()
        populated_state.fields.data[:] = 144.93
        populated_state.rollback_to_stable_state()
        assert np.all(populated_state.fields.data == 0)
        assert "Memory reverted to state" in caplog.text

    # --- 3. VECTORIZED PHYSICAL AUDIT (RULE 7) ---

    def test_audit_velocity_corridor_violation(self, populated_state, caplog):
        """Triggers the 'STABILITY TRIGGER' when velocity exceeds limits."""
        caplog.set_level(logging.ERROR)
        
        pc = PhysicalConstraintsManager()
        pc.max_velocity = 10.0
        pc.min_velocity = -10.0
        pc.min_pressure = -1e5
        pc.max_pressure = 1e5
        populated_state.physical_constraints = pc
        
        populated_state.fields.data[0, FI.VX] = 50.0 
        
        with pytest.raises(ArithmeticError, match="Velocity Corridor Violation"):
            populated_state.audit_physical_bounds()

    def test_audit_pressure_reconstruction_and_bounds(self, populated_state, caplog):
        """Verifies Real-Pressure reconstruction triggers on out-of-bounds results."""
        caplog.set_level(logging.ERROR)
        
        bc = BoundaryCondition()
        bc.location = "x_min"
        bc.type = "pressure"
        bc.values = {"p": 101325.0}
        
        bcm = BoundaryConditionManager()
        bcm.conditions = [bc]
        populated_state.boundary_conditions = bcm
        
        pc = PhysicalConstraintsManager()
        pc.max_pressure = 200000.0
        pc.min_pressure = 50000.0
        pc.max_velocity = 1000.0
        pc.min_velocity = -1000.0
        populated_state.physical_constraints = pc
        
        populated_state.fields.data[:, FI.P_NEXT] = 10_000_000.0
        
        with pytest.raises(ArithmeticError, match="Pressure Corridor Violation"):
            populated_state.audit_physical_bounds()

    # --- 4. STATE TRANSITIONS ---

    def test_ready_for_time_loop_trigger(self, populated_state):
        """Verifies that setting ready_for_time_loop=True triggers integrity checks."""
        # This fails because stencil_matrix is still None at this point in the fixture
        with pytest.raises(RuntimeError):
            populated_state.ready_for_time_loop = True