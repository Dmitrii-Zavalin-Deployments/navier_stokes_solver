# tests/common/test_solver_state.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.common.solver_state import SolverState, verify_foundation_integrity


class TestSolverStateFoundation:

    @pytest.fixture
    def populated_state(self):
        """Creates a state with an allocated foundation for testing."""
        state = SolverState()
        # Mock minimal dependencies
        class MockManager:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)
            def to_dict(self): return {}

        state.grid = MockManager(nx=2, ny=2, nz=2)
        state.fields.allocate(8) # 2x2x2
        return state

    # --- 1. MEMORY INTEGRITY (POST) ---

    def test_verify_foundation_integrity_success(self, populated_state, caplog):
        """Verifies that a healthy state passes the pre-flight check."""
        caplog.set_level(logging.INFO)
        
        # We need a dummy stencil matrix to satisfy the loop
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
        
        # Create a block with WRONG pointer data (Memory Drift)
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
        
        # 1. Capture stable state (all zeros)
        populated_state.capture_stable_state()
        
        # 2. Pollute the foundation (Simulate a numerical explosion)
        populated_state.fields.data[:] = 144.93
        
        # 3. Rollback
        populated_state.rollback_to_stable_state()
        
        # 4. Verify purity
        assert np.all(populated_state.fields.data == 0)
        assert "Memory reverted to state" in caplog.text

    # --- 3. VECTORIZED PHYSICAL AUDIT (RULE 7) ---

    def test_audit_velocity_corridor_violation(self, populated_state, caplog):
        """Triggers the 'STABILITY TRIGGER' when velocity exceeds limits."""
        caplog.set_level(logging.ERROR)
        
        class MockConstraints:
            max_velocity = 10.0
            min_velocity = -10.0
            def to_dict(self): return {}

        populated_state.physical_constraints = MockConstraints()
        # Set an unphysical velocity
        populated_state.fields.data[0, FI.VX] = 50.0 
        
        with pytest.raises(ArithmeticError, match="Velocity Corridor Violation"):
            populated_state.audit_physical_bounds()
        
        assert "AUDIT [Limit]: Velocity range" in caplog.text

    def test_audit_pressure_reconstruction_and_bounds(self, populated_state, caplog):
        """Verifies Real-Pressure reconstruction triggers on out-of-bounds results."""
        caplog.set_level(logging.ERROR)
        
        # Mock boundary conditions for pressure reference
        class MockBC:
            location = "x_min"
            type = "pressure"
            values = {"p": 101325.0} # Physical Atmosphere
        
        class MockBCMgr:
            conditions = [MockBC()]

        populated_state.boundary_conditions = MockBCMgr()
        populated_state.physical_constraints = type('obj', (object,), {
            'max_pressure': 200000.0, 'min_pressure': 50000.0
        })
        
        # Set trial pressure (P_NEXT) to something that will result in unphysical P_REAL
        # If P_REF_PHYSICAL is 101k, and we set trial to 10M, it should trigger
        populated_state.fields.data[:, FI.P_NEXT] = 10_000_000.0
        
        with pytest.raises(ArithmeticError, match="Pressure Corridor Violation"):
            populated_state.audit_physical_bounds()
        
        assert "AUDIT [Explosion]: Real pressure" in caplog.text

    # --- 4. DATA TYPES & SERIALIZATION ---

    def test_ready_for_time_loop_trigger(self, populated_state):
        """Verifies that setting ready_for_time_loop=True triggers integrity checks."""
        # This should fail because we haven't set up the stencil matrix correctly yet
        with pytest.raises(RuntimeError):
            populated_state.ready_for_time_loop = True