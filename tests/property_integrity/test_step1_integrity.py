# tests/property_integrity/test_step1_integrity.py

import numpy as np
import pytest

from src.step1.orchestrate_step1 import orchestrate_step1

# Rule 8: Explicit import of the primary interface
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep1Integrity:
    """AUDITOR: Step 1 Structural Gate."""

    @pytest.fixture(scope="class")
    def state(self):
        # Rule 5: Explicit initialization (No hardcoded defaults)
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        
        # Rule 5: Explicit injection of tuning parameters
        input_data.solver_settings = {
            "ppe_tolerance": 1e-6,
            "ppe_atol": 1e-10,
            "ppe_max_iter": 1000,
            "ppe_omega": 1.5
        }
        
        # Rule 5: Passing the hydrated object directly
        return orchestrate_step1(input_data)

    def test_departmental_containers(self, state):
        """Rule 4: Validates existence of required sub-containers."""
        assert hasattr(state, "grid"), "Missing GridManager"
        assert hasattr(state, "config"), "Missing SimulationParameterManager"
        assert hasattr(state, "fields"), "Missing FieldManager"
        assert hasattr(state, "masks"), "Missing MaskManager"

    def test_no_convenience_leaks(self, state):
        """Rule 4: Ensures no convenience aliases (shortcuts) exist on root."""
        forbidden = ["nx", "ny", "nz", "dt", "density", "ppe_tolerance"]
        for alias in forbidden:
            assert not hasattr(state, alias), f"Rule 4 Violation: Alias '{alias}' found on state root."

    def test_foundation_integrity(self, state):
        """Rule 1 & 9: Verifies FieldManager foundation allocation."""
        assert state.fields.data.dtype == np.float32, "Field buffer must be float32"
        assert state.fields.data.size > 0, "Foundation memory allocation empty"
        assert isinstance(state.fields.U, np.ndarray), "Velocity field is not an array"
        assert state.fields.P.ndim == 3, "Pressure field is not 3D"

    def test_explicit_parameter_ingestion(self, state):
        """Rule 5: Ensures tuning parameters were explicitly ingested."""
        assert state.config.ppe_tolerance == 1e-6

    def test_scale_guard_memory_architecture(self, state):
        """
        Rule 0: Scale Guard.
        Ensures the field foundation is a contiguous NumPy array 
        and not a redundant object list.
        """
        # Verify the foundation is indeed a NumPy array (Rule 0)
        assert isinstance(state.fields.data, np.ndarray), \
            "Rule 0 Violation: Foundation must be a NumPy array, not a list."
        
        # Verify contiguous memory allocation for cache-locality (Rule 0)
        assert state.fields.data.flags['C_CONTIGUOUS'], \
            "Rule 0 Violation: Memory foundation must be C-contiguous."
            
        # Verify dimensionality (4x4x4 grid = 64 cells)
        expected_shape = (64, state.fields.data.shape[1])
        assert state.fields.data.shape == expected_shape, \
            f"Rule 0 Violation: Expected foundation shape {expected_shape}, got {state.fields.data.shape}"