# tests/property_integrity/test_step1_integrity.py

import numpy as np
import pytest

from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input

class TestStep1Integrity:
    """AUDITOR: Step 1 Structural Gate."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """
        Fixture that provides both the initialized State and the Context
        to maintain SSoT integrity throughout the test lifecycle.
        """
        # Rule 5: Explicit initialization
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        
        # Rule 5: Explicit configuration injection
        config = SolverConfig()
        config.ppe_tolerance = 1e-6
        config.ppe_atol = 1e-10
        config.ppe_max_iter = 1000
        config.ppe_omega = 1.5
        
        context = SimulationContext(input_data=input_data, config=config)
        state = orchestrate_step1(context)
        
        return state, context

    def test_departmental_containers(self, setup_data):
        """Rule 4: Validates existence of required sub-containers."""
        state, _ = setup_data
        assert hasattr(state, "grid"), "Missing GridManager"
        assert hasattr(state, "fields"), "Missing FieldManager"
        assert hasattr(state, "masks"), "Missing MaskManager"
        # Note: 'config' is in context, not necessarily inside state (Rule 4).

    def test_no_convenience_leaks(self, setup_data):
        """Rule 4: Ensures no convenience aliases exist on root."""
        state, _ = setup_data
        forbidden = ["nx", "ny", "nz", "dt", "density", "ppe_tolerance"]
        for alias in forbidden:
            assert not hasattr(state, alias), f"Rule 4 Violation: Alias '{alias}' found on state root."

    def test_foundation_integrity(self, setup_data):
        """Rule 1 & 9: Verifies FieldManager foundation allocation."""
        state, _ = setup_data
        assert state.fields.data.dtype == np.float32, "Field buffer must be float32"
        assert state.fields.data.size > 0, "Foundation memory allocation empty"
        assert isinstance(state.fields.U, np.ndarray), "Velocity field is not an array"
        assert state.fields.P.ndim == 3, "Pressure field is not 3D"

    def test_explicit_parameter_ingestion(self, setup_data):
        """Rule 5: Ensures tuning parameters were explicitly ingested."""
        _, context = setup_data
        assert context.config.ppe_tolerance == 1e-6

    def test_scale_guard_memory_architecture(self, setup_data):
        """Rule 0: Scale Guard (Memory Locality)."""
        state, _ = setup_data
        assert isinstance(state.fields.data, np.ndarray), "Foundation must be a NumPy array."
        assert state.fields.data.flags['C_CONTIGUOUS'], "Memory foundation must be C-contiguous."
        expected_shape = (64, state.fields.data.shape[1])
        assert state.fields.data.shape == expected_shape, f"Expected {expected_shape}, got {state.fields.data.shape}"