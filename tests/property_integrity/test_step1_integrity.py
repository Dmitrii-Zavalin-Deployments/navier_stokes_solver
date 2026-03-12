# tests/property_integrity/test_step1_integrity.py

import numpy as np
import pytest

from src.common.solver_input import SolverInput
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers import solver_input_schema_dummy


class TestStep1Integrity:
    """
    AUDITOR: Step 1 Structural Gate.
    Verifies that 'orchestrate_step1' populates the SolverState 
    in strict compliance with Phase C Architectural Mandates.
    """

    @pytest.fixture(scope="class")
    def state(self):
        """The Live Pipeline Instance for this Audit."""
        dummy = solver_input_schema_dummy()
        # Rule 5: Explicit initialization
        dummy["solver_settings"] = {
            "ppe_tolerance": 1e-6,
            "ppe_atol": 1e-10,
            "ppe_max_iter": 1000,
            "ppe_omega": 1.5
        }
        return orchestrate_step1(SolverInput(**dummy))

    # =========================================================================
    # AUDIT: SSoT Hierarchy (Rule 4)
    # =========================================================================

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

    # =========================================================================
    # AUDIT: Hybrid Memory Foundation (Rule 1, 9)
    # =========================================================================

    def test_foundation_integrity(self, state):
        """Rule 1 & 9: Verifies FieldManager foundation allocation."""
        # Check buffer type (Rule 1)
        assert state.fields.data.dtype == np.float32, "Field buffer must be float32"
        # Check that allocation was actually performed
        assert state.fields.data.size > 0, "Foundation memory allocation empty"
        # Verify MAC grid structure
        assert isinstance(state.fields.U, np.ndarray), "Velocity field is not an array"
        assert state.fields.P.ndim == 3, "Pressure field is not 3D"

    # =========================================================================
    # AUDIT: Deterministic Initialization (Rule 5)
    # =========================================================================

    def test_explicit_parameter_ingestion(self, state):
        """Rule 5: Ensures tuning parameters were explicitly ingested."""
        # Verify that specific config keys exist in the container
        # Note: Accessing via the correct sub-container
        assert state.sim_params.time_step > 0
        assert hasattr(state.boundary_conditions, "lookup_table")