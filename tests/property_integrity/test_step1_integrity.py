# tests/property_integrity/test_step1_integrity.py

import numpy as np
import pytest
from src.common.solver_input import SolverInput
from src.step1.orchestrate_step1 import orchestrate_step1

class TestStep1Integrity:
    """AUDITOR: Step 1 Structural Gate."""

    @pytest.fixture(scope="class")
    def state(self):
        # 1. Create the base object
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        
        # 2. Directly assign settings (Rule 5 compliance)
        input_data.solver_settings = {
            "ppe_tolerance": 1e-6,
            "ppe_atol": 1e-10,
            "ppe_max_iter": 1000,
            "ppe_omega": 1.5
        }
        
        # 3. Pass the object directly (No ** unpacking needed)
        return orchestrate_step1(input_data)

    def test_departmental_containers(self, state):
        assert hasattr(state, "grid"), "Missing GridManager"
        assert hasattr(state, "config"), "Missing SimulationParameterManager"
        assert hasattr(state, "fields"), "Missing FieldManager"
        assert hasattr(state, "masks"), "Missing MaskManager"

    def test_no_convenience_leaks(self, state):
        forbidden = ["nx", "ny", "nz", "dt", "density", "ppe_tolerance"]
        for alias in forbidden:
            assert not hasattr(state, alias), f"Rule 4 Violation: Alias '{alias}' found on state root."

    def test_foundation_integrity(self, state):
        assert state.fields.data.dtype == np.float32, "Field buffer must be float32"
        assert state.fields.data.size > 0, "Foundation memory allocation empty"
        assert isinstance(state.fields.U, np.ndarray), "Velocity field is not an array"
        assert state.fields.P.ndim == 3, "Pressure field is not 3D"

    def test_explicit_parameter_ingestion(self, state):
        # Accessing the validated config
        assert state.config.ppe_tolerance == 1e-6
