# tests/property_integrity/test_step1_initialization.py

import pytest

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep1Initialization:
    """AUDITOR: Step 1 Structural Gate & Initial Hydration Verification."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        # Explicit input hydration ensures we test the exact state defined in our schema
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        config = SolverConfig(ppe_tolerance=1e-6, ppe_atol=1e-9, ppe_max_iter=1000, ppe_omega=1.0)
        config.ppe_tolerance = 1e-6
        
        context = SimulationContext(input_data=input_data, config=config)
        state = orchestrate_step1(context)
        
        return state, context

    def test_departmental_containers(self, setup_data):
        """Rule 4: Validates existence of required sub-containers."""
        state, _ = setup_data
        assert state.grid is not None, "Missing GridManager"
        assert state.fields is not None, "Missing FieldManager"
        assert state.mask is not None, "Missing MaskManager"

    def test_no_convenience_leaks(self, setup_data):
        """Rule 4: Ensures no convenience aliases exist on root."""
        state, _ = setup_data
        forbidden = ["nx", "ny", "nz", "dt", "density", "ppe_tolerance"]
        for alias in forbidden:
            assert not hasattr(state, alias), f"Rule 4 Violation: Alias '{alias}' found on state root."

    def test_foundation_integrity(self, setup_data):
        """Rule 1 & 9: Verifies FieldManager foundation allocation."""
        state, _ = setup_data
        assert state.fields.data is not None
        assert state.fields.data.size > 0, "Foundation memory allocation empty"
        # Access data via index/schema, not by convenience aliases (Rule 8)
        assert state.fields.data.ndim == 2, "Foundation must be (n_cells, FI.num_fields())"

    def test_scale_guard_memory_architecture(self, setup_data):
        """Rule 0: Scale Guard (Memory Locality)."""
        state, _ = setup_data
        assert state.fields.data.flags['C_CONTIGUOUS'], "Memory foundation must be C-contiguous."
        n_cells = (state.grid.nx + 2) * (state.grid.ny + 2) * (state.grid.nz + 2)
        expected_shape = (n_cells, FI.num_fields())
        assert state.fields.data.shape == expected_shape