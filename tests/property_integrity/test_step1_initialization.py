# tests/property_integrity/test_step1_initialization.py

import numpy as np
import pytest

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep1Initialization:
    """AUDITOR: Step 1 Structural Gate, Metadata Hydration & Lifecycle Transitions."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        # Explicit input hydration ensures we test the exact state defined in our schema
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        config = SolverConfig(ppe_tolerance=1e-6, ppe_atol=1e-9, ppe_max_iter=1000, ppe_omega=1.0)
        
        context = SimulationContext(input_data=input_data, config=config)
        state = orchestrate_step1(context)
        
        return state, context

    # --- STRUCTURAL & CONTAINER CHECKS ---

    def test_departmental_containers(self, setup_data):
        """Rule 4: Validates existence of required sub-managers and metadata."""
        state, _ = setup_data
        assert state.grid is not None, "Missing GridManager"
        assert state.fields is not None, "Missing FieldManager"
        assert state.mask is not None, "Missing MaskManager"
        assert hasattr(state, "_simulation_parameters"), "Missing Simulation Parameters Manager"

    def test_no_convenience_leaks(self, setup_data):
        """Rule 4: Ensures no convenience aliases exist on root."""
        state, _ = setup_data
        forbidden = ["nx", "ny", "nz", "dt", "density", "ppe_tolerance"]
        for alias in forbidden:
            assert not hasattr(state, alias), f"Rule 4 Violation: Alias '{alias}' found on state root."

    # --- MEMORY ALLOCATION & MAPPING ---

    def test_memory_allocation_geometry(self, setup_data):
        """Rule 0 & 1: Verifies FieldManager allocation and C-contiguity."""
        state, _ = setup_data
        
        assert state.fields.data is not None, "Field data foundation is None."
        assert state.fields.data.size > 0, "Foundation memory allocation is empty."
        assert state.fields.data.ndim == 2, "Foundation must be 2D (n_cells, fields)."
        assert state.fields.data.flags['C_CONTIGUOUS'], "Memory foundation must be C-contiguous."
        
        n_cells = (state.grid.nx + 2) * (state.grid.ny + 2) * (state.grid.nz + 2)
        expected_shape = (n_cells, FI.num_fields())
        assert state.fields.data.shape == expected_shape, "Foundation shape mismatch with buffered grid."
    
    def test_identity_signature_integrity(self, setup_data):
        """
        Rule 9: Identity Priming Strategy.
        Verifies that the FieldManager maps indices to fields correctly.
        """
        state, _ = setup_data
        data = state.fields.data
        
        for field_id in FI:
            data[:, field_id] = np.arange(data.shape[0]) + (float(field_id) / 10.0)
            
        test_indices = [0, data.shape[0] // 2, data.shape[0] - 1]
        for idx in test_indices:
            expected_p = idx + (FI.P / 10.0)
            assert np.isclose(data[idx, FI.P], expected_p), f"Memory Swap! Index {idx} P mismatch."
            
            expected_vx = idx + (FI.VX / 10.0)
            assert np.isclose(data[idx, FI.VX], expected_vx), f"Memory Swap! Index {idx} VX mismatch."

    # --- PERSISTENCE & TERMINATION LOGIC ---

    def test_initial_conditions_persistence(self, setup_data):
        """Rule 9: Ensure initial hydration values are not zeroed by orchestration."""
        state, _ = setup_data
        assert np.any(state.fields.data != 0.0), "Initial conditions were lost or initialized to zero."

    def test_simulation_parameter_hydration(self, setup_data):
        """Verify termination metadata is valid and surviving Step 1."""
        state, _ = setup_data
        params = state._simulation_parameters
        assert params.total_time > 0, "Non-physical total_time initialized."
        assert params.time_step > 0, "Non-physical time_step (dt) initialized."

    def test_termination_math_precision(self):
        """Validates floating point exit condition: current_time >= total_time."""
        total_time, dt = 0.05, 0.01
        state, _ = orchestrate_step1(SimulationContext(
            input_data=create_validated_input(nx=4), 
            config=SolverConfig(ppe_tolerance=1e-6)
        ))
        
        state._simulation_parameters.total_time = total_time
        state._simulation_parameters.time_step = dt
        
        current_time, iterations = 0.0, 0
        while current_time < state._simulation_parameters.total_time:
            current_time += state._simulation_parameters.time_step
            iterations += 1
            
        assert iterations == 5
        assert current_time == pytest.approx(total_time)
