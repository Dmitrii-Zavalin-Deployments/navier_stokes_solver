# tests/step1/test_orchestrate_step1.py

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1

def test_orchestrate_step1_debug_path():
    """
    Targets lines 35, 102-103 in src/step1/orchestrate_step1.py.
    Forces the DEBUG constant to True to verify logging logic.
    """
    # 1. Setup a comprehensive Mock SimulationContext
    mock_context = MagicMock()
    input_data = mock_context.input_data
    
    # Grid Setup
    input_data.grid.nx, input_data.grid.ny, input_data.grid.nz = 2, 2, 2
    input_data.grid.x_min, input_data.grid.x_max = 0.0, 1.0
    input_data.grid.y_min, input_data.grid.y_max = 0.0, 1.0
    input_data.grid.z_min, input_data.grid.z_max = 0.0, 1.0
    
    # Domain & Properties
    input_data.domain_configuration.type = "INTERNAL"
    
    # IMPORTANT: Prevent MagicMock from hallucinating the reference velocity attribute
    # which triggers the TypeError in the setter.
    del input_data.domain_configuration._reference_velocity
    
    input_data.fluid_properties.density = 1.0
    input_data.fluid_properties.viscosity = 0.01
    
    # Forces & Constraints
    input_data.external_forces.force_vector = [0.0, -9.81, 0.0]
    input_data.physical_constraints.min_velocity = -10.0
    input_data.physical_constraints.max_velocity = 10.0
    input_data.physical_constraints.min_pressure = 0.0
    input_data.physical_constraints.max_pressure = 100.0
    
    # Initial Conditions & Parameters
    input_data.initial_conditions.velocity = [0.0, 0.0, 0.0]
    input_data.initial_conditions.pressure = 101325.0
    input_data.simulation_parameters.time_step = 0.01
    input_data.simulation_parameters.total_time = 1.0
    input_data.simulation_parameters.output_interval = 10
    
    # Topology (8 elements for 2x2x2)
    input_data.mask.data = [1] * 8
    
    # Boundary Conditions
    bc_item = MagicMock()
    bc_item.location = "left"
    bc_item.type = "noslip"
    bc_item.values = [0.0, 0.0, 0.0]
    input_data.boundary_conditions.items = [bc_item]

    # 2. Patch DEBUG and run orchestration
    with patch("src.step1.orchestrate_step1.DEBUG", True):
        state = orchestrate_step1(mock_context)
        
        # 3. Basic Integrity Checks
        assert state is not None
        assert state.fields.data.shape[0] == 64
        assert state.grid.nx == 2

def test_orchestrate_step1_reference_velocity_branch():
    """
    Targets lines 51-52: Ensure reference_velocity is assigned if explicitly present.
    """
    mock_context = MagicMock()
    input_data = mock_context.input_data
    
    # Provide a VALID 3D vector to satisfy the setter's size check
    ref_vel = [5.0, 0.0, 0.0]
    input_data.domain_configuration._reference_velocity = ref_vel
    input_data.domain_configuration.reference_velocity = ref_vel
    input_data.domain_configuration.type = "EXTERNAL"

    # Minimal hydration
    input_data.grid.nx, input_data.grid.ny, input_data.grid.nz = 1, 1, 1
    input_data.grid.x_min = input_data.grid.x_max = 0.0
    input_data.grid.y_min = input_data.grid.y_max = 0.0
    input_data.grid.z_min = input_data.grid.z_max = 0.0
    input_data.fluid_properties.density = 1.225
    input_data.fluid_properties.viscosity = 1.8e-5
    input_data.external_forces.force_vector = [0.0, 0.0, 0.0]
    input_data.physical_constraints.min_velocity = -100.0
    input_data.physical_constraints.max_velocity = 100.0
    input_data.physical_constraints.min_pressure = 0.0
    input_data.physical_constraints.max_pressure = 1e6
    input_data.initial_conditions.velocity = [0.0, 0.0, 0.0]
    input_data.initial_conditions.pressure = 101325.0
    input_data.simulation_parameters.time_step = 0.1
    input_data.simulation_parameters.total_time = 1.0
    input_data.simulation_parameters.output_interval = 1
    input_data.mask.data = [1]
    input_data.boundary_conditions.items = []

    state = orchestrate_step1(mock_context)
    
    assert isinstance(state.domain_configuration.reference_velocity, np.ndarray)
    assert state.domain_configuration.reference_velocity.size == 3